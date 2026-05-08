"""
Bake a pyFAI azimuthal integrator into a self-contained CSR artifact for
repeated integration in another language (Julia, C++, …) without re-running
pyFAI's geometry layer per frame.

`npt=int` selects 1D (radial); `npt=(nrad, nazim)` selects 2D, with rows
packed in radial-major order: `row = bin_rad * nazim + bin_azim` (matches
pyFAI's `splitBBox_common` `cinsert(i*bins1 + j, ...)`).

Assumptions:
  * No variance / error propagation.
  * Dark current already subtracted.
  * Solid-angle, polarization, `radial_range`, `azimuth_range`, and the
    detector's static mask are frozen at bake time.
  * Empty bins come back as NaN. To match pyFAI, set `ai._empty = NaN`
    (its default is 0.0).

------------------------------------------------------------------------------
Math
------------------------------------------------------------------------------

pyFAI's per-bin reduction (see `engines/CSR_engine.py`, `engines/preproc.py`,
and the OpenCL `csr_integrate4` / `CSRxVec4` in
`resources/openCL/ocl_azim_CSR.cl`) is:

    sum_sig[b] = Σ_j A[b,j] * raw[j]                       (over valid j)
    sum_nrm[b] = Σ_j A[b,j] * corr[j]                      (over valid j)
    I[b]       = sum_sig[b] / sum_nrm[b]

with `corr[j] = solidangle[j] * polarization[j]` and "valid j" meaning
`isfinite(raw[j]) ∧ isfinite(corr[j])`. pyFAI rebalances the denominator
per frame over the finite pixels — that's what makes NaN/Inf in the input
safe.

We can't pre-divide by `sum_nrm` at bake time (it depends on which pixels
are finite per frame), so the bake ships two value arrays sharing one
`indices`/`indptr` structure:

    A_raw[b,j]  = A[b,j]
    A_corr[b,j] = A[b,j] * corr[j]

Per frame the consumer evaluates:

    m       = isfinite(x)
    x_clean = where(m, x, 0)
    S[b]    = Σ_j A_raw[b,j]  * x_clean[j]
    N[b]    = Σ_j A_corr[b,j] * m[j]
    I[b]    = S[b] / N[b]      # NaN where N == 0

The reference numpy `integrate` below leaves these as two unfused SpMVs;
the Julia consumer fuses them into a single matrix walk per frame. Empty
bins fall out as 0/0 → NaN. The detector's static mask is applied at bake
time (masked pixels never appear in `indices`).

For 2D the math is identical — same matrix, more rows. Reshape the flat
`(nbins,)` (or `(nbins, B)`) result to `(nrad, nazim)` in numpy or
`(nazim, nrad, …)` in Julia (column-major); both are zero-copy.

------------------------------------------------------------------------------
On-disk layout (HDF5, format_version = 2)
------------------------------------------------------------------------------

Datasets:
  data_raw     float32 (nnz,)      A_raw values
  data_corr    float32 (nnz,)      A_corr values (= A_raw * corr[col])
  indices      int32   (nnz,)      0-based column indices, C-order pixel index
  indptr       int32   (nbins+1,)  CSR row pointer
  bin_centers0 float32 (nbins0,)   radial axis, in `unit0`'s display scale
  bin_centers1 float32 (nbins1,)   azimuthal axis (only present if ndim == 2)

Attributes:
  shape           int64[2]  detector shape in pyFAI/C order (H, W)
  ndim            int       1 or 2
  unit0           str       pyFAI radial unit string
  unit1           str       pyFAI azimuthal unit string (only if ndim == 2)
  split           str       pixel-splitting scheme used at bake
  npt0            int       number of radial bins
  npt1            int       number of azimuthal bins (only if ndim == 2)
  format_version  int       2
"""

import argparse

import numpy as np
import h5py
import pyFAI
from scipy.sparse import csr_matrix
from pyFAI import units as pyFAI_units


FORMAT_VERSION = 1


def _split_npt_unit(npt, unit):
    """Normalize `npt`/`unit` into `(ndim, npt_arg, unit_arg, unit0, unit1_or_None)`."""
    if isinstance(npt, (tuple, list)) and len(npt) == 2:
        if isinstance(unit, (tuple, list)) and len(unit) == 2:
            unit0_str, unit1_str = str(unit[0]), str(unit[1])
        else:
            # pyFAI defaults the azimuthal axis to chi_deg when unit is scalar.
            unit0_str, unit1_str = str(unit), "chi_deg"
        return (2,
                (int(npt[0]), int(npt[1])),
                (unit0_str, unit1_str),
                unit0_str, unit1_str)
    return (1, int(npt), str(unit), str(unit), None)


def bake_for_batch(ai, npt, *,
                   unit="q_A^-1",
                   split="bbox",
                   solidangle=True,
                   polarization_factor=None,
                   radial_range=None,
                   azimuth_range=None):
    """Build a baked 1D or 2D integrator from an AzimuthalIntegrator.

    :param ai: a configured pyFAI.AzimuthalIntegrator
    :param npt: number of radial bins (int → 1D) or `(nrad, nazim)` tuple (2D)
    :param unit: pyFAI radial unit string for 1D, or `(radial, azimuthal)` tuple
        for 2D. A scalar unit for 2D defaults the azimuthal axis to `chi_deg`.
    :param split: pixel-splitting scheme: "no", "bbox", or "full"
    :param solidangle: if True, fold solid-angle correction into A_corr
    :param polarization_factor: polarization factor in [-1, 1], or None
    :param radial_range: optional (rmin, rmax) in `unit` (or `unit[0]`)
    :param azimuth_range: optional (chi_min, chi_max) in degrees
    :return: dict with `data_raw`, `data_corr`, `indices`, `indptr`,
        `bin_centers0`, `shape`, `unit0`, `split`, `npt0`, `ndim`, plus
        `bin_centers1`, `unit1`, `npt1` for 2D.
    """
    shape = ai.detector.shape
    npix = int(np.prod(shape))

    ndim, npt_arg, unit_arg, unit0_str, unit1_str = _split_npt_unit(npt, unit)

    detector_mask = ai.detector.mask
    # azimuth_range is in degrees; pos1_range expects radians (and the
    # discontinuity-aware shifted range).
    pos1_range = (ai.normalize_azimuth_range(azimuth_range)
                  if azimuth_range is not None else None)
    integ = ai.setup_sparse_integrator(
        shape=shape,
        npt=npt_arg,
        mask=detector_mask,
        pos0_range=radial_range,
        pos1_range=pos1_range,
        unit=unit_arg,
        split=split,
        algo="CSR",
        scale=False,
    )

    data_raw = np.asarray(integ.data, dtype=np.float32)
    indices = np.asarray(integ.indices, dtype=np.int32)
    indptr = np.asarray(integ.indptr, dtype=np.int32)

    # setup_sparse_integrator stores bin centers in S.I. units; rescale to
    # match integrateNd_ng's display axes.
    unit0_obj = pyFAI_units.to_unit(unit0_str)
    if ndim == 1:
        bin_centers0 = (np.asarray(integ.bin_centers, dtype=np.float32)
                        * np.float32(unit0_obj.scale))
        bin_centers1 = None
        npt0, npt1 = npt_arg, None
    else:
        unit1_obj = pyFAI_units.to_unit(unit1_str)
        bin_centers0 = (np.asarray(integ.bin_centers0, dtype=np.float32)
                        * np.float32(unit0_obj.scale))
        bin_centers1 = (np.asarray(integ.bin_centers1, dtype=np.float32)
                        * np.float32(unit1_obj.scale))
        npt0, npt1 = npt_arg

    corr = np.ones(npix, dtype=np.float32)
    if solidangle:
        # `True` (not `1`) selects the physical cos^3 correction; `1` would
        # give cos^1.
        sa = ai.solidAngleArray(shape, True).astype(np.float32, copy=False)
        corr *= sa.ravel()
    if polarization_factor is not None:
        pol = ai.polarization(shape=shape,
                              factor=float(polarization_factor)).astype(np.float32,
                                                                        copy=False)
        corr *= pol.ravel()

    # Consumer only checks isfinite(image), not isfinite(corr) — a non-finite
    # corr would silently corrupt N.
    if not np.all(np.isfinite(corr)):
        raise ValueError("non-finite values in baked correction array; "
                         "check solid-angle / polarization computation")

    data_corr = (data_raw * corr[indices]).astype(np.float32)

    out = {
        "data_raw": data_raw,
        "data_corr": data_corr,
        "indices": indices,
        "indptr": indptr,
        "bin_centers0": bin_centers0,
        "shape": tuple(int(s) for s in shape),
        "unit0": unit0_str,
        "split": str(split),
        "npt0": int(npt0),
        "ndim": int(ndim),
        # pyFAI's runtime preproc masks pixels with |raw - DUMMY| <= DELTA_DUMMY
        # per frame. The bake doesn't pre-apply this (it's data-dependent);
        # persist the values for the consumer.
        "dummy": np.float32(getattr(ai.detector, "DUMMY", np.nan)),
        "delta_dummy": np.float32(getattr(ai.detector, "DELTA_DUMMY", np.nan)),
    }
    if ndim == 2:
        out["bin_centers1"] = bin_centers1
        out["unit1"] = unit1_str
        out["npt1"] = int(npt1)
    return out


def write_hdf5(baked, path):
    """Write a baked dict to an HDF5 file."""
    with h5py.File(path, "w") as f:
        for key in ("data_raw", "data_corr", "indices", "indptr",
                    "bin_centers0"):
            f.create_dataset(key, data=baked[key], compression="gzip")
        if baked["ndim"] == 2:
            f.create_dataset("bin_centers1", data=baked["bin_centers1"],
                             compression="gzip")
        f.attrs["shape"] = np.asarray(baked["shape"], dtype=np.int64)
        f.attrs["ndim"] = baked["ndim"]
        f.attrs["unit0"] = baked["unit0"]
        f.attrs["split"] = baked["split"]
        f.attrs["npt0"] = baked["npt0"]
        if baked["ndim"] == 2:
            f.attrs["unit1"] = baked["unit1"]
            f.attrs["npt1"] = baked["npt1"]
        f.attrs["dummy"] = baked["dummy"]
        f.attrs["delta_dummy"] = baked["delta_dummy"]
        f.attrs["format_version"] = FORMAT_VERSION


def integrate(baked, image):
    """Reference numpy/scipy integration. Loops two SpMVs per frame
    (numerator and denominator); the Julia consumer fuses them.

    Per frame: ``I = (A_raw @ where(m, x, 0)) / (A_corr @ m)`` with
    ``m = isfinite(x)``. Empty bins and bins with no finite pixels come
    back as NaN. To match pyFAI's `integrate1d_ng` / `integrate2d_ng`,
    set `ai._empty = np.nan`.

    Output shape:
      * 1D, single frame    →  (nbins0,)
      * 1D, batch of B      →  (nbins0, B)
      * 2D, single frame    →  (nbins0, nbins1)
      * 2D, batch of B      →  (nbins0, nbins1, B)

    :param baked: the dict returned by `bake_for_batch`
    :param image: 2D detector image, OR 3D batch of shape (B, *shape)
    """
    nbins0 = int(baked["npt0"])
    nbins1 = int(baked.get("npt1", 1))
    nbins = nbins0 * nbins1 if baked["ndim"] == 2 else nbins0
    npix = int(np.prod(baked["shape"]))
    A_raw = csr_matrix((baked["data_raw"], baked["indices"], baked["indptr"]),
                       shape=(nbins, npix))
    A_corr = csr_matrix((baked["data_corr"], baked["indices"], baked["indptr"]),
                        shape=(nbins, npix))

    def _one(x):
        m = np.isfinite(x).astype(np.float32)
        x_clean = np.where(m > 0, x, np.float32(0))
        S = np.asarray(A_raw @ x_clean)
        N = np.asarray(A_corr @ m)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (S / N).astype(np.float32)

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        I = _one(arr.ravel())
        batch_shape = None
    elif arr.ndim == 3:
        B = arr.shape[0]
        I = np.empty((nbins, B), dtype=np.float32)
        flat = arr.reshape(B, -1)
        for i in range(B):
            I[:, i] = _one(flat[i])
        batch_shape = B
    else:
        raise ValueError(f"image must be 2D or 3D, got ndim={arr.ndim}")

    if baked["ndim"] == 2:
        if batch_shape is None:
            return I.reshape(nbins0, nbins1)
        return I.reshape(nbins0, nbins1, batch_shape)
    return I

def main():
    parser = argparse.ArgumentParser(
        description="Bake a pyFAI integrator from a PONI file to HDF5.")
    parser.add_argument("poni_path", help="Path to the PONI file")
    parser.add_argument("npt", type=int, help="Number of radial bins")
    parser.add_argument("output_path", help="Output HDF5 file path")
    args = parser.parse_args()

    ai = pyFAI.load(args.poni_path)
    baked = bake_for_batch(ai, args.npt)
    write_hdf5(baked, args.output_path)


if __name__ == "__main__":
    main()
