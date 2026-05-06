"""Regression test for bake_for_batch.

Compares the baked SpMM integration against pyFAI's reference
ai.integrate1d_ng for each of the three pixel-splitting modes.

Run with: ./run_tests.py -s julia_export/test_bake_for_batch.py
or:       python -m unittest julia_export.test_bake_for_batch
or just:  python julia_export/test_bake_for_batch.py

Optional Julia-side regression:
    PYFAI_JULIA_PROJECT=/path/to/env python julia_export/test_bake_for_batch.py

Runs `julia --project=$PYFAI_JULIA_PROJECT` against `integrate.jl`, comparing
its output to the Python reference. Requires `julia` on PATH plus HDF5.jl and
DimensionalData.jl in the given project. Skipped if the env var is unset.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from bake_for_batch import bake_for_batch, write_hdf5, integrate

# pyFAI in-tree imports — assume bootstrap or installed pyFAI.
from pyFAI.test.utilstest import create_fake_data


NPT = 800
UNIT = "2th_deg"

# When set, every _check_split run is also executed via the Julia consumer
# script and the result compared against the Python reference.
JULIA_PROJECT = os.environ.get("PYFAI_JULIA_PROJECT")
JULIA_AVAILABLE = bool(JULIA_PROJECT) and shutil.which("julia") is not None

def _julia_integrate(baked, image, *, frame_shape):
    """Run `integrate.jl` against `baked` + `image` (a single frame or a
    stack), returning the result as a numpy array in the same layout as
    `bake_for_batch.integrate`:
        1D, single frame  →  (nbins0,)
        1D, batch of B    →  (nbins0, B)
        2D, single frame  →  (nbins0, nbins1)
        2D, batch of B    →  (nbins0, nbins1, B)
    HDF5.jl reverses axes on every read/write, so the Julia-side shape on
    disk is the reverse of what we want in numpy."""
    import h5py

    is_stack = image.ndim == len(frame_shape) + 1
    ndim = baked["ndim"]
    with tempfile.TemporaryDirectory() as td:
        setup_path = os.path.join(td, "setup.h5")
        input_path = os.path.join(td, "input.h5")
        output_path = os.path.join(td, "output.h5")
        script_path = os.path.join(td, "driver.jl")

        write_hdf5(baked, setup_path)

        # The Julia consumer expects a (W, H) frame (= transpose of pyFAI's
        # (H, W) layout) so that column-major `vec` matches pyFAI's C-order
        # flat index. HDF5.jl reverses axes on read, so writing the numpy
        # arrays unchanged lands a Julia (W, H) frame / (W, H, B) stack —
        # i.e. Julia sees the transpose of the numpy frame, which is exactly
        # what the new BakedIntegrator convention requires.
        with h5py.File(input_path, "w") as f:
            f.create_dataset("img", data=image)

        # include(\"{os.path.join(HERE, "integrate.jl")}\")
        driver = textwrap.dedent(f"""
            using HDF5
            using QSpaceTools: load_baked, integrate

            b = load_baked(\"{setup_path}\")
            img = h5open(\"{input_path}\", "r") do f; read(f["img"]); end
            y = parent(integrate(b, img))
            h5open(\"{output_path}\", "w") do f
                f["y"] = y
            end
        """)
        with open(script_path, "w") as f:
            f.write(driver)

        subprocess.run(
            ["julia", "--startup-file=no", f"--project={JULIA_PROJECT}", script_path],
            check=True,
        )

        with h5py.File(output_path, "r") as f:
            y = f["y"][...]
        # Convert HDF5.jl's reversed-axis on-disk layout back to the numpy
        # convention used by `bake_for_batch.integrate`. The `parent(...)` of
        # the Julia DimArray has these shapes (Julia order):
        #   1D single:  (nbins0,)            → numpy (nbins0,)
        #   1D batch:   (nbins0, B)          → numpy (B, nbins0)        → .T
        #   2D single:  (nbins1, nbins0)     → numpy (nbins0, nbins1)
        #   2D batch:   (nbins1, nbins0, B)  → numpy (B, nbins0, nbins1)→ moveaxis
        if ndim == 1:
            return y.T if is_stack else y
        # ndim == 2
        if is_stack:
            return np.moveaxis(y, 0, -1)  # (B, nbins0, nbins1) → (nbins0, nbins1, B)
        return y


class TestBakeForBatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image, cls.ai = create_fake_data(poissonian=False)
        cls.image = cls.image.astype(np.float32)
        # pyFAI's CSR engine snapshots its empty-bin sentinel from ai._empty
        # at creation time (the `dummy=` kwarg only controls the *input*
        # sentinel). The exporter always emits NaN for empty bins, so align
        # pyFAI here for genuinely-empty bins (azimuth_range / radial_range
        # cuts) to match.
        cls.ai._empty = np.float32(np.nan)

    def _check_split(self, split, *, with_pol=False, azimuth_range=None):
        ai = self.ai
        ai.reset()

        pol_factor = 0.97 if with_pol else None

        method = (split, "csr", "cython")
        ref_kwargs = dict(npt=NPT, unit=UNIT, method=method,
                          correctSolidAngle=True, dummy=np.nan)
        if pol_factor is not None:
            ref_kwargs["polarization_factor"] = pol_factor
        if azimuth_range is not None:
            ref_kwargs["azimuth_range"] = azimuth_range

        ref = ai.integrate1d_ng(self.image, **ref_kwargs)
        ref_q, ref_I = ref.radial, ref.intensity

        ai.reset()
        baked = bake_for_batch(ai, npt=NPT, unit=UNIT, split=split,
                               solidangle=True,
                               polarization_factor=pol_factor,
                               azimuth_range=azimuth_range)

        # bin centers must match
        np.testing.assert_allclose(baked["bin_centers0"], ref_q, rtol=1e-6,
                                   err_msg=f"bin centers differ for split={split}")

        got_I = integrate(baked, self.image)

        nan_mask = np.isnan(got_I)
        np.testing.assert_array_equal(
            nan_mask, np.isnan(ref_I),
            err_msg=f"empty-bin pattern differs for split={split}")
        np.testing.assert_allclose(got_I[~nan_mask], ref_I[~nan_mask],
                                   rtol=1e-4, atol=1e-4,
                                   err_msg=f"intensity mismatch for split={split}")

        if JULIA_AVAILABLE:
            jl_I = _julia_integrate(baked, self.image,
                                    frame_shape=self.image.shape)
            np.testing.assert_array_equal(
                np.isnan(jl_I), nan_mask,
                err_msg=f"Julia NaN pattern differs for split={split}")
            np.testing.assert_allclose(
                jl_I[~nan_mask], got_I[~nan_mask],
                rtol=1e-5, atol=1e-5,
                err_msg=f"Julia ≠ Python integrate for split={split}")

    def test_split_no(self):
        self._check_split("no")

    def test_split_bbox(self):
        self._check_split("bbox")

    def test_split_full(self):
        self._check_split("full")

    def test_with_polarization_bbox(self):
        self._check_split("bbox", with_pol=True)

    def test_with_azimuth_range_bbox(self):
        self._check_split("bbox", azimuth_range=(-45.0, 45.0))

    def test_batch_matches_loop(self):
        """A 3D batch should give the same result as integrating each frame."""
        self.ai.reset()
        baked = bake_for_batch(self.ai, npt=NPT, unit=UNIT, split="bbox")
        rng = np.random.default_rng(7)
        batch = np.stack([
            self.image,
            self.image * 0.5,
            self.image + rng.uniform(0, 1, self.image.shape).astype(np.float32),
        ], axis=0)
        got_batch = integrate(baked, batch)         # (nbins, 3)
        for i in range(batch.shape[0]):
            got_one = integrate(baked, batch[i])     # (nbins,)
            mask = ~np.isnan(got_one)
            np.testing.assert_allclose(got_batch[mask, i], got_one[mask],
                                       rtol=1e-6, atol=1e-6)

    def test_nan_pixels_match_pyfai(self):
        """An image with NaN pixels should produce the same output from the
        baked integrator as from pyFAI's native integrate1d_ng."""
        self.ai.reset()

        rng = np.random.default_rng(5)
        poisoned = self.image.copy()
        poisoned.flat[rng.choice(poisoned.size, size=50, replace=False)] = np.nan

        method = ("bbox", "csr", "cython")
        ref = self.ai.integrate1d_ng(poisoned, npt=NPT, unit=UNIT, method=method,
                                     correctSolidAngle=True, dummy=np.nan)

        self.ai.reset()
        baked = bake_for_batch(self.ai, npt=NPT, unit=UNIT, split="bbox")
        got = integrate(baked, poisoned)

        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref.intensity),
                                      err_msg="NaN pattern differs from pyFAI")
        finite = ~np.isnan(got)
        np.testing.assert_allclose(got[finite], ref.intensity[finite],
                                   rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(JULIA_AVAILABLE,
                         "set PYFAI_JULIA_PROJECT and have `julia` on PATH")
    def test_julia_batch_matches_python(self):
        self.ai.reset()
        baked = bake_for_batch(self.ai, npt=NPT, unit=UNIT, split="bbox")
        rng = np.random.default_rng(11)
        stack = np.stack([
            self.image,
            self.image * 0.5,
            self.image + rng.uniform(0, 1, self.image.shape).astype(np.float32),
        ], axis=0)
        ref = integrate(baked, stack)                         # (nbins, B)
        got = _julia_integrate(baked, stack, frame_shape=self.image.shape)
        nan_ref = np.isnan(ref)
        nan_got = np.isnan(got)
        np.testing.assert_array_equal(nan_got, nan_ref,
                                      err_msg="Julia NaN pattern differs (batch)")
        np.testing.assert_allclose(got[~nan_got], ref[~nan_ref],
                                   rtol=1e-5, atol=1e-5,
                                   err_msg="Julia ≠ Python integrate (batch)")

    def test_hdf5_roundtrip(self):
        import h5py
        self.ai.reset()
        baked = bake_for_batch(self.ai, npt=NPT, unit=UNIT, split="bbox")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "setup.h5")
            write_hdf5(baked, path)
            with h5py.File(path, "r") as f:
                for key in ("data_raw", "data_corr", "indices", "indptr",
                            "bin_centers0"):
                    np.testing.assert_array_equal(f[key][...], baked[key])
                self.assertEqual(tuple(f.attrs["shape"]), baked["shape"])
                self.assertEqual(int(f.attrs["ndim"]), baked["ndim"])
                self.assertEqual(f.attrs["unit0"], baked["unit0"])
                self.assertEqual(f.attrs["split"], baked["split"])
                self.assertEqual(int(f.attrs["npt0"]), baked["npt0"])


NPT_RAD_2D = 500
NPT_AZIM_2D = 180


class TestBakeForBatch2D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image, cls.ai = create_fake_data(poissonian=False)
        cls.image = cls.image.astype(np.float32)
        cls.ai._empty = np.float32(np.nan)

    def _check_split_2d(self, split, *, with_pol=False, azimuth_range=None):
        ai = self.ai
        ai.reset()

        pol_factor = 0.97 if with_pol else None
        method = (split, "csr", "cython")
        ref_kwargs = dict(npt_rad=NPT_RAD_2D, npt_azim=NPT_AZIM_2D, unit=UNIT,
                          method=method, correctSolidAngle=True, dummy=np.nan)
        if pol_factor is not None:
            ref_kwargs["polarization_factor"] = pol_factor
        if azimuth_range is not None:
            ref_kwargs["azimuth_range"] = azimuth_range

        ref = ai.integrate2d_ng(self.image, **ref_kwargs)
        # pyFAI integrate2d returns intensity shape (npt_azim, npt_rad);
        # bake_for_batch returns (npt_rad, npt_azim). Transpose for compare.
        ref_I = ref.intensity.T
        ref_q = ref.radial
        ref_chi = ref.azimuthal

        ai.reset()
        baked = bake_for_batch(ai,
                               npt=(NPT_RAD_2D, NPT_AZIM_2D),
                               unit=UNIT,
                               split=split,
                               solidangle=True,
                               polarization_factor=pol_factor,
                               azimuth_range=azimuth_range)

        self.assertEqual(baked["ndim"], 2)
        np.testing.assert_allclose(baked["bin_centers0"], ref_q, rtol=1e-6,
                                   err_msg=f"radial centers differ for split={split}")
        np.testing.assert_allclose(baked["bin_centers1"], ref_chi, rtol=1e-5,
                                   err_msg=f"azimuthal centers differ for split={split}")

        got_I = integrate(baked, self.image)
        self.assertEqual(got_I.shape, (NPT_RAD_2D, NPT_AZIM_2D))

        nan_mask = np.isnan(got_I)
        np.testing.assert_array_equal(
            nan_mask, np.isnan(ref_I),
            err_msg=f"empty-bin pattern differs for split={split}")
        np.testing.assert_allclose(got_I[~nan_mask], ref_I[~nan_mask],
                                   rtol=1e-4, atol=1e-4,
                                   err_msg=f"intensity mismatch for split={split}")

        if JULIA_AVAILABLE:
            jl_I = _julia_integrate(baked, self.image,
                                    frame_shape=self.image.shape)
            self.assertEqual(jl_I.shape, (NPT_RAD_2D, NPT_AZIM_2D))
            np.testing.assert_array_equal(
                np.isnan(jl_I), nan_mask,
                err_msg=f"Julia 2D NaN pattern differs for split={split}")
            np.testing.assert_allclose(
                jl_I[~nan_mask], got_I[~nan_mask],
                rtol=1e-5, atol=1e-5,
                err_msg=f"Julia ≠ Python integrate (2D) for split={split}")

    def test_2d_split_no(self):
        self._check_split_2d("no")

    def test_2d_split_bbox(self):
        self._check_split_2d("bbox")

    def test_2d_split_full(self):
        self._check_split_2d("full")

    def test_2d_with_polarization_bbox(self):
        self._check_split_2d("bbox", with_pol=True)

    def test_2d_batch_matches_loop(self):
        self.ai.reset()
        baked = bake_for_batch(self.ai, npt=(NPT_RAD_2D, NPT_AZIM_2D),
                               unit=UNIT, split="bbox")
        rng = np.random.default_rng(13)
        batch = np.stack([
            self.image,
            self.image * 0.5,
            self.image + rng.uniform(0, 1, self.image.shape).astype(np.float32),
        ], axis=0)
        got_batch = integrate(baked, batch)
        self.assertEqual(got_batch.shape, (NPT_RAD_2D, NPT_AZIM_2D, 3))
        for i in range(batch.shape[0]):
            got_one = integrate(baked, batch[i])
            self.assertEqual(got_one.shape, (NPT_RAD_2D, NPT_AZIM_2D))
            mask = ~np.isnan(got_one)
            np.testing.assert_allclose(got_batch[..., i][mask], got_one[mask],
                                       rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(JULIA_AVAILABLE,
                         "set PYFAI_JULIA_PROJECT and have `julia` on PATH")
    def test_2d_julia_batch_matches_python(self):
        self.ai.reset()
        baked = bake_for_batch(self.ai, npt=(NPT_RAD_2D, NPT_AZIM_2D),
                               unit=UNIT, split="bbox")
        rng = np.random.default_rng(17)
        stack = np.stack([
            self.image,
            self.image * 0.5,
            self.image + rng.uniform(0, 1, self.image.shape).astype(np.float32),
        ], axis=0)
        ref = integrate(baked, stack)
        got = _julia_integrate(baked, stack, frame_shape=self.image.shape)
        self.assertEqual(got.shape, ref.shape)
        nan_ref = np.isnan(ref)
        np.testing.assert_array_equal(np.isnan(got), nan_ref,
                                      err_msg="Julia 2D NaN pattern differs (batch)")
        np.testing.assert_allclose(got[~nan_ref], ref[~nan_ref],
                                   rtol=1e-5, atol=1e-5,
                                   err_msg="Julia ≠ Python integrate (2D batch)")

    def test_2d_hdf5_roundtrip(self):
        import h5py
        self.ai.reset()
        baked = bake_for_batch(self.ai, npt=(NPT_RAD_2D, NPT_AZIM_2D),
                               unit=UNIT, split="bbox")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "setup.h5")
            write_hdf5(baked, path)
            with h5py.File(path, "r") as f:
                for key in ("data_raw", "data_corr", "indices", "indptr",
                            "bin_centers0", "bin_centers1"):
                    np.testing.assert_array_equal(f[key][...], baked[key])
                self.assertEqual(int(f.attrs["ndim"]), 2)
                self.assertEqual(f.attrs["unit0"], baked["unit0"])
                self.assertEqual(f.attrs["unit1"], baked["unit1"])
                self.assertEqual(int(f.attrs["npt0"]), baked["npt0"])
                self.assertEqual(int(f.attrs["npt1"]), baked["npt1"])


if __name__ == "__main__":
    unittest.main()
