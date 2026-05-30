module QSpaceToolsPythonCallExt

using QSpaceTools: QSpaceTools
using PythonCall: Py, pyconvert, pyisinstance, pybuiltins, pytype

# Build a BakedIntegrator straight from the Python `baked` dict returned by
# `bake_for_batch`, skipping the HDF5 layer entirely. The +1 index shift and
# the (H,W)→(W,H) flip live in `QSpaceTools._baked_from`; here we only supply
# the per-field accessor (every field is a plain dict entry, so one rule).
function QSpaceTools.load_baked(baked::Py)
    if !pyisinstance(baked, pybuiltins.dict)
        throw(ArgumentError("load_baked(::Py) expected the dict returned by bake_for_batch, got a $(pytype(baked))"))
    end

    QSpaceTools._baked_from((T, key) -> pyconvert(T, baked[key]))
end

end # module QSpaceToolsPythonCallExt
