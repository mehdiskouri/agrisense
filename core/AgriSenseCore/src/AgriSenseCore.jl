module AgriSenseCore

using SparseArrays
using LinearAlgebra
using Statistics
using Random

# ---------------------------------------------------------------------------
# Conditional GPU support — CPU fallback when CUDA is unavailable
# ---------------------------------------------------------------------------
using KernelAbstractions
using Adapt
using StaticArrays
using StructArrays

const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

if HAS_CUDA
    using CUDA
    using CUDA.CUSPARSE
end

# ---------------------------------------------------------------------------
# Backend selection — runtime dispatch to GPU or CPU
# ---------------------------------------------------------------------------

"""
    get_backend() -> KernelAbstractions.Backend

Return `CUDABackend()` if a functional CUDA GPU is available, otherwise `CPU()`.
"""
function get_backend()
    HAS_CUDA ? CUDABackend() : CPU()
end

"""
    get_array_type() -> Type

Return `CuArray` if CUDA is available, otherwise `Array`.
"""
function get_array_type()
    HAS_CUDA ? CuArray : Array
end

# ---------------------------------------------------------------------------
# Sub-modules — loaded in dependency order
# ---------------------------------------------------------------------------
include("types.jl")
include("hypergraph.jl")
include("models/irrigation.jl")
include("models/nutrients.jl")
include("models/yield.jl")
include("models/anomaly.jl")
include("synthetic/correlations.jl")
include("synthetic/soil.jl")
include("synthetic/weather.jl")
include("synthetic/npk.jl")
include("synthetic/vision.jl")
include("synthetic/generator.jl")
include("bridge.jl")

# ---------------------------------------------------------------------------
# Public API (exposed to Python via bridge.jl)
# ---------------------------------------------------------------------------
export build_graph, query_farm_status, irrigation_schedule
export nutrient_report, yield_forecast, detect_anomalies
export generate_synthetic
export get_backend, get_array_type, HAS_CUDA

end # module AgriSenseCore
