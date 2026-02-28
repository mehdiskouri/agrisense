module AgriSenseCore

using SparseArrays
using LinearAlgebra
using Statistics
using Random
using Dates

# ---------------------------------------------------------------------------
# Conditional GPU support — CPU fallback when CUDA is unavailable
# ---------------------------------------------------------------------------
using KernelAbstractions
using Adapt
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
include("synthetic/lighting.jl")
include("synthetic/generator.jl")
include("bridge.jl")

# ---------------------------------------------------------------------------
# Public API (exposed to Python via bridge.jl)
# ---------------------------------------------------------------------------

# Core types
export ZoneConfig, ModelConfig, FarmProfile
export HyperGraphLayer, LayeredHyperGraph
export DEFAULT_HISTORY_SIZE, push_features!, get_history

# Hypergraph engine
export build_hypergraph, to_gpu, to_cpu
export cross_layer_query, query_layer, update_vertex_features!
export add_hyperedge!, remove_hyperedge!, add_vertex!
export LAYER_FEATURE_DIMS, feature_dim
export aggregate_by_edge, multi_layer_features

# Bridge API (Python-facing)
export build_graph, query_farm_status, irrigation_schedule
export nutrient_report, yield_forecast, detect_anomalies
export generate_synthetic
export serialize_graph, deserialize_graph
export update_features, train_yield_residual

# Model internals (for testing / advanced use)
export compute_irrigation_schedule, compute_nutrient_report
export compute_yield_forecast, compute_anomaly_detection
export compute_stress_coefficients, fit_residual_model, train_yield_residual!
export RESIDUAL_COEFFICIENTS
export hargreaves_et0, growth_progress_to_kc
export DEFAULT_WILTING_POINT, DEFAULT_FIELD_CAPACITY, DEFAULT_VALVE_CAPACITY
export MIN_HISTORY_FOR_ANOMALY

# GPU utilities
export get_backend, get_array_type, HAS_CUDA
export array_backend, launch_kernel!, ensure_cpu

# Graph cache
export GRAPH_CACHE, cache_graph!, get_cached_graph, evict_graph!, clear_cache!

# GPU kernels (for testing / advanced use)
export fao_yield_kernel!, nutrient_stress_kernel!, weather_stress_kernel!
export rolling_stats_kernel!, western_electric_kernel!
export water_balance_kernel!, threshold_trigger_kernel!, hargreaves_et0_kernel!
export push_features_kernel!

# Anomaly helpers
export anomaly_type_from_layer
export severity_to_urgency, suggest_amendment
export SOIL_DEPTH_MM
export CADENCE_MINUTES, compute_derived_features

end # module AgriSenseCore
