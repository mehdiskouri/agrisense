# ---------------------------------------------------------------------------
# Core types for the layered hypergraph engine
# ---------------------------------------------------------------------------

"""
Configuration for a single zone within a farm.
"""
struct ZoneConfig
    id::String
    name::String
    zone_type::Symbol          # :open_field, :greenhouse
    area_m2::Float64
    soil_type::String
end

# NOTE: No Adapt.@adapt_structure — ZoneConfig is a CPU-only config object
# consumed during build_hypergraph and never stored on the graph.

"""
Configuration for which predictive models to run on a farm.
"""
struct ModelConfig
    irrigation::Bool
    nutrients::Bool
    yield_forecast::Bool
    anomaly_detection::Bool
end

# NOTE: No Adapt.@adapt_structure — ModelConfig is CPU-only.

"""
Farm-level profile that determines active topology and behaviour.
"""
struct FarmProfile
    farm_id::String
    farm_type::Symbol           # :open_field, :greenhouse, :hybrid
    active_layers::Set{Symbol}  # {:soil, :irrigation, :weather, ...}
    zones::Vector{ZoneConfig}
    models::ModelConfig
end

# NOTE: No Adapt.@adapt_structure — FarmProfile contains Set{Symbol} and
# String fields that cannot be GPU-transferred.  FarmProfile is a transient
# config object consumed during build_hypergraph then discarded.

# ---------------------------------------------------------------------------
# Hypergraph layer — sparse incidence matrix + feature matrices + ring buffer
# ---------------------------------------------------------------------------

"""Default ring buffer depth: 96 slots = 24h at 15-min cadence."""
const DEFAULT_HISTORY_SIZE = 96
const CADENCE_MINUTES = 15  # sensor sampling interval in minutes (96 slots = 24h)

"""
A single layer of the hypergraph.

- `incidence`:        |V| × |E_l| sparse incidence matrix (1.0 where vertex ∈ hyperedge)
- `vertex_features`:  |V| × d_l  dense feature matrix  (current sensor readings)
- `feature_history`:  |V| × d_l × buffer_size  ring buffer of past readings
- `history_head`:     circular write cursor (1-indexed, next slot to write)
- `history_length`:   how many valid entries are in the buffer (0 → buffer_size)
- `edge_metadata`:    per-hyperedge metadata dicts
- `vertex_ids`:       ordered list of vertex UUID strings (row index → id)
- `edge_ids`:         ordered list of hyperedge UUID strings (col index → id)
"""
mutable struct HyperGraphLayer{M<:AbstractSparseMatrix{Float32},
                                F<:AbstractMatrix{Float32},
                                H<:AbstractArray{Float32, 3}}
    incidence::M
    vertex_features::F
    feature_history::H                    # (n_vertices, d, buffer_size)
    history_head::Int                     # next write position (1-indexed)
    history_length::Int                   # valid entries count (0 → buffer_size)
    edge_metadata::Vector{Dict{String,Any}}
    vertex_ids::Vector{String}
    edge_ids::Vector{String}
    # --- Validity masks (Phase 13) ---
    # true = valid reading written, false = no data / NaN.
    # Use Array{Bool} (not BitArray) so CuArray{Bool} GPU transfer works.
    feature_history_mask::AbstractArray{Bool, 3}   # (n_vertices, d, buffer_size)
    vertex_features_mask::AbstractArray{Bool, 2}   # (n_vertices, d)
end

# Convenience constructor — backward compat: masks default to all-false (empty buffer)
# Creates masks on the same backend (GPU/CPU) as vertex_features.
function HyperGraphLayer(incidence::M, vertex_features::F, feature_history::H,
                         history_head::Int, history_length::Int,
                         edge_metadata::Vector{Dict{String,Any}},
                         vertex_ids::Vector{String},
                         edge_ids::Vector{String}) where {M<:AbstractSparseMatrix{Float32},
                                                           F<:AbstractMatrix{Float32},
                                                           H<:AbstractArray{Float32, 3}}
    nv, d = size(vertex_features)
    buf_size = size(feature_history, 3)
    backend = array_backend(vertex_features)
    if !(backend isa CPU) && HAS_CUDA
        fh_mask = CUDA.zeros(Bool, nv, d, buf_size)
        vf_mask = CUDA.zeros(Bool, nv, d)
    else
        fh_mask = Array{Bool,3}(undef, nv, d, buf_size)
        fill!(fh_mask, false)
        vf_mask = Array{Bool,2}(undef, nv, d)
        fill!(vf_mask, false)
    end
    return HyperGraphLayer{M,F,H}(incidence, vertex_features, feature_history,
                                   history_head, history_length,
                                   edge_metadata, vertex_ids, edge_ids,
                                   fh_mask, vf_mask)
end

# Custom Adapt rule — only move numeric arrays to GPU; strings / dicts stay on CPU.
# The default @adapt_structure would try to send String vectors to CuArray and crash.
function Adapt.adapt_structure(to, layer::HyperGraphLayer)
    adapted_inc = Adapt.adapt(to, layer.incidence)
    adapted_vf  = Adapt.adapt(to, layer.vertex_features)
    adapted_fh  = Adapt.adapt(to, layer.feature_history)
    adapted_fhm = Adapt.adapt(to, layer.feature_history_mask)
    adapted_vfm = Adapt.adapt(to, layer.vertex_features_mask)
    # Use the 8-arg constructor (which allocates dummy masks), then overwrite masks
    lyr = HyperGraphLayer(adapted_inc, adapted_vf, adapted_fh,
                           layer.history_head, layer.history_length,
                           layer.edge_metadata, layer.vertex_ids, layer.edge_ids)
    lyr.feature_history_mask = adapted_fhm
    lyr.vertex_features_mask = adapted_vfm
    return lyr
end

# ---------------------------------------------------------------------------
# Backend detection from array type — determines GPU vs CPU from the data itself
# ---------------------------------------------------------------------------

"""
    array_backend(arr) -> KernelAbstractions.Backend

Return the compute backend appropriate for `arr`.
`CuArray` → `CUDABackend()`, everything else → `CPU()`.
"""
function array_backend(arr::AbstractArray)
    HAS_CUDA && arr isa CUDA.AnyCuArray ? CUDABackend() : CPU()
end

"""
    launch_kernel!(kern, backend, ndrange, args...)

Thin wrapper: launch a KernelAbstractions kernel and synchronize.
"""
function launch_kernel!(kern, backend, ndrange, args...)
    kern(backend, 256)(args...; ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    ensure_cpu(arr) -> Array

Bring an array back to CPU. No-op for plain `Array`.
"""
ensure_cpu(arr::Array) = arr
ensure_cpu(arr::AbstractArray) = Array(arr)
ensure_cpu(arr::AbstractSparseMatrix) = SparseMatrixCSC{Float32,Int32}(sparse(Array(arr)))

# ---------------------------------------------------------------------------
# GPU-safe push_features! kernel
# ---------------------------------------------------------------------------

@kernel function push_features_kernel!(vf, fh, fh_mask, vf_mask, @Const(features_gpu),
                                        vertex_idx::Int32, head::Int32, n_feat::Int32)
    f = @index(Global)
    @inbounds begin
        if f <= n_feat
            val = features_gpu[f]
            vf[vertex_idx, f] = val
            fh[vertex_idx, f, head] = val
            valid = !isnan(val)
            fh_mask[vertex_idx, f, head] = valid
            vf_mask[vertex_idx, f] = valid
        end
    end
end

# ---------------------------------------------------------------------------
# Ring buffer helpers
# ---------------------------------------------------------------------------

"""
    push_features!(layer, vertex_idx, features)

Write `features` into the current snapshot AND advance the ring buffer.
GPU-safe: uses a kernel when arrays are on GPU.
"""
function push_features!(layer::HyperGraphLayer, vertex_idx::Int,
                        features::Vector{Float32})
    d = size(layer.vertex_features, 2)
    n_feat = min(length(features), d)
    buf_size = size(layer.feature_history, 3)

    backend = array_backend(layer.vertex_features)
    if backend isa CPU
        # CPU path — direct scalar indexing
        feat_view = @view features[1:n_feat]
        layer.vertex_features[vertex_idx, 1:n_feat] .= feat_view
        layer.feature_history[vertex_idx, 1:n_feat, layer.history_head] .= feat_view
        # Update validity masks — true where value is not NaN
        valid_mask = .!isnan.(feat_view)
        layer.feature_history_mask[vertex_idx, 1:n_feat, layer.history_head] .= valid_mask
        layer.vertex_features_mask[vertex_idx, 1:n_feat] .= valid_mask
    else
        # GPU path — upload features once, launch kernel
        features_gpu = CuArray(features[1:n_feat])
        launch_kernel!(push_features_kernel!, backend, n_feat,
                       layer.vertex_features, layer.feature_history,
                       layer.feature_history_mask, layer.vertex_features_mask,
                       features_gpu,
                       Int32(vertex_idx), Int32(layer.history_head), Int32(n_feat))
    end

    # Head/length are CPU scalars on the mutable struct
    layer.history_head = mod1(layer.history_head + 1, buf_size)
    layer.history_length = min(layer.history_length + 1, buf_size)
    return nothing
end

"""
    get_history(layer, vertex_idx; return_mask=false)

Return `(d, valid_length)` matrix of historical readings for a vertex,
ordered oldest-first.  Returns a `(d, 0)` matrix if buffer is empty.
Always returns CPU arrays.

When `return_mask=true`, returns `(data::Matrix{Float32}, mask::Matrix{Bool})`
where `mask[f, t]` is `true` iff the slot held a valid (non-NaN) reading.
"""
function get_history(layer::HyperGraphLayer, vertex_idx::Int; return_mask::Bool=false)
    d = size(layer.vertex_features, 2)
    buf_size = size(layer.feature_history, 3)
    len = layer.history_length

    if len == 0
        data = zeros(Float32, d, 0)
        return return_mask ? (data, Matrix{Bool}(undef, d, 0)) : data
    end

    # Pull the vertex's full history slice to CPU if needed
    hist_slice = ensure_cpu(layer.feature_history[vertex_idx:vertex_idx, :, :])  # (1, d, buf_size)
    mask_slice = if return_mask
        ensure_cpu(layer.feature_history_mask[vertex_idx:vertex_idx, :, :])
    else
        nothing
    end

    result = Matrix{Float32}(undef, d, len)
    mask_result = return_mask ? Matrix{Bool}(undef, d, len) : nothing

    if len < buf_size
        for t in 1:len
            result[:, t] .= @view hist_slice[1, :, t]
            if return_mask
                mask_result[:, t] .= @view mask_slice[1, :, t]
            end
        end
    else
        for t in 1:len
            src_idx = mod1(layer.history_head + t - 1, buf_size)
            result[:, t] .= @view hist_slice[1, :, src_idx]
            if return_mask
                mask_result[:, t] .= @view mask_slice[1, :, src_idx]
            end
        end
    end
    return return_mask ? (result, mask_result) : result
end

"""
The full layered hypergraph for one farm.
"""
mutable struct LayeredHyperGraph
    farm_id::String
    n_vertices::Int
    vertex_index::Dict{String,Int}              # vertex_id → row index
    layers::Dict{Symbol, HyperGraphLayer}
end

# Custom Adapt rule — keep cpu-only fields (farm_id, vertex_index) on CPU,
# delegate each HyperGraphLayer to its own custom Adapt.adapt_structure.
# This makes `Adapt.adapt(backend, graph)` a safe alternative to `to_gpu`.
function Adapt.adapt_structure(to, graph::LayeredHyperGraph)
    adapted_layers = Dict{Symbol, HyperGraphLayer}(
        name => Adapt.adapt_structure(to, layer)
        for (name, layer) in graph.layers
    )
    return LayeredHyperGraph(
        graph.farm_id,       # String — stays on CPU
        graph.n_vertices,    # Int — scalar
        graph.vertex_index,  # Dict{String,Int} — stays on CPU
        adapted_layers,
    )
end

# ---------------------------------------------------------------------------
# Constructors from plain Dict (coming from Python bridge)
# ---------------------------------------------------------------------------

function ZoneConfig(d::Dict)
    ZoneConfig(
        string(d["id"]),
        string(d["name"]),
        Symbol(d["zone_type"]),
        Float64(d["area_m2"]),
        string(get(d, "soil_type", "unknown")),
    )
end

function ModelConfig(d::Dict)
    ModelConfig(
        get(d, "irrigation", true),
        get(d, "nutrients", true),
        get(d, "yield_forecast", true),
        get(d, "anomaly_detection", true),
    )
end

function FarmProfile(d::Dict)
    zones = [ZoneConfig(z) for z in get(d, "zones", Dict[])]
    models = ModelConfig(get(d, "models", Dict()))
    FarmProfile(
        string(d["farm_id"]),
        Symbol(d["farm_type"]),
        Set(Symbol.(get(d, "active_layers", String[]))),
        zones,
        models,
    )
end
