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

"""
Configuration for which predictive models to run on a farm.
"""
struct ModelConfig
    irrigation::Bool
    nutrients::Bool
    yield_forecast::Bool
    anomaly_detection::Bool
end

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

# ---------------------------------------------------------------------------
# Hypergraph layer — sparse incidence matrix + feature matrices + ring buffer
# ---------------------------------------------------------------------------

"""Default ring buffer depth: 96 slots = 24h at 15-min cadence."""
const DEFAULT_HISTORY_SIZE = 96

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
end

# Custom Adapt rule — only move numeric arrays to GPU; strings / dicts stay on CPU.
# The default @adapt_structure would try to send String vectors to CuArray and crash.
function Adapt.adapt_structure(to, layer::HyperGraphLayer)
    HyperGraphLayer(
        Adapt.adapt(to, layer.incidence),
        Adapt.adapt(to, layer.vertex_features),
        Adapt.adapt(to, layer.feature_history),  # numeric 3D → safe for GPU
        layer.history_head,
        layer.history_length,
        layer.edge_metadata,   # keep on CPU
        layer.vertex_ids,      # keep on CPU
        layer.edge_ids,        # keep on CPU
    )
end

# ---------------------------------------------------------------------------
# Ring buffer helpers
# ---------------------------------------------------------------------------

"""
    push_features!(layer, vertex_idx, features)

Write `features` into the current snapshot AND advance the ring buffer.
"""
function push_features!(layer::HyperGraphLayer, vertex_idx::Int,
                        features::Vector{Float32})
    d = size(layer.vertex_features, 2)
    n_feat = min(length(features), d)

    # Update current snapshot
    layer.vertex_features[vertex_idx, 1:n_feat] .= @view features[1:n_feat]

    # Write into ring buffer at current head position
    buf_size = size(layer.feature_history, 3)
    layer.feature_history[vertex_idx, 1:n_feat, layer.history_head] .= @view features[1:n_feat]

    # Advance head (wraps around)
    layer.history_head = mod1(layer.history_head + 1, buf_size)
    layer.history_length = min(layer.history_length + 1, buf_size)
    return nothing
end

"""
    get_history(layer, vertex_idx) -> Matrix{Float32}

Return `(d, valid_length)` matrix of historical readings for a vertex,
ordered oldest-first.  Returns a `(d, 0)` matrix if buffer is empty.
"""
function get_history(layer::HyperGraphLayer, vertex_idx::Int)::Matrix{Float32}
    d = size(layer.vertex_features, 2)
    buf_size = size(layer.feature_history, 3)
    len = layer.history_length

    if len == 0
        return zeros(Float32, d, 0)
    end

    result = Matrix{Float32}(undef, d, len)
    if len < buf_size
        # Buffer not yet full — data is in positions 1:len
        for t in 1:len
            result[:, t] .= @view layer.feature_history[vertex_idx, :, t]
        end
    else
        # Buffer full — oldest is at head (it's the next to be overwritten)
        for t in 1:len
            src_idx = mod1(layer.history_head + t - 1, buf_size)
            result[:, t] .= @view layer.feature_history[vertex_idx, :, src_idx]
        end
    end
    return result
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
