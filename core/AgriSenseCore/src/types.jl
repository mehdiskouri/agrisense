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
# Hypergraph layer — sparse incidence matrix + feature matrices
# ---------------------------------------------------------------------------

"""
A single layer of the hypergraph.

- `incidence`:       |V| × |E_l| sparse incidence matrix (1.0 where vertex ∈ hyperedge)
- `vertex_features`: |V| × d_l  dense feature matrix  (sensor readings, attributes)
- `edge_metadata`:   per-hyperedge metadata dicts
- `vertex_ids`:      ordered list of vertex UUID strings (row index → id)
- `edge_ids`:        ordered list of hyperedge UUID strings (col index → id)
"""
mutable struct HyperGraphLayer{M<:AbstractSparseMatrix{Float32},
                                F<:AbstractMatrix{Float32}}
    incidence::M
    vertex_features::F
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
        layer.edge_metadata,   # keep on CPU
        layer.vertex_ids,      # keep on CPU
        layer.edge_ids,        # keep on CPU
    )
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
