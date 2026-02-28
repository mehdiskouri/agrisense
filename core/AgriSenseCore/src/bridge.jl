# ---------------------------------------------------------------------------
# bridge.jl — Python-facing API via juliacall
#
# Contract: all inputs/outputs are plain Dict, Vector, Array.
#           GPU is used internally; CuArray never crosses the boundary.
# ---------------------------------------------------------------------------

"""
    build_graph(farm_config::Dict) -> Dict

Build a layered hypergraph from a farm configuration dict.
Returns a serialized graph state (CPU arrays) for storage on the Python side.
"""
function build_graph(farm_config::Dict)::Dict{String,Any}
    profile = FarmProfile(farm_config)
    vertices = get(farm_config, "vertices", Dict{String,Any}[])
    edges = get(farm_config, "edges", Dict{String,Any}[])

    graph = build_hypergraph(profile, vertices, edges)

    # Serialize to plain dicts for Python
    return serialize_graph(graph)
end

"""
    query_farm_status(graph_state::Dict, zone_id::String) -> Dict

Query current status of a zone across all active layers.
"""
function query_farm_status(graph_state::Dict, zone_id::String)::Dict{String,Any}
    graph = deserialize_graph(graph_state)
    status = Dict{String,Any}()
    for (layer_name, _) in graph.layers
        status[string(layer_name)] = query_layer(graph, layer_name, zone_id)
    end
    return status
end

"""
    irrigation_schedule(graph_state::Dict, horizon_days::Int) -> Vector{Dict}
"""
function irrigation_schedule(graph_state::Dict, horizon_days::Int)::Vector{Dict{String,Any}}
    graph = deserialize_graph(graph_state)
    return compute_irrigation_schedule(graph, Dict{String,Any}(), horizon_days)
end

"""
    nutrient_report(graph_state::Dict) -> Vector{Dict}
"""
function nutrient_report(graph_state::Dict)::Vector{Dict{String,Any}}
    graph = deserialize_graph(graph_state)
    return compute_nutrient_report(graph)
end

"""
    yield_forecast(graph_state::Dict) -> Vector{Dict}
"""
function yield_forecast(graph_state::Dict)::Vector{Dict{String,Any}}
    graph = deserialize_graph(graph_state)
    return compute_yield_forecast(graph)
end

"""
    detect_anomalies(graph_state::Dict) -> Vector{Dict}
"""
function detect_anomalies(graph_state::Dict)::Vector{Dict{String,Any}}
    graph = deserialize_graph(graph_state)
    return compute_anomaly_detection(graph)
end

"""
    generate_synthetic(; farm_type::String, days::Int, seed::Int) -> Dict
"""
function generate_synthetic(; farm_type::String="greenhouse",
                              days::Int=90,
                              seed::Int=42)::Dict{String,Any}
    return generate(Symbol(farm_type), days, seed)
end

# ---------------------------------------------------------------------------
# Internal serialization helpers
# ---------------------------------------------------------------------------

function serialize_graph(graph::LayeredHyperGraph)::Dict{String,Any}
    layers_dict = Dict{String,Any}()
    for (name, layer) in graph.layers
        layers_dict[string(name)] = Dict{String,Any}(
            "incidence_rows" => findnz(layer.incidence)[1],
            "incidence_cols" => findnz(layer.incidence)[2],
            "incidence_vals" => findnz(layer.incidence)[3],
            "n_vertices" => size(layer.incidence, 1),
            "n_edges" => size(layer.incidence, 2),
            "vertex_features" => Array(layer.vertex_features),
            "edge_metadata" => layer.edge_metadata,
            "vertex_ids" => layer.vertex_ids,
            "edge_ids" => layer.edge_ids,
        )
    end
    return Dict{String,Any}(
        "farm_id" => graph.farm_id,
        "n_vertices" => graph.n_vertices,
        "vertex_index" => graph.vertex_index,
        "layers" => layers_dict,
    )
end

function deserialize_graph(state::Dict)::LayeredHyperGraph
    vertex_index = Dict{String,Int}(string(k) => Int(v) for (k, v) in state["vertex_index"])
    n_vertices = Int(state["n_vertices"])

    layers = Dict{Symbol, HyperGraphLayer}()
    for (name, ld) in state["layers"]
        rows = Int32.(ld["incidence_rows"])
        cols = Int32.(ld["incidence_cols"])
        vals = Float32.(ld["incidence_vals"])
        nv = Int(ld["n_vertices"])
        ne = Int(ld["n_edges"])
        B = sparse(rows, cols, vals, nv, ne)
        vf = Float32.(ld["vertex_features"])
        em = ld["edge_metadata"]
        vids = String.(ld["vertex_ids"])
        eids = String.(ld["edge_ids"])
        layers[Symbol(name)] = HyperGraphLayer(B, vf, em, vids, eids)
    end

    LayeredHyperGraph(string(state["farm_id"]), n_vertices, vertex_index, layers)
end
