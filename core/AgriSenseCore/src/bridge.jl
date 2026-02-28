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
    irrigation_schedule(graph_state::Dict, horizon_days::Int, weather_forecast::Dict=Dict()) -> Vector{Dict}
"""
function irrigation_schedule(graph_state::Dict, horizon_days::Int,
                              weather_forecast::Dict=Dict{String,Any}())::Vector{Dict{String,Any}}
    graph = deserialize_graph(graph_state)
    return compute_irrigation_schedule(graph, weather_forecast, horizon_days)
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
    update_features(graph_state, layer, vertex_id, features) -> Dict

Deserialize the graph, push features into the specified layer (snapshot + ring buffer),
and return the re-serialized graph state.
"""
function update_features(graph_state::Dict, layer::String,
                          vertex_id::String,
                          features::Vector)::Dict{String,Any}
    graph = deserialize_graph(graph_state)
    lsym = Symbol(layer)
    haskey(graph.layers, lsym) || error("update_features: layer '$layer' not found")
    haskey(graph.vertex_index, vertex_id) ||
        error("update_features: vertex '$vertex_id' not found")

    vidx = graph.vertex_index[vertex_id]
    push_features!(graph.layers[lsym], vidx, Float32.(features))
    return serialize_graph(graph)
end

"""
    train_yield_residual(graph_state, outcomes) -> Dict

Fit the ridge residual model from actual yield outcomes.
`outcomes` should map vertex_id → observed yield (Float64 or Float32).
Returns a status dict.
"""
function train_yield_residual(graph_state::Dict,
                               outcomes::Dict)::Dict{String,Any}
    graph = deserialize_graph(graph_state)
    actual = Dict{String,Float32}(
        string(k) => Float32(v) for (k, v) in outcomes
    )
    train_yield_residual!(graph, actual)
    has_coeff = RESIDUAL_COEFFICIENTS[] !== nothing
    return Dict{String,Any}(
        "status" => has_coeff ? "trained" : "insufficient_data",
        "n_observations" => length(actual),
        "n_coefficients" => has_coeff ? length(RESIDUAL_COEFFICIENTS[]) : 0,
    )
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
        I, J, V = findnz(layer.incidence)  # single call instead of triple
        layers_dict[string(name)] = Dict{String,Any}(
            "incidence_rows" => I,
            "incidence_cols" => J,
            "incidence_vals" => V,
            "n_vertices" => size(layer.incidence, 1),
            "n_edges" => size(layer.incidence, 2),
            "vertex_features" => Array(layer.vertex_features),
            "feature_history" => Array(layer.feature_history),
            "history_head" => layer.history_head,
            "history_length" => layer.history_length,
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
    # --- validate top-level keys ---
    for key in ("farm_id", "n_vertices", "vertex_index", "layers")
        haskey(state, key) || error("deserialize_graph: missing required key '$key'")
    end

    vertex_index = Dict{String,Int}(string(k) => Int(v) for (k, v) in state["vertex_index"])
    n_vertices = Int(state["n_vertices"])

    layers = Dict{Symbol, HyperGraphLayer}()
    for (name, ld) in state["layers"]
        try
            rows = Int32.(ld["incidence_rows"])
            cols = Int32.(ld["incidence_cols"])
            vals = Float32.(ld["incidence_vals"])
            nv = Int(ld["n_vertices"])
            ne = Int(ld["n_edges"])
            B = sparse(rows, cols, vals, nv, ne)
            vf = Float32.(ld["vertex_features"])
            # Ring buffer — restore from serialized state or initialise empty
            fh = if haskey(ld, "feature_history")
                Float32.(ld["feature_history"])
            else
                d = size(vf, 2)
                zeros(Float32, nv, d, DEFAULT_HISTORY_SIZE)
            end
            hhead = Int(get(ld, "history_head", 1))
            hlen  = Int(get(ld, "history_length", 0))
            em = Vector{Dict{String,Any}}(ld["edge_metadata"])
            vids = String.(ld["vertex_ids"])
            eids = String.(ld["edge_ids"])
            layers[Symbol(name)] = HyperGraphLayer(B, vf, fh, hhead, hlen, em, vids, eids)
        catch e
            error("deserialize_graph: failed to reconstruct layer '$name' — $(sprint(showerror, e))")
        end
    end

    LayeredHyperGraph(string(state["farm_id"]), n_vertices, vertex_index, layers)
end
