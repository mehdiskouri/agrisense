# ---------------------------------------------------------------------------
# Hypergraph construction, query, and update operations
# ---------------------------------------------------------------------------

"""
    build_hypergraph(profile::FarmProfile, vertices::Dict, edges::Dict) -> LayeredHyperGraph

Construct the layered hypergraph from farm profile, vertex definitions, and edge definitions.
Each layer gets a sparse incidence matrix built from the vertex→edge memberships.
"""
function build_hypergraph(profile::FarmProfile,
                          vertices::Vector{Dict{String,Any}},
                          edges::Vector{Dict{String,Any}})::LayeredHyperGraph

    # Build global vertex index (all vertices across all layers share the same row space)
    vertex_ids = [string(v["id"]) for v in vertices]
    vertex_index = Dict(id => i for (i, id) in enumerate(vertex_ids))
    n_vertices = length(vertex_ids)

    layers = Dict{Symbol, HyperGraphLayer}()

    for layer_sym in profile.active_layers
        layer_edges = filter(e -> Symbol(e["layer"]) == layer_sym, edges)
        n_edges = length(layer_edges)

        if n_edges == 0
            continue
        end

        # Build sparse incidence matrix
        row_inds = Int32[]
        col_inds = Int32[]
        edge_ids = String[]
        edge_meta = Dict{String,Any}[]

        for (j, e) in enumerate(layer_edges)
            push!(edge_ids, string(e["id"]))
            push!(edge_meta, get(e, "metadata", Dict{String,Any}()))
            for vid in e["vertex_ids"]
                vid_str = string(vid)
                if haskey(vertex_index, vid_str)
                    push!(row_inds, Int32(vertex_index[vid_str]))
                    push!(col_inds, Int32(j))
                end
            end
        end

        B = sparse(row_inds, col_inds, ones(Float32, length(row_inds)),
                   n_vertices, n_edges)

        # Initialize vertex features as zeros (will be populated by ingestion)
        vertex_features = zeros(Float32, n_vertices, 1)

        layers[layer_sym] = HyperGraphLayer(
            B, vertex_features, edge_meta, vertex_ids, edge_ids,
        )
    end

    LayeredHyperGraph(profile.farm_id, n_vertices, vertex_index, layers)
end

"""
    to_gpu(graph::LayeredHyperGraph) -> LayeredHyperGraph

Move all layer data to GPU. Returns a new graph with CuSparseMatrixCSR incidence
and CuMatrix features. No-op if CUDA is unavailable.
"""
function to_gpu(graph::LayeredHyperGraph)
    if !HAS_CUDA
        return graph
    end
    gpu_layers = Dict{Symbol, HyperGraphLayer}()
    for (name, layer) in graph.layers
        gpu_layers[name] = Adapt.adapt(get_array_type(), layer)
    end
    LayeredHyperGraph(graph.farm_id, graph.n_vertices, graph.vertex_index, gpu_layers)
end

"""
    cross_layer_query(graph, layer_a::Symbol, layer_b::Symbol) -> Matrix{Float32}

Compute the cross-layer connectivity: B_a' * B_b.
Result[i,j] = number of shared vertices between edge i of layer_a and edge j of layer_b.
This is a single SpMM call on GPU.
"""
function cross_layer_query(graph::LayeredHyperGraph,
                           layer_a::Symbol,
                           layer_b::Symbol)::Matrix{Float32}
    Ba = graph.layers[layer_a].incidence
    Bb = graph.layers[layer_b].incidence
    result = Ba' * Bb
    return Matrix(result)  # densify for bridge serialization
end

"""
    query_layer(graph, layer::Symbol, vertex_id::String) -> Dict

Get the hyperedges and features for a specific vertex in a layer.
"""
function query_layer(graph::LayeredHyperGraph,
                     layer::Symbol,
                     vertex_id::String)::Dict{String,Any}
    if !haskey(graph.layers, layer)
        return Dict{String,Any}("error" => "Layer $layer not found")
    end
    lyr = graph.layers[layer]
    if !haskey(graph.vertex_index, vertex_id)
        return Dict{String,Any}("error" => "Vertex $vertex_id not found")
    end
    row = graph.vertex_index[vertex_id]
    # Sparse row → find which edges this vertex belongs to
    edge_indices = findnz(lyr.incidence[row, :])[1]
    edge_ids = [lyr.edge_ids[j] for j in edge_indices]
    features = Vector(lyr.vertex_features[row, :])

    Dict{String,Any}(
        "vertex_id" => vertex_id,
        "layer" => string(layer),
        "edge_ids" => edge_ids,
        "features" => features,
    )
end

"""
    update_vertex_features!(graph, layer::Symbol, vertex_id::String, features::Vector{Float32})

In-place update of a vertex's feature vector in a specific layer.
"""
function update_vertex_features!(graph::LayeredHyperGraph,
                                  layer::Symbol,
                                  vertex_id::String,
                                  features::Vector{Float32})
    lyr = graph.layers[layer]
    row = graph.vertex_index[vertex_id]
    n_feat = length(features)
    # Expand feature matrix if new features are wider
    if n_feat > size(lyr.vertex_features, 2)
        old = lyr.vertex_features
        new_feat = zeros(Float32, size(old, 1), n_feat)
        new_feat[:, 1:size(old, 2)] .= old
        lyr.vertex_features = new_feat
    end
    lyr.vertex_features[row, 1:n_feat] .= features
    return nothing
end
