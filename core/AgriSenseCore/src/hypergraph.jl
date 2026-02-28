# ---------------------------------------------------------------------------
# Hypergraph construction, query, and update operations
# ---------------------------------------------------------------------------

"""
Default feature dimensionality per layer.
Maps layer symbol → number of columns in the vertex_features matrix.
Layers not listed here default to 1.
"""
const LAYER_FEATURE_DIMS = Dict{Symbol,Int}(
    :soil        => 4,   # moisture, temperature, conductivity, pH
    :irrigation  => 3,   # flow_rate, pressure, valve_state
    :weather     => 5,   # temp, humidity, precip, wind_speed, solar_rad
    :npk         => 3,   # nitrogen, phosphorus, potassium
    :lighting    => 3,   # par, dli, spectrum_index
    :vision      => 4,   # canopy_coverage, growth_stage, anomaly_score, ndvi
    :crop_requirements => 2,  # target_yield, growth_progress
)

"""
    feature_dim(layer::Symbol) -> Int

Return the feature dimensionality for a given layer (defaults to 1).
"""
feature_dim(layer::Symbol)::Int = get(LAYER_FEATURE_DIMS, layer, 1)

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

        # Initialize vertex features with correct dimensionality per layer
        vertex_features = zeros(Float32, n_vertices, feature_dim(layer_sym))

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
    haskey(graph.layers, layer_a) || error("cross_layer_query: layer :$layer_a not found")
    haskey(graph.layers, layer_b) || error("cross_layer_query: layer :$layer_b not found")
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
        avail = join(collect(keys(graph.layers)), ", ")
        return Dict{String,Any}("error" => "Layer :$layer not found. Available: [$avail]")
    end
    lyr = graph.layers[layer]
    if !haskey(graph.vertex_index, vertex_id)
        return Dict{String,Any}("error" => "Vertex '$vertex_id' not found in graph ($(graph.n_vertices) vertices)")
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

# ---------------------------------------------------------------------------
# Topology mutation helpers
# ---------------------------------------------------------------------------

"""
    add_hyperedge!(graph, layer::Symbol, edge_id::String, vertex_ids::Vector{String};
                   metadata=Dict{String,Any}())

Add a new hyperedge to an existing layer.  Updates the sparse incidence matrix and edge
metadata in-place.  Vertices not present in `graph.vertex_index` are silently skipped.
If the layer doesn't exist, creates it.
"""
function add_hyperedge!(graph::LayeredHyperGraph,
                        layer::Symbol,
                        edge_id::String,
                        vertex_ids::Vector{String};
                        metadata::Dict{String,Any}=Dict{String,Any}())
    nv = graph.n_vertices
    if !haskey(graph.layers, layer)
        # Create a brand-new layer with this single edge
        row_inds = Int32[]
        for vid in vertex_ids
            if haskey(graph.vertex_index, vid)
                push!(row_inds, Int32(graph.vertex_index[vid]))
            end
        end
        B = sparse(row_inds, ones(Int32, length(row_inds)), ones(Float32, length(row_inds)),
                   nv, 1)
        vf = zeros(Float32, nv, feature_dim(layer))
        all_vids = collect(keys(graph.vertex_index))
        sort!(all_vids; by=v -> graph.vertex_index[v])
        graph.layers[layer] = HyperGraphLayer(B, vf, [metadata], all_vids, [edge_id])
        return nothing
    end

    lyr = graph.layers[layer]
    old_ne = size(lyr.incidence, 2)
    new_col = zeros(Float32, nv)
    for vid in vertex_ids
        if haskey(graph.vertex_index, vid)
            new_col[graph.vertex_index[vid]] = 1.0f0
        end
    end
    # Extend incidence: hcat existing sparse with new column
    lyr.incidence = hcat(lyr.incidence, sparse(new_col))
    push!(lyr.edge_metadata, metadata)
    push!(lyr.edge_ids, edge_id)
    return nothing
end

"""
    remove_hyperedge!(graph, layer::Symbol, edge_id::String) -> Bool

Remove a hyperedge from a layer by its edge_id.  Returns `true` if found and removed.
"""
function remove_hyperedge!(graph::LayeredHyperGraph,
                           layer::Symbol,
                           edge_id::String)::Bool
    if !haskey(graph.layers, layer)
        return false
    end
    lyr = graph.layers[layer]
    idx = findfirst(==(edge_id), lyr.edge_ids)
    if idx === nothing
        return false
    end
    # Remove the column from the incidence matrix
    keep = setdiff(1:size(lyr.incidence, 2), idx)
    lyr.incidence = lyr.incidence[:, keep]
    deleteat!(lyr.edge_ids, idx)
    deleteat!(lyr.edge_metadata, idx)
    return true
end

"""
    add_vertex!(graph, vertex_id::String) -> Int

Add a new vertex to the graph.  Expands every layer's incidence matrix and
vertex_features by one row.  Returns the new row index.
Raises an error if the vertex already exists.
"""
function add_vertex!(graph::LayeredHyperGraph, vertex_id::String)::Int
    if haskey(graph.vertex_index, vertex_id)
        error("Vertex $vertex_id already exists at index $(graph.vertex_index[vertex_id])")
    end
    graph.n_vertices += 1
    new_idx = graph.n_vertices
    graph.vertex_index[vertex_id] = new_idx

    for (_, lyr) in graph.layers
        nv_old, ne = size(lyr.incidence)
        # Add a zero row to incidence
        lyr.incidence = vcat(lyr.incidence, spzeros(Float32, 1, ne))
        # Add a zero row to vertex_features
        lyr.vertex_features = vcat(lyr.vertex_features, zeros(Float32, 1, size(lyr.vertex_features, 2)))
        push!(lyr.vertex_ids, vertex_id)
    end
    return new_idx
end

# ---------------------------------------------------------------------------
# CPU round-trip helper
# ---------------------------------------------------------------------------

"""
    to_cpu(graph::LayeredHyperGraph) -> LayeredHyperGraph

Move all layer data back to CPU.  For CPU-resident graphs this is a no-op
(sparse matrices and plain arrays stay as-is).
"""
function to_cpu(graph::LayeredHyperGraph)::LayeredHyperGraph
    cpu_layers = Dict{Symbol, HyperGraphLayer}()
    for (name, layer) in graph.layers
        # Collect to CPU arrays first, then convert to standard types
        B_cpu = SparseMatrixCSC{Float32,Int32}(sparse(Array(layer.incidence)))
        vf = Matrix{Float32}(Array(layer.vertex_features))
        cpu_layers[name] = HyperGraphLayer(B_cpu, vf, layer.edge_metadata,
                                            layer.vertex_ids, layer.edge_ids)
    end
    LayeredHyperGraph(graph.farm_id, graph.n_vertices,
                      copy(graph.vertex_index), cpu_layers)
end
