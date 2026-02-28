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
    :crop_requirements => 5,  # target_yield, growth_progress, n_target, p_target, k_target
)

"""
    feature_dim(layer::Symbol) -> Int

Return the feature dimensionality for a given layer (defaults to 1).
"""
feature_dim(layer::Symbol)::Int = get(LAYER_FEATURE_DIMS, layer, 1)

# ---------------------------------------------------------------------------
# GPU / CPU array constructors — build arrays on the right device
# ---------------------------------------------------------------------------

"""
    make_sparse_incidence(row_inds, col_inds, nv, ne) -> AbstractSparseMatrix{Float32}

Build an incidence matrix. On GPU when `HAS_CUDA`, on CPU otherwise.
`sparse()` is CPU-only so we build on CPU then move.
"""
function make_sparse_incidence(row_inds::Vector{Int32}, col_inds::Vector{Int32},
                                nv::Int, ne::Int)
    B_cpu = sparse(row_inds, col_inds, ones(Float32, length(row_inds)), nv, ne)
    if HAS_CUDA
        return CUSPARSE.CuSparseMatrixCSR(B_cpu)
    end
    return B_cpu
end

"""
    make_dense_zeros(T, dims...) -> AbstractArray{T}

Allocate a zero array on GPU if `HAS_CUDA`, CPU otherwise.
"""
function make_dense_zeros(::Type{T}, dims...) where T
    HAS_CUDA ? CUDA.zeros(T, dims...) : zeros(T, dims...)
end

# ---------------------------------------------------------------------------
# Build hypergraph — graph is GPU-resident from birth when CUDA available
# ---------------------------------------------------------------------------

"""
    build_hypergraph(profile::FarmProfile, vertices::Dict, edges::Dict) -> LayeredHyperGraph

Construct the layered hypergraph from farm profile, vertex definitions, and edge definitions.
Each layer gets a sparse incidence matrix built from the vertex→edge memberships.
Arrays live on GPU when CUDA is available, CPU otherwise.
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

        B = make_sparse_incidence(row_inds, col_inds, n_vertices, n_edges)

        # Initialize vertex features with correct dimensionality per layer
        d = feature_dim(layer_sym)
        vertex_features = make_dense_zeros(Float32, n_vertices, d)
        feature_history = make_dense_zeros(Float32, n_vertices, d, DEFAULT_HISTORY_SIZE)

        layers[layer_sym] = HyperGraphLayer(
            B, vertex_features, feature_history, 1, 0,
            edge_meta, vertex_ids, edge_ids,
        )
    end

    LayeredHyperGraph(profile.farm_id, n_vertices, vertex_index, layers)
end

# ---------------------------------------------------------------------------
# GPU ↔ CPU transfer
# ---------------------------------------------------------------------------

"""
    to_gpu(graph::LayeredHyperGraph) -> LayeredHyperGraph

Move all layer data to GPU. Returns a new graph with CuSparseMatrixCSR incidence
and CuMatrix features. No-op if CUDA is unavailable.
Uses explicit CUSPARSE conversion for reliable sparse handling.
"""
function to_gpu(graph::LayeredHyperGraph)
    if !HAS_CUDA
        return graph
    end
    gpu_layers = Dict{Symbol, HyperGraphLayer}()
    for (name, layer) in graph.layers
        gpu_layers[name] = _to_gpu_layer(layer)
    end
    LayeredHyperGraph(graph.farm_id, graph.n_vertices, graph.vertex_index, gpu_layers)
end

"""
    to_cpu(graph::LayeredHyperGraph) -> LayeredHyperGraph

Move all layer data back to CPU.  For CPU-resident graphs this is a no-op
(sparse matrices and plain arrays stay as-is).
"""
function to_cpu(graph::LayeredHyperGraph)::LayeredHyperGraph
    cpu_layers = Dict{Symbol, HyperGraphLayer}()
    for (name, layer) in graph.layers
        cpu_layers[name] = _ensure_cpu_layer(layer)
    end
    LayeredHyperGraph(graph.farm_id, graph.n_vertices,
                      copy(graph.vertex_index), cpu_layers)
end

# ---------------------------------------------------------------------------
# Persistent graph cache — GPU-resident graphs keyed by farm_id
# ---------------------------------------------------------------------------

"""Module-level cache: farm_id → GPU-resident LayeredHyperGraph."""
const GRAPH_CACHE = Dict{String, LayeredHyperGraph}()

"""
    cache_graph!(farm_id, graph) -> LayeredHyperGraph

Store a graph in the cache. Automatically moves to GPU if `HAS_CUDA`.
Returns the cached (possibly GPU-resident) graph.
"""
function cache_graph!(farm_id::String, graph::LayeredHyperGraph)::LayeredHyperGraph
    gpu_graph = to_gpu(graph)
    GRAPH_CACHE[farm_id] = gpu_graph
    return gpu_graph
end

"""
    get_cached_graph(farm_id) -> Union{LayeredHyperGraph, Nothing}

Return the cached graph for `farm_id`, or `nothing` if not cached.
"""
function get_cached_graph(farm_id::String)::Union{LayeredHyperGraph, Nothing}
    return get(GRAPH_CACHE, farm_id, nothing)
end

"""
    evict_graph!(farm_id) -> Bool

Remove a graph from the cache. Returns `true` if it was present.
"""
function evict_graph!(farm_id::String)::Bool
    had = haskey(GRAPH_CACHE, farm_id)
    delete!(GRAPH_CACHE, farm_id)
    return had
end

"""
    clear_cache!() -> Nothing

Remove all graphs from the cache.
"""
function clear_cache!()
    empty!(GRAPH_CACHE)
    return nothing
end

# ---------------------------------------------------------------------------
# Cross-layer query
# ---------------------------------------------------------------------------

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
    result = transpose(Ba) * Bb
    return Matrix{Float32}(ensure_cpu(result))
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
    # Pull incidence row to CPU for findnz
    B_cpu = ensure_cpu(lyr.incidence)
    edge_indices = findnz(B_cpu[row, :])[1]
    edge_ids = [lyr.edge_ids[j] for j in edge_indices]
    features = Vector{Float32}(ensure_cpu(lyr.vertex_features)[row, :])

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
    backend = array_backend(lyr.vertex_features)
    # Expand feature matrix if new features are wider
    if n_feat > size(lyr.vertex_features, 2)
        old = lyr.vertex_features
        new_feat = make_dense_zeros(Float32, size(old, 1), n_feat)
        new_feat[:, 1:size(old, 2)] .= old
        lyr.vertex_features = new_feat
    end
    if backend isa CPU
        lyr.vertex_features[row, 1:n_feat] .= features
    else
        lyr.vertex_features[row:row, 1:n_feat] .= CuArray(reshape(Float32.(features), 1, n_feat))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Topology mutation helpers (CPU round-trip for GPU graphs)
# ---------------------------------------------------------------------------

"""Pull a layer's numeric arrays to CPU for topology mutation."""
function _ensure_cpu_layer(layer::HyperGraphLayer)
    B_cpu = ensure_cpu(layer.incidence)
    vf_cpu = Matrix{Float32}(ensure_cpu(layer.vertex_features))
    fh_cpu = Array{Float32,3}(ensure_cpu(layer.feature_history))
    HyperGraphLayer(B_cpu, vf_cpu, fh_cpu,
                    layer.history_head, layer.history_length,
                    layer.edge_metadata, layer.vertex_ids, layer.edge_ids)
end

"""Move a single layer's numeric arrays to GPU. No-op if not HAS_CUDA."""
function _to_gpu_layer(layer::HyperGraphLayer)
    if !HAS_CUDA
        return layer
    end
    B_cpu = ensure_cpu(layer.incidence)
    B_gpu = CUSPARSE.CuSparseMatrixCSR(B_cpu)
    vf_gpu = CuArray(ensure_cpu(layer.vertex_features))
    fh_gpu = CuArray(ensure_cpu(layer.feature_history))
    HyperGraphLayer(B_gpu, vf_gpu, fh_gpu,
                    layer.history_head, layer.history_length,
                    layer.edge_metadata, layer.vertex_ids, layer.edge_ids)
end

"""
    add_hyperedge!(graph, layer::Symbol, edge_id::String, vertex_ids::Vector{String};
                   metadata=Dict{String,Any}())

Add a new hyperedge to an existing layer. Works on both CPU and GPU graphs.
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
        B = make_sparse_incidence(row_inds, ones(Int32, length(row_inds)), nv, 1)
        d = feature_dim(layer)
        vf = make_dense_zeros(Float32, nv, d)
        fh = make_dense_zeros(Float32, nv, d, DEFAULT_HISTORY_SIZE)
        all_vids = collect(keys(graph.vertex_index))
        sort!(all_vids; by=v -> graph.vertex_index[v])
        graph.layers[layer] = HyperGraphLayer(B, vf, fh, 1, 0, [metadata], all_vids, [edge_id])
        return nothing
    end

    lyr = graph.layers[layer]
    cpu_lyr = _ensure_cpu_layer(lyr)
    new_col = zeros(Float32, nv)
    for vid in vertex_ids
        if haskey(graph.vertex_index, vid)
            new_col[graph.vertex_index[vid]] = 1.0f0
        end
    end
    cpu_lyr.incidence = hcat(cpu_lyr.incidence, sparse(new_col))
    push!(cpu_lyr.edge_metadata, metadata)
    push!(cpu_lyr.edge_ids, edge_id)
    graph.layers[layer] = _to_gpu_layer(cpu_lyr)
    return nothing
end

"""
    remove_hyperedge!(graph, layer::Symbol, edge_id::String) -> Bool

Remove a hyperedge from a layer by its edge_id.
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
    cpu_lyr = _ensure_cpu_layer(lyr)
    keep = setdiff(1:size(cpu_lyr.incidence, 2), idx)
    cpu_lyr.incidence = cpu_lyr.incidence[:, keep]
    deleteat!(cpu_lyr.edge_ids, idx)
    deleteat!(cpu_lyr.edge_metadata, idx)
    graph.layers[layer] = _to_gpu_layer(cpu_lyr)
    return true
end

"""
    add_vertex!(graph, vertex_id::String) -> Int

Add a new vertex to the graph. Works on both CPU and GPU graphs.
"""
function add_vertex!(graph::LayeredHyperGraph, vertex_id::String)::Int
    if haskey(graph.vertex_index, vertex_id)
        error("Vertex $vertex_id already exists at index $(graph.vertex_index[vertex_id])")
    end
    graph.n_vertices += 1
    new_idx = graph.n_vertices
    graph.vertex_index[vertex_id] = new_idx

    for (name, lyr) in graph.layers
        cpu_lyr = _ensure_cpu_layer(lyr)
        nv_old, ne = size(cpu_lyr.incidence)
        d = size(cpu_lyr.vertex_features, 2)
        buf_size = size(cpu_lyr.feature_history, 3)
        cpu_lyr.incidence = vcat(cpu_lyr.incidence, spzeros(Float32, 1, ne))
        cpu_lyr.vertex_features = vcat(cpu_lyr.vertex_features, zeros(Float32, 1, d))
        cpu_lyr.feature_history = cat(cpu_lyr.feature_history, zeros(Float32, 1, d, buf_size); dims=1)
        push!(cpu_lyr.vertex_ids, vertex_id)
        graph.layers[name] = _to_gpu_layer(cpu_lyr)
    end
    return new_idx
end

# ---------------------------------------------------------------------------
# Aggregation helpers used by predictive models — SpMM on GPU
# ---------------------------------------------------------------------------

"""
    aggregate_by_edge(layer; reduce=mean) -> Matrix{Float32}

Compute per-hyperedge aggregation of vertex features.
Returns a `(n_edges, d)` CPU matrix where each row is `reduce` of member-vertex features.

For `mean` and `sum`: uses SpMM (`B' * vf`) which runs on CUSPARSE when data is on GPU.
For other reducers: falls back to CPU scalar loop.
"""
function aggregate_by_edge(layer::HyperGraphLayer; reduce::Function=mean)::Matrix{Float32}
    B = layer.incidence           # (n_vertices, n_edges)
    vf = layer.vertex_features    # (n_vertices, d)
    nv, ne = size(B)
    d = size(vf, 2)

    if reduce === mean || reduce === sum
        # SpMM path — works on CPU SparseMatrixCSC and GPU CuSparseMatrixCSR
        raw = transpose(B) * vf   # (ne, d) — sum of member features per edge
        if reduce === mean
            degree = vec(sum(ensure_cpu(B); dims=1))  # (ne,)
            degree_safe = max.(degree, 1.0f0)
            raw_cpu = ensure_cpu(raw)
            return Matrix{Float32}(raw_cpu ./ degree_safe)
        end
        return Matrix{Float32}(ensure_cpu(raw))
    end

    # Fallback CPU path for other reducers
    B_cpu = ensure_cpu(B)
    vf_cpu = ensure_cpu(vf)
    result = zeros(Float32, ne, d)
    for e in 1:ne
        members = findall(!iszero, @view B_cpu[:, e])
        isempty(members) && continue
        for col in 1:d
            vals = Float32[vf_cpu[v, col] for v in members]
            result[e, col] = reduce(vals)
        end
    end
    return result
end

"""
    multi_layer_features(graph, layers::Vector{Symbol}) -> AbstractMatrix{Float32}

Horizontally concatenate vertex features from several layers.
Preserves array backend (CuMatrix on GPU, Matrix on CPU).
"""
function multi_layer_features(graph::LayeredHyperGraph, layers::Vector{Symbol})::AbstractMatrix{Float32}
    cols = AbstractMatrix{Float32}[]
    for lsym in layers
        haskey(graph.layers, lsym) || continue
        push!(cols, graph.layers[lsym].vertex_features)
    end
    isempty(cols) && return make_dense_zeros(Float32, graph.n_vertices, 0)
    return hcat(cols...)
end
