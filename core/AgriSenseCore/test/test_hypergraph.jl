using Test
using AgriSenseCore
using SparseArrays

# Helper: build a standard test config used across many test sets
function make_test_config(;
    n_vertices=3,
    layers=["soil", "irrigation"],
    shared_vertex="v2",
)
    vertices = [Dict{String,Any}("id" => "v$i", "type" => "sensor") for i in 1:n_vertices]
    edges = Dict{String,Any}[]
    # Soil edge spans v1, v2
    if "soil" in layers
        push!(edges, Dict{String,Any}(
            "id" => "e-soil-1", "layer" => "soil",
            "vertex_ids" => ["v1", shared_vertex],
            "metadata" => Dict{String,Any}("depth" => 30),
        ))
    end
    # Irrigation edge spans v2, v3
    if "irrigation" in layers && n_vertices >= 3
        push!(edges, Dict{String,Any}(
            "id" => "e-irr-1", "layer" => "irrigation",
            "vertex_ids" => [shared_vertex, "v3"],
            "metadata" => Dict{String,Any}("valve" => "on"),
        ))
    end
    Dict{String,Any}(
        "farm_id" => "test-farm",
        "farm_type" => "greenhouse",
        "active_layers" => layers,
        "zones" => [Dict("id" => "z1", "name" => "Zone-1", "zone_type" => "greenhouse",
                         "area_m2" => 100.0, "soil_type" => "loam")],
        "models" => Dict("irrigation" => true, "nutrients" => true,
                         "yield_forecast" => true, "anomaly_detection" => true),
        "vertices" => vertices,
        "edges" => edges,
    )
end

# ---------------------------------------------------------------------------
@testset "Hypergraph Construction & Query" begin

    @testset "build_graph returns valid state" begin
        config = make_test_config()
        state = build_graph(config)
        @test state["farm_id"] == "test-farm"
        @test state["n_vertices"] == 3
        @test haskey(state["layers"], "soil")
        @test haskey(state["layers"], "irrigation")
    end

    @testset "incidence matrix dimensions" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        @test size(graph.layers[:soil].incidence) == (3, 1)
        @test size(graph.layers[:irrigation].incidence) == (3, 1)
    end

    @testset "incidence matrix sparsity pattern" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        B_soil = graph.layers[:soil].incidence
        # v1 and v2 should be in edge 1
        @test B_soil[1, 1] == 1.0f0  # v1
        @test B_soil[2, 1] == 1.0f0  # v2
        @test B_soil[3, 1] == 0.0f0  # v3 not in soil edge
    end

    @testset "feature dimensions per layer" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        @test size(graph.layers[:soil].vertex_features, 2) == AgriSenseCore.LAYER_FEATURE_DIMS[:soil]
        @test size(graph.layers[:irrigation].vertex_features, 2) == AgriSenseCore.LAYER_FEATURE_DIMS[:irrigation]
    end

    @testset "feature_dim defaults to 1 for unknown layers" begin
        @test feature_dim(:unknown_layer) == 1
        @test feature_dim(:soil) == 4
        @test feature_dim(:weather) == 5
    end

    @testset "empty layer is skipped" begin
        config = make_test_config(layers=["soil", "weather"])
        # No weather edges → weather layer should not be created
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        @test haskey(graph.layers, :soil)
        @test !haskey(graph.layers, :weather)
    end

    @testset "vertex_index correctness" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        @test graph.vertex_index["v1"] == 1
        @test graph.vertex_index["v2"] == 2
        @test graph.vertex_index["v3"] == 3
        @test graph.n_vertices == 3
    end
end

# ---------------------------------------------------------------------------
@testset "Query Operations" begin

    @testset "query_farm_status returns layer data" begin
        config = make_test_config()
        state = build_graph(config)
        status = query_farm_status(state, "v1")
        @test haskey(status, "soil")
        @test status["soil"]["vertex_id"] == "v1"
    end

    @testset "query_layer returns correct edges for vertex" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        result = query_layer(graph, :soil, "v2")
        @test result["vertex_id"] == "v2"
        @test "e-soil-1" in result["edge_ids"]
        @test length(result["features"]) == AgriSenseCore.LAYER_FEATURE_DIMS[:soil]
    end

    @testset "query_layer missing layer returns error" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        result = query_layer(graph, :nonexistent, "v1")
        @test haskey(result, "error")
        @test occursin("not found", result["error"])
        @test occursin("Available", result["error"])
    end

    @testset "query_layer missing vertex returns error" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        result = query_layer(graph, :soil, "v999")
        @test haskey(result, "error")
        @test occursin("not found", result["error"])
    end

    @testset "cross-layer query dimensions and overlap" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        cross = cross_layer_query(graph, :soil, :irrigation)
        @test size(cross) == (1, 1)
        # v2 is shared → overlap = 1
        @test cross[1, 1] == 1.0f0
    end

    @testset "cross-layer query missing layer throws" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        @test_throws ErrorException cross_layer_query(graph, :soil, :nonexistent)
        @test_throws ErrorException cross_layer_query(graph, :nonexistent, :soil)
    end
end

# ---------------------------------------------------------------------------
@testset "Update Vertex Features" begin

    @testset "basic update" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        feat = Float32[0.5, 0.3, 0.8, 0.1]
        update_vertex_features!(graph, :soil, "v1", feat)
        @test graph.layers[:soil].vertex_features[1, :] == feat
    end

    @testset "auto-expand feature matrix" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        # Soil has 4 dims; push a 6-dim vector → should expand
        big_feat = Float32[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        update_vertex_features!(graph, :soil, "v1", big_feat)
        @test size(graph.layers[:soil].vertex_features, 2) == 6
        @test graph.layers[:soil].vertex_features[1, :] == big_feat
        # Other rows should still be zero-padded
        @test all(graph.layers[:soil].vertex_features[3, :] .== 0.0f0)
    end
end

# ---------------------------------------------------------------------------
@testset "Add Hyperedge" begin

    @testset "add to existing layer" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        @test size(graph.layers[:soil].incidence, 2) == 1
        add_hyperedge!(graph, :soil, "e-soil-2", ["v1", "v3"];
                       metadata=Dict{String,Any}("new" => true))
        @test size(graph.layers[:soil].incidence, 2) == 2
        @test "e-soil-2" in graph.layers[:soil].edge_ids
        # v1 and v3 should be in the new column
        B = graph.layers[:soil].incidence
        @test B[1, 2] == 1.0f0  # v1
        @test B[2, 2] == 0.0f0  # v2
        @test B[3, 2] == 1.0f0  # v3
    end

    @testset "add to new layer creates it" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        @test !haskey(graph.layers, :weather)
        add_hyperedge!(graph, :weather, "e-weather-1", ["v1", "v2"])
        @test haskey(graph.layers, :weather)
        @test size(graph.layers[:weather].incidence) == (3, 1)
        @test size(graph.layers[:weather].vertex_features, 2) == AgriSenseCore.LAYER_FEATURE_DIMS[:weather]
    end

    @testset "skips unknown vertex IDs" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        add_hyperedge!(graph, :soil, "e-soil-skip", ["v1", "v999"])
        B = graph.layers[:soil].incidence
        # Only v1 should be present in the new column
        @test B[1, 2] == 1.0f0
        @test nnz(B[:, 2]) == 1
    end
end

# ---------------------------------------------------------------------------
@testset "Remove Hyperedge" begin

    @testset "remove existing edge" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        @test remove_hyperedge!(graph, :soil, "e-soil-1") == true
        @test size(graph.layers[:soil].incidence, 2) == 0
        @test isempty(graph.layers[:soil].edge_ids)
    end

    @testset "remove non-existent edge returns false" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        @test remove_hyperedge!(graph, :soil, "e-nope") == false
    end

    @testset "remove from non-existent layer returns false" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        @test remove_hyperedge!(graph, :nonexistent, "e-soil-1") == false
    end
end

# ---------------------------------------------------------------------------
@testset "Add Vertex" begin

    @testset "add new vertex expands all layers" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        idx = add_vertex!(graph, "v4")
        @test idx == 4
        @test graph.n_vertices == 4
        @test graph.vertex_index["v4"] == 4
        @test size(graph.layers[:soil].incidence, 1) == 4
        @test size(graph.layers[:irrigation].incidence, 1) == 4
        @test size(graph.layers[:soil].vertex_features, 1) == 4
    end

    @testset "add duplicate vertex throws" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        @test_throws ErrorException add_vertex!(graph, "v1")
    end
end

# ---------------------------------------------------------------------------
@testset "GPU / CPU Round-trip" begin

    @testset "to_gpu and to_cpu are identity on CPU backend" begin
        config = make_test_config()
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        gpu_graph = to_gpu(graph)
        cpu_graph = to_cpu(gpu_graph)
        @test cpu_graph.farm_id == graph.farm_id
        @test cpu_graph.n_vertices == graph.n_vertices
        @test Set(keys(cpu_graph.layers)) == Set(keys(graph.layers))
        for (name, lyr) in cpu_graph.layers
            orig = graph.layers[name]
            @test Array(lyr.incidence) == Array(orig.incidence)
            @test lyr.vertex_features == orig.vertex_features
            @test lyr.vertex_ids == orig.vertex_ids
            @test lyr.edge_ids == orig.edge_ids
        end
    end

    @testset "backend detection" begin
        backend = get_backend()
        # Just verify it returns something callable — type depends on CUDA availability
        @test backend isa Any
    end
end
