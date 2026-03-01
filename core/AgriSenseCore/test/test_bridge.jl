using Test
using AgriSenseCore
using SparseArrays

# Helper: build a graph for serialization tests
function make_bridge_graph()
    config = Dict{String,Any}(
        "farm_id" => "bridge-farm",
        "farm_type" => "greenhouse",
        "active_layers" => ["soil", "irrigation"],
        "zones" => [Dict("id" => "z1", "name" => "Z1", "zone_type" => "greenhouse",
                         "area_m2" => 100.0)],
        "models" => Dict{String,Any}(),
        "vertices" => [
            Dict{String,Any}("id" => "v1", "type" => "sensor"),
            Dict{String,Any}("id" => "v2", "type" => "sensor"),
            Dict{String,Any}("id" => "v3", "type" => "valve"),
        ],
        "edges" => [
            Dict{String,Any}("id" => "e1", "layer" => "soil",
                 "vertex_ids" => ["v1", "v2"],
                 "metadata" => Dict{String,Any}("depth" => 20)),
            Dict{String,Any}("id" => "e2", "layer" => "irrigation",
                 "vertex_ids" => ["v2", "v3"],
                 "metadata" => Dict{String,Any}()),
        ],
    )
    profile = FarmProfile(config)
    graph = build_hypergraph(profile, config["vertices"], config["edges"])
    to_cpu(graph)  # bridge tests validate serialization on CPU arrays
end

@testset "Serialization Round-trip" begin

    @testset "serialize → deserialize preserves topology" begin
        graph = make_bridge_graph()
        state = serialize_graph(graph)
        restored = deserialize_graph(state)

        @test restored.farm_id == graph.farm_id
        @test restored.n_vertices == graph.n_vertices
        @test Set(keys(restored.vertex_index)) == Set(keys(graph.vertex_index))
        @test Set(keys(restored.layers)) == Set(keys(graph.layers))

        for (name, lyr) in restored.layers
            orig = graph.layers[name]
            @test Array(lyr.incidence) == Array(orig.incidence)
            @test lyr.vertex_features ≈ orig.vertex_features
            @test lyr.edge_ids == orig.edge_ids
            @test lyr.vertex_ids == orig.vertex_ids
        end
    end

    @testset "serialize → deserialize preserves edge metadata" begin
        graph = make_bridge_graph()
        state = serialize_graph(graph)
        restored = deserialize_graph(state)
        soil = restored.layers[:soil]
        @test soil.edge_metadata[1]["depth"] == 20
    end

    @testset "serialize produces only plain types" begin
        graph = make_bridge_graph()
        state = serialize_graph(graph)

        @test state isa Dict{String,Any}
        @test state["farm_id"] isa String
        @test state["n_vertices"] isa Int
        @test state["layers"] isa Dict

        soil_layer = state["layers"]["soil"]
        @test soil_layer["incidence_rows"] isa Vector
        @test soil_layer["incidence_cols"] isa Vector
        @test soil_layer["incidence_vals"] isa Vector
        @test soil_layer["vertex_features"] isa Matrix
    end

    @testset "round-trip after topology mutation" begin
        graph = make_bridge_graph()
        # Mutate: add a vertex and an edge
        add_vertex!(graph, "v4")
        add_hyperedge!(graph, :soil, "e-new", ["v1", "v4"])

        state = serialize_graph(graph)
        restored = deserialize_graph(state)

        @test restored.n_vertices == 4
        @test haskey(restored.vertex_index, "v4")
        @test size(restored.layers[:soil].incidence, 2) == 2
        @test "e-new" in restored.layers[:soil].edge_ids
    end
end

@testset "Bridge API via build_graph" begin

    @testset "build_graph end-to-end" begin
        config = Dict{String,Any}(
            "farm_id" => "api-farm",
            "farm_type" => "open_field",
            "active_layers" => ["soil"],
            "zones" => [],
            "models" => Dict{String,Any}(),
            "vertices" => [
                Dict{String,Any}("id" => "a", "type" => "sensor"),
                Dict{String,Any}("id" => "b", "type" => "sensor"),
            ],
            "edges" => [
                Dict{String,Any}("id" => "e1", "layer" => "soil",
                     "vertex_ids" => ["a", "b"],
                     "metadata" => Dict{String,Any}()),
            ],
        )
        state = build_graph(config)
        @test state["farm_id"] == "api-farm"
        @test state["n_vertices"] == 2
        @test haskey(state["layers"], "soil")
    end

    @testset "query_farm_status works on serialized state" begin
        config = Dict{String,Any}(
            "farm_id" => "qf",
            "farm_type" => "greenhouse",
            "active_layers" => ["soil"],
            "zones" => [],
            "models" => Dict{String,Any}(),
            "vertices" => [Dict{String,Any}("id" => "v1", "type" => "s")],
            "edges" => [Dict{String,Any}("id" => "e1", "layer" => "soil",
                        "vertex_ids" => ["v1"], "metadata" => Dict{String,Any}())],
        )
        state = build_graph(config)
        status = query_farm_status(state, "v1")
        @test haskey(status, "soil")
        @test status["soil"]["vertex_id"] == "v1"
    end
end

@testset "Deserialize Error Handling" begin

    @testset "missing required key raises" begin
        bad = Dict{String,Any}("farm_id" => "x")
        @test_throws ErrorException deserialize_graph(bad)
    end

    @testset "corrupt layer data raises with context" begin
        bad = Dict{String,Any}(
            "farm_id" => "x",
            "n_vertices" => 2,
            "vertex_index" => Dict("a" => 1, "b" => 2),
            "layers" => Dict{String,Any}(
                "soil" => Dict{String,Any}(
                    "incidence_rows" => "not_an_array",  # corrupt!
                ),
            ),
        )
        @test_throws ErrorException deserialize_graph(bad)
    end
end

# ===========================================================================
# Phase 13 — Mask Serialization Tests
# ===========================================================================
@testset "Phase 13 — Mask Serialization" begin

    @testset "serialize includes mask fields" begin
        graph = make_bridge_graph()
        # Push some data including NaN
        sol = graph.layers[:soil]
        push_features!(sol, 1, Float32[0.3, 25.0, 1.2, 6.5])
        push_features!(sol, 1, Float32[NaN, 24.0, NaN, 6.0])

        state = serialize_graph(graph)
        soil_ld = state["layers"]["soil"]
        @test haskey(soil_ld, "feature_history_mask")
        @test haskey(soil_ld, "vertex_features_mask")
        @test soil_ld["feature_history_mask"] isa Array{Bool, 3}
        @test soil_ld["vertex_features_mask"] isa Array{Bool, 2}
    end

    @testset "round-trip preserves masks" begin
        graph = make_bridge_graph()
        sol = graph.layers[:soil]
        push_features!(sol, 1, Float32[0.3, 25.0, 1.2, 6.5])
        push_features!(sol, 1, Float32[NaN, 24.0, NaN, 6.0])

        state = serialize_graph(graph)
        restored = deserialize_graph(state)

        orig_fhm = graph.layers[:soil].feature_history_mask
        rest_fhm = restored.layers[:soil].feature_history_mask
        @test orig_fhm == rest_fhm

        orig_vfm = graph.layers[:soil].vertex_features_mask
        rest_vfm = restored.layers[:soil].vertex_features_mask
        @test orig_vfm == rest_vfm
    end

    @testset "backward compat: legacy state without masks defaults to all-true" begin
        graph = make_bridge_graph()
        state = serialize_graph(graph)
        # Remove mask fields to simulate legacy data
        for (_, ld) in state["layers"]
            delete!(ld, "feature_history_mask")
            delete!(ld, "vertex_features_mask")
        end
        restored = deserialize_graph(state)
        for (_, lyr) in restored.layers
            # Legacy data: assume all readings valid → masks filled with true
            @test all(lyr.feature_history_mask)
            @test all(lyr.vertex_features_mask)
        end
    end

    # ── Phase 14: Lightweight ack & batch update ──────────────────────────
    @testset "Phase 14: update_features returns lightweight ack" begin
        graph = make_bridge_graph()
        state = serialize_graph(graph)
        ack = update_features(state, "soil", "v1", [0.5, 22.0, 1.0, 6.5])
        @test ack["ok"] == true
        @test haskey(ack, "farm_id")
        @test haskey(ack, "version")
        @test ack["layer"] == "soil"
        @test ack["vertex_id"] == "v1"
        # ack must NOT contain full graph state
        @test !haskey(ack, "layers")
        @test !haskey(ack, "n_vertices")
    end

    @testset "Phase 14: graph_version increments on cache" begin
        farm_id = "test-phase14-version-$(rand(1:100000))"
        graph = make_bridge_graph()
        v0 = graph_version(farm_id)
        @test v0 == 0
        cache_graph!(farm_id, graph)
        v1 = graph_version(farm_id)
        @test v1 >= 1
        cache_graph!(farm_id, graph)
        v2 = graph_version(farm_id)
        @test v2 > v1
        evict_graph!(farm_id)
        @test graph_version(farm_id) == 0
    end

    @testset "Phase 14: batch_update_features" begin
        farm_id = "test-phase14-batch-$(rand(1:100000))"
        graph = make_bridge_graph()
        cache_graph!(farm_id, graph)

        updates = [
            Dict{String,Any}("layer" => "soil", "vertex_id" => "v1", "features" => [0.1, 20.0, 0.5, 6.0]),
            Dict{String,Any}("layer" => "soil", "vertex_id" => "v2", "features" => [0.2, 21.0, 0.6, 6.1]),
            Dict{String,Any}("layer" => "irrigation", "vertex_id" => "v3", "features" => [25.0, 60.0, 0.0]),
        ]
        ack = batch_update_features(farm_id, updates)
        @test ack["ok"] == true
        @test ack["n_updated"] == 3
        @test ack["farm_id"] == farm_id
        @test haskey(ack, "version")

        evict_graph!(farm_id)
    end

    @testset "Phase 14: get_graph_by_id" begin
        farm_id = "test-phase14-getgraph-$(rand(1:100000))"
        graph = make_bridge_graph()
        cache_graph!(farm_id, graph)

        state = get_graph_by_id(farm_id)
        @test haskey(state, "layers")
        @test haskey(state, "n_vertices")
        @test haskey(state, "farm_id")
        @test state["farm_id"] == farm_id

        evict_graph!(farm_id)
    end

    @testset "Phase 14: ensure_graph" begin
        farm_id = "test-phase14-ensure-$(rand(1:100000))"
        # Not cached → should error
        @test_throws ErrorException ensure_graph(farm_id)

        graph = make_bridge_graph()
        cache_graph!(farm_id, graph)
        ack = ensure_graph(farm_id)
        @test ack["ok"] == true
        @test ack["cached"] == true
        @test ack["farm_id"] == farm_id

        evict_graph!(farm_id)
    end

    @testset "Phase 14: batch_update_features on uncached graph errors" begin
        farm_id = "test-phase14-batch-nocache-$(rand(1:100000))"
        updates = [
            Dict{String,Any}("layer" => "soil", "vertex_id" => "v1", "features" => [0.1, 20.0, 0.5, 6.0]),
        ]
        @test_throws ErrorException batch_update_features(farm_id, updates)
    end
end
