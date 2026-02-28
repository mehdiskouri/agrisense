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
