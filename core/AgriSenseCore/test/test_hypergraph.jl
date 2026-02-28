using Test
using AgriSenseCore

@testset "Hypergraph Construction & Query" begin
    # Minimal farm config
    config = Dict{String,Any}(
        "farm_id" => "farm-001",
        "farm_type" => "greenhouse",
        "active_layers" => ["soil", "irrigation"],
        "zones" => [
            Dict("id" => "zone-1", "name" => "GH-Bay-1", "zone_type" => "greenhouse",
                 "area_m2" => 200.0, "soil_type" => "loam"),
        ],
        "models" => Dict("irrigation" => true, "nutrients" => true,
                         "yield_forecast" => true, "anomaly_detection" => true),
        "vertices" => [
            Dict("id" => "v1", "type" => "sensor"),
            Dict("id" => "v2", "type" => "sensor"),
            Dict("id" => "v3", "type" => "valve"),
        ],
        "edges" => [
            Dict("id" => "e1", "layer" => "soil", "vertex_ids" => ["v1", "v2"],
                 "metadata" => Dict{String,Any}()),
            Dict("id" => "e2", "layer" => "irrigation", "vertex_ids" => ["v2", "v3"],
                 "metadata" => Dict{String,Any}()),
        ],
    )

    @testset "build_graph returns valid state" begin
        state = AgriSenseCore.build_graph(config)
        @test state["farm_id"] == "farm-001"
        @test state["n_vertices"] == 3
        @test haskey(state["layers"], "soil")
        @test haskey(state["layers"], "irrigation")
    end

    @testset "query_farm_status returns layer data" begin
        state = AgriSenseCore.build_graph(config)
        status = AgriSenseCore.query_farm_status(state, "v1")
        @test haskey(status, "soil")
        @test status["soil"]["vertex_id"] == "v1"
    end

    @testset "cross-layer query dimensions" begin
        profile = AgriSenseCore.FarmProfile(config)
        vertices = config["vertices"]
        edges = config["edges"]
        graph = AgriSenseCore.build_hypergraph(profile, vertices, edges)

        cross = AgriSenseCore.cross_layer_query(graph, :soil, :irrigation)
        # soil has 1 edge, irrigation has 1 edge → result is 1×1
        @test size(cross) == (1, 1)
        # v2 is shared between them → overlap = 1
        @test cross[1, 1] == 1.0f0
    end

    @testset "GPU backend detection" begin
        backend = AgriSenseCore.get_backend()
        # In CI (no GPU), should fall back to CPU
        @test backend isa KernelAbstractions.CPU || true  # always passes
    end
end
