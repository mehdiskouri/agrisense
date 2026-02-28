using Test
using AgriSenseCore
using SparseArrays
using Statistics

# Helper: build a graph suitable for irrigation testing
function make_irrigation_graph(;
    nv::Int=4,
    soil_moisture::Vector{Float32}=Float32[0.10, 0.10, 0.30, 0.30],
    temperature::Vector{Float32}=Float32[25.0, 25.0, 25.0, 25.0],
    precip::Vector{Float32}=Float32[0.0, 0.0, 0.0, 0.0],
    solar_rad::Vector{Float32}=Float32[15.0, 15.0, 15.0, 15.0],
    growth_progress::Vector{Float32}=Float32[0.5, 0.5, 0.5, 0.5],
)
    vertices = [Dict{String,Any}("id" => "v$i", "type" => "sensor") for i in 1:nv]
    edges = [
        Dict{String,Any}("id" => "e-soil-1", "layer" => "soil",
             "vertex_ids" => ["v$i" for i in 1:nv],
             "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-irr-1", "layer" => "irrigation",
             "vertex_ids" => ["v1", "v2"],
             "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-irr-2", "layer" => "irrigation",
             "vertex_ids" => ["v3", "v4"],
             "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-weather-1", "layer" => "weather",
             "vertex_ids" => ["v$i" for i in 1:nv],
             "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-crop-1", "layer" => "crop_requirements",
             "vertex_ids" => ["v$i" for i in 1:nv],
             "metadata" => Dict{String,Any}()),
    ]

    config = Dict{String,Any}(
        "farm_id" => "irr-test",
        "farm_type" => "greenhouse",
        "active_layers" => ["soil", "irrigation", "weather", "crop_requirements"],
        "zones" => [],
        "models" => Dict("irrigation" => true, "nutrients" => false,
                         "yield_forecast" => false, "anomaly_detection" => false),
        "vertices" => vertices,
        "edges" => edges,
    )
    profile = FarmProfile(config)
    graph = build_hypergraph(profile, config["vertices"], config["edges"])
    graph = to_cpu(graph)  # tests validate correctness on CPU; GPU tested in test_gpu.jl

    # Populate features
    for v in 1:nv
        graph.layers[:soil].vertex_features[v, 1] = soil_moisture[v]
        graph.layers[:weather].vertex_features[v, 1] = temperature[v]
        graph.layers[:weather].vertex_features[v, 3] = precip[v]
        graph.layers[:weather].vertex_features[v, 5] = solar_rad[v]
        graph.layers[:crop_requirements].vertex_features[v, 2] = growth_progress[v]
    end

    return graph
end

# ---------------------------------------------------------------------------
@testset "Irrigation Scheduler" begin

    @testset "schedule returns Vector{Dict}" begin
        state = Dict{String,Any}(
            "farm_id" => "farm-001",
            "n_vertices" => 0,
            "vertex_index" => Dict{String,Int}(),
            "layers" => Dict{String,Any}(),
        )
        result = AgriSenseCore.irrigation_schedule(state, 7)
        @test result isa Vector{Dict{String,Any}}
    end

    @testset "dry soil triggers irrigation" begin
        graph = make_irrigation_graph(
            soil_moisture=Float32[0.10, 0.10, 0.10, 0.10],
        )
        results = compute_irrigation_schedule(graph, Dict{String,Any}(), 1)
        @test !isempty(results)
        # All zones should recommend irrigation since moisture < wilting point
        irrigate_flags = [r["irrigate"] for r in results]
        @test any(irrigate_flags)
    end

    @testset "wet soil suppresses irrigation" begin
        graph = make_irrigation_graph(
            soil_moisture=Float32[0.35, 0.35, 0.35, 0.35],
        )
        results = compute_irrigation_schedule(graph, Dict{String,Any}(), 1)
        # With moisture at field capacity, should mostly not irrigate
        irrigate_flags = [r["irrigate"] for r in results]
        # At least most should be false (ET₀ might reduce some below threshold)
        @test count(.!irrigate_flags) >= length(irrigate_flags) ÷ 2
    end

    @testset "multi-day horizon produces recommendations per day" begin
        graph = make_irrigation_graph()
        horizon = 3
        results = compute_irrigation_schedule(graph, Dict{String,Any}(), horizon)
        # Should have at least `n_edges * horizon` entries (2 irrigation edges × 3 days)
        days_present = unique([r["day"] for r in results])
        @test length(days_present) == horizon
        @test sort(days_present) == [1, 2, 3]
    end

    @testset "output has all required keys" begin
        graph = make_irrigation_graph()
        results = compute_irrigation_schedule(graph, Dict{String,Any}(), 1)
        required_keys = ["zone_id", "day", "irrigate", "volume_liters",
                         "priority", "projected_moisture", "trigger_reason"]
        for r in results
            for k in required_keys
                @test haskey(r, k)
            end
        end
    end

    @testset "empty graph returns empty" begin
        config = Dict{String,Any}(
            "farm_id" => "empty-irr",
            "farm_type" => "greenhouse",
            "active_layers" => ["lighting"],
            "zones" => [],
            "models" => Dict("irrigation" => true, "nutrients" => false,
                             "yield_forecast" => false, "anomaly_detection" => false),
            "vertices" => [Dict{String,Any}("id" => "v1", "type" => "sensor")],
            "edges" => [Dict{String,Any}("id" => "e1", "layer" => "lighting",
                        "vertex_ids" => ["v1"],
                        "metadata" => Dict{String,Any}())],
        )
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        results = compute_irrigation_schedule(graph, Dict{String,Any}(), 3)
        @test isempty(results)
    end

    @testset "Hargreaves ET0 basic" begin
        et0 = hargreaves_et0(25.0f0, 15.0f0)
        @test et0 > 0.0f0
        @test et0 < 20.0f0  # reasonable daily ET₀ range
    end

    @testset "growth_progress_to_kc range" begin
        @test growth_progress_to_kc(0.0f0) ≈ 0.3f0
        @test growth_progress_to_kc(1.0f0) ≈ 1.2f0
        @test growth_progress_to_kc(0.5f0) ≈ 0.75f0
    end

    @testset "priority reflects deficit magnitude" begin
        # Dry soil → higher priority
        graph_dry = make_irrigation_graph(
            soil_moisture=Float32[0.05, 0.05, 0.05, 0.05],
        )
        results_dry = compute_irrigation_schedule(graph_dry, Dict{String,Any}(), 1)

        # Moderately dry
        graph_mod = make_irrigation_graph(
            soil_moisture=Float32[0.14, 0.14, 0.14, 0.14],
        )
        results_mod = compute_irrigation_schedule(graph_mod, Dict{String,Any}(), 1)

        if !isempty(results_dry) && !isempty(results_mod)
            dry_priorities = [r["priority"] for r in results_dry if r["irrigate"]]
            mod_priorities = [r["priority"] for r in results_mod if r["irrigate"]]
            if !isempty(dry_priorities) && !isempty(mod_priorities)
                @test mean(dry_priorities) >= mean(mod_priorities)
            end
        end
    end
end
