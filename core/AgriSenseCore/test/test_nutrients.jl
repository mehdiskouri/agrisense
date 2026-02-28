using Test
using AgriSenseCore
using SparseArrays
using Statistics

# Helper: build a graph with :npk, :crop_requirements, and optionally :vision layers
function make_nutrient_graph(;
    nv::Int=4,
    current_npk::Matrix{Float32}=Float32[50 30 40; 80 60 70; 10 5 8; 90 80 90],
    required_npk::Matrix{Float32}=Float32[80 60 70; 80 60 70; 80 60 70; 80 60 70],
    growth_progress::Vector{Float32}=Float32[0.5, 0.5, 0.5, 0.5],
    with_vision::Bool=false,
    vision_anomaly_scores::Vector{Float32}=Float32[0.0, 0.0, 0.0, 0.0],
)
    vertices = [Dict{String,Any}("id" => "v$i", "type" => "sensor") for i in 1:nv]
    edges = [
        Dict{String,Any}("id" => "e-npk-1", "layer" => "npk",
             "vertex_ids" => ["v1", "v2"],
             "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-npk-2", "layer" => "npk",
             "vertex_ids" => ["v3", "v4"],
             "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-crop-1", "layer" => "crop_requirements",
             "vertex_ids" => ["v1", "v2"],
             "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-crop-2", "layer" => "crop_requirements",
             "vertex_ids" => ["v3", "v4"],
             "metadata" => Dict{String,Any}()),
    ]
    if with_vision
        push!(edges, Dict{String,Any}("id" => "e-vis-1", "layer" => "vision",
              "vertex_ids" => ["v$i" for i in 1:nv],
              "metadata" => Dict{String,Any}()))
    end

    layers_list = ["npk", "crop_requirements"]
    with_vision && push!(layers_list, "vision")

    config = Dict{String,Any}(
        "farm_id" => "nutrient-test",
        "farm_type" => "greenhouse",
        "active_layers" => layers_list,
        "zones" => [],
        "models" => Dict("irrigation" => false, "nutrients" => true,
                         "yield_forecast" => false, "anomaly_detection" => false),
        "vertices" => vertices,
        "edges" => edges,
    )
    profile = FarmProfile(config)
    graph = build_hypergraph(profile, config["vertices"], config["edges"])
    graph = to_cpu(graph)  # tests validate correctness on CPU; GPU tested in test_gpu.jl

    # Populate :npk features (3-dim: N, P, K)
    for v in 1:nv
        graph.layers[:npk].vertex_features[v, 1:3] .= current_npk[v, :]
    end

    # Populate :crop_requirements features (5-dim)
    for v in 1:nv
        # col 1 = target_yield, col 2 = growth_progress, cols 3-5 = NPK targets
        graph.layers[:crop_requirements].vertex_features[v, 1] = 5.0f0  # target yield
        graph.layers[:crop_requirements].vertex_features[v, 2] = growth_progress[v]
        graph.layers[:crop_requirements].vertex_features[v, 3:5] .= required_npk[v, :]
    end

    # Populate :vision if present
    if with_vision
        for v in 1:nv
            graph.layers[:vision].vertex_features[v, 3] = vision_anomaly_scores[v]
        end
    end

    return graph
end

# ---------------------------------------------------------------------------
@testset "Nutrient Scoring" begin

    @testset "deficit below target reported correctly" begin
        # v1/v2: N deficit = 80-50=30, 80-80=0
        graph = make_nutrient_graph()
        report = compute_nutrient_report(graph)
        @test length(report) == 2  # 2 edges

        # Find zone for e-npk-1 (v1, v2)
        zone1 = findfirst(r -> r["zone_id"] == "e-npk-1", report)
        @test zone1 !== nothing
        r1 = report[zone1]
        # v1: N deficit=30, P deficit=30, K deficit=30
        # v2: N deficit=0, P deficit=0, K deficit=0
        # mean N deficit = 15
        @test r1["nitrogen_deficit"] ≈ 15.0 atol=1.0
        @test r1["phosphorus_deficit"] ≈ 15.0 atol=1.0
        @test r1["potassium_deficit"] ≈ 15.0 atol=1.0
    end

    @testset "no deficit when NPK at target" begin
        npk_at_target = Float32[80 60 70; 80 60 70; 80 60 70; 80 60 70]
        graph = make_nutrient_graph(current_npk=npk_at_target)
        report = compute_nutrient_report(graph)
        for r in report
            @test r["nitrogen_deficit"] ≈ 0.0 atol=0.01
            @test r["phosphorus_deficit"] ≈ 0.0 atol=0.01
            @test r["potassium_deficit"] ≈ 0.0 atol=0.01
            @test r["urgency"] == "low"
        end
    end

    @testset "high deficit severity mapped to critical urgency" begin
        # Very low NPK, all below requirements
        low_npk = Float32[5 2 3; 5 2 3; 5 2 3; 5 2 3]
        high_req = Float32[100 100 100; 100 100 100; 100 100 100; 100 100 100]
        graph = make_nutrient_graph(current_npk=low_npk, required_npk=high_req)
        report = compute_nutrient_report(graph)
        for r in report
            sev = r["severity_score"]
            @test sev > 0.5  # Should be high or critical
            @test r["urgency"] in ["high", "critical"]
        end
    end

    @testset "vision confirmation boosts severity" begin
        # Moderate deficit
        npk = Float32[40 30 35; 40 30 35; 40 30 35; 40 30 35]
        graph_novis = make_nutrient_graph(current_npk=npk, with_vision=false)
        graph_vis = make_nutrient_graph(
            current_npk=npk, with_vision=true,
            vision_anomaly_scores=Float32[0.8, 0.8, 0.8, 0.8],
        )
        report_novis = compute_nutrient_report(graph_novis)
        report_vis   = compute_nutrient_report(graph_vis)

        # Severity with vision should be ≥ severity without
        for i in eachindex(report_novis)
            z_id = report_novis[i]["zone_id"]
            v_idx = findfirst(r -> r["zone_id"] == z_id, report_vis)
            v_idx === nothing && continue
            @test report_vis[v_idx]["severity_score"] >= report_novis[i]["severity_score"]
            @test report_vis[v_idx]["visual_confirmed"] == true
        end
    end

    @testset "suggested amendment reflects deficit nutrients" begin
        # Only N is low
        npk = Float32[20 60 70; 20 60 70; 20 60 70; 20 60 70]
        req = Float32[80 60 70; 80 60 70; 80 60 70; 80 60 70]
        graph = make_nutrient_graph(current_npk=npk, required_npk=req)
        report = compute_nutrient_report(graph)
        for r in report
            @test occursin("nitrogen", r["suggested_amendment"])
            @test !occursin("phosphorus", r["suggested_amendment"])
            @test !occursin("potassium", r["suggested_amendment"])
        end
    end

    @testset "empty graph returns empty report" begin
        config = Dict{String,Any}(
            "farm_id" => "empty",
            "farm_type" => "greenhouse",
            "active_layers" => ["soil"],
            "zones" => [],
            "models" => Dict("irrigation" => false, "nutrients" => true,
                             "yield_forecast" => false, "anomaly_detection" => false),
            "vertices" => [Dict{String,Any}("id" => "v1", "type" => "sensor")],
            "edges" => [Dict{String,Any}("id" => "e1", "layer" => "soil",
                        "vertex_ids" => ["v1"],
                        "metadata" => Dict{String,Any}())],
        )
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        report = compute_nutrient_report(graph)
        @test isempty(report)
    end

    @testset "output has all required keys" begin
        graph = make_nutrient_graph()
        report = compute_nutrient_report(graph)
        required_keys = ["zone_id", "nitrogen_deficit", "phosphorus_deficit",
                         "potassium_deficit", "severity_score", "urgency",
                         "suggested_amendment", "visual_confirmed"]
        for r in report
            for k in required_keys
                @test haskey(r, k)
            end
        end
    end

    @testset "configurable NPK weights change severity" begin
        # Heavy N deficit only
        npk = Float32[20 60 70; 20 60 70; 20 60 70; 20 60 70]
        req = Float32[80 60 70; 80 60 70; 80 60 70; 80 60 70]
        graph = make_nutrient_graph(current_npk=npk, required_npk=req)

        # Default weights (0.50, 0.25, 0.25) — N-heavier
        report_n_heavy = compute_nutrient_report(graph)
        # Equal weights
        report_equal   = compute_nutrient_report(graph;
                              weights=(Float32(1/3), Float32(1/3), Float32(1/3)))
        # P-heavy weights (should lower severity since P is at target)
        report_p_heavy = compute_nutrient_report(graph;
                              weights=(0.10f0, 0.80f0, 0.10f0))

        for i in eachindex(report_n_heavy)
            zid = report_n_heavy[i]["zone_id"]
            eq_idx = findfirst(r -> r["zone_id"] == zid, report_equal)
            ph_idx = findfirst(r -> r["zone_id"] == zid, report_p_heavy)
            eq_idx === nothing && continue
            ph_idx === nothing && continue
            # N-heavy ≥ equal weight severity (N is the deficit nutrient)
            @test report_n_heavy[i]["severity_score"] >= report_equal[eq_idx]["severity_score"] - 0.01
            # P-heavy < N-heavy (P has no deficit)
            @test report_p_heavy[ph_idx]["severity_score"] <= report_n_heavy[i]["severity_score"] + 0.01
        end
    end
end
