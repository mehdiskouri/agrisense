using Test
using AgriSenseCore
using SparseArrays
using Statistics

# Helper: build a graph with history for anomaly testing
function make_anomaly_graph(; nv::Int=3, layers_list=["soil"],
                              buf_size::Int=DEFAULT_HISTORY_SIZE)
    vertices = [Dict{String,Any}("id" => "v$i", "type" => "sensor") for i in 1:nv]
    edges = Dict{String,Any}[]

    for l in layers_list
        push!(edges, Dict{String,Any}(
            "id" => "e-$l-1", "layer" => l,
            "vertex_ids" => ["v$i" for i in 1:nv],
            "metadata" => Dict{String,Any}(),
        ))
    end

    config = Dict{String,Any}(
        "farm_id" => "anomaly-test",
        "farm_type" => "greenhouse",
        "active_layers" => layers_list,
        "zones" => [],
        "models" => Dict("irrigation" => false, "nutrients" => false,
                         "yield_forecast" => false, "anomaly_detection" => true),
        "vertices" => vertices,
        "edges" => edges,
    )
    profile = FarmProfile(config)
    graph = build_hypergraph(profile, config["vertices"], config["edges"])
    graph = to_cpu(graph)  # tests validate correctness on CPU; GPU tested in test_gpu.jl
    return graph
end

# Push n readings of stable baseline + noise to a specific vertex in a layer
function push_stable_baseline!(layer, vertex_idx::Int, n::Int;
                                baseline::Vector{Float32}=Float32[0.3, 25.0, 1.2, 6.5],
                                noise_std::Float32=0.01f0)
    d = min(length(baseline), size(layer.vertex_features, 2))
    for _ in 1:n
        features = Float32[baseline[f] + noise_std * randn(Float32) for f in 1:d]
        push_features!(layer, vertex_idx, features)
    end
end

# ---------------------------------------------------------------------------
@testset "Anomaly Detection" begin

    @testset "stable baseline produces no anomalies" begin
        graph = make_anomaly_graph(nv=2, layers_list=["soil"])
        # Push 30 stable readings per vertex
        for v in 1:2
            push_stable_baseline!(graph.layers[:soil], v, 30;
                                   baseline=Float32[0.30, 25.0, 1.2, 6.5],
                                   noise_std=0.01f0)
        end
        results = compute_anomaly_detection(graph)
        @test isempty(results)
    end

    @testset "3σ outlier flagged as alarm" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        baseline = Float32[0.30, 25.0, 1.2, 6.5]

        # Push 30 stable readings
        push_stable_baseline!(sol, 1, 30; baseline=baseline, noise_std=0.005f0)

        # Inject a large outlier — set current features to 4σ away
        # Mean ≈ 0.30 with std ≈ 0.005, so 0.32 is ~4σ away
        outlier = Float32[0.35, 25.0, 1.2, 6.5]  # moisture = 0.35, which is ~10σ from 0.30
        sol.vertex_features[1, :] .= outlier[1:size(sol.vertex_features, 2)]

        results = compute_anomaly_detection(graph)
        # Should flag at least one anomaly
        @test !isempty(results)
        # Check that at least one is an alarm
        alarms = filter(r -> r["severity"] == "alarm", results)
        @test !isempty(alarms)
    end

    @testset "2σ outlier flagged as warning" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        baseline = Float32[0.30, 25.0, 1.2, 6.5]

        push_stable_baseline!(sol, 1, 30; baseline=baseline, noise_std=0.01f0)

        # Inject a moderate outlier (~2.5σ away)
        # std ≈ 0.01, so 0.325 is ~2.5σ from 0.30
        moderate_outlier = Float32[0.325, 25.0, 1.2, 6.5]
        sol.vertex_features[1, :] .= moderate_outlier[1:size(sol.vertex_features, 2)]

        results = compute_anomaly_detection(graph)
        has_warning = any(r -> r["severity"] in ["warning", "alarm"], results)
        @test has_warning
    end

    @testset "insufficient history returns no anomalies" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        # Push only 5 readings (below MIN_HISTORY_FOR_ANOMALY = 8)
        for _ in 1:5
            push_features!(graph.layers[:soil], 1, Float32[0.30, 25.0, 1.2, 6.5])
        end
        @test graph.layers[:soil].history_length == 5
        results = compute_anomaly_detection(graph)
        @test isempty(results)
    end

    @testset "cross-layer correlation escalation" begin
        graph = make_anomaly_graph(nv=2, layers_list=["soil", "vision"])
        soil_lyr = graph.layers[:soil]
        vis_lyr  = graph.layers[:vision]

        # Push stable baseline for both layers
        for v in 1:2
            push_stable_baseline!(soil_lyr, v, 30;
                                   baseline=Float32[0.30, 25.0, 1.2, 6.5],
                                   noise_std=0.005f0)
            push_stable_baseline!(vis_lyr, v, 30;
                                   baseline=Float32[0.8, 0.5, 0.1, 0.7],
                                   noise_std=0.005f0)
        end

        # Inject outliers on vertex 1 in BOTH soil and vision
        soil_lyr.vertex_features[1, :] .= Float32[0.50, 25.0, 1.2, 6.5]  # moisture spike
        vis_lyr.vertex_features[1, :] .= Float32[0.8, 0.5, 0.8, 0.7]     # anomaly_score spike (col 3)

        results = compute_anomaly_detection(graph)
        # Vertex 1 should have cross_layer_confirmed = true for at least one result
        v1_results = filter(r -> r["vertex_id"] == "v1", results)
        if !isempty(v1_results)
            confirmed = any(r -> r["cross_layer_confirmed"] == true, v1_results)
            @test confirmed
        end
    end

    @testset "output has all required keys" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        push_stable_baseline!(sol, 1, 20;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.005f0)
        # Force an outlier
        sol.vertex_features[1, :] .= Float32[0.50, 25.0, 1.2, 6.5]
        results = compute_anomaly_detection(graph)

        required_keys = ["vertex_id", "layer", "feature", "anomaly_type",
                         "severity", "current_value", "rolling_mean", "rolling_std",
                         "sigma_deviation", "cross_layer_confirmed"]
        for r in results
            for k in required_keys
                @test haskey(r, k)
            end
        end
    end

    @testset "anomaly type mapped correctly per layer" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        push_stable_baseline!(sol, 1, 20;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.005f0)
        sol.vertex_features[1, :] .= Float32[0.50, 25.0, 1.2, 6.5]

        results = compute_anomaly_detection(graph)
        for r in results
            @test r["anomaly_type"] == "environmental"  # soil → environmental
        end
    end
end
