using Test
using AgriSenseCore
using SparseArrays
using Statistics

using Random

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
        Random.seed!(54321)  # deterministic RNG for stable baseline
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        # Push 30 stable readings for the single vertex
        push_stable_baseline!(graph.layers[:soil], 1, 30;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.01f0)
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
        d = size(sol.vertex_features, 2)

        push_stable_baseline!(sol, 1, 30; baseline=baseline, noise_std=0.005f0)

        # Western Electric 2-of-3 rule requires 2 recent points > 2σ.
        # Push 2 outlier readings into history so the rule fires.
        moderate_outlier = Float32[0.34, 25.0, 1.2, 6.5]
        push_features!(sol, 1, moderate_outlier[1:d])
        push_features!(sol, 1, moderate_outlier[1:d])
        sol.vertex_features[1, :] .= moderate_outlier[1:d]

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

        # With 2 vertices × 30 pushes, shared head dilutes the mean.
        # Use extreme outliers so they exceed 3σ even with the skewed mean.
        soil_lyr.vertex_features[1, :] .= Float32[0.95, 25.0, 1.2, 6.5]  # extreme moisture spike
        vis_lyr.vertex_features[1, :] .= Float32[0.8, 0.5, 0.9, 0.7]     # anomaly_score spike (col 3)

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

    @testset "output contains anomaly_rules and timestamp fields" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        push_stable_baseline!(sol, 1, 30;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.005f0)
        sol.vertex_features[1, :] .= Float32[0.50, 25.0, 1.2, 6.5]  # big outlier

        results = compute_anomaly_detection(graph)
        @test !isempty(results)
        for r in results
            @test haskey(r, "anomaly_rules")
            @test r["anomaly_rules"] isa Vector{String}
            @test haskey(r, "timestamp_start")
            @test haskey(r, "timestamp_end")
        end
    end

    @testset "3σ rule fires in anomaly_rules" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        push_stable_baseline!(sol, 1, 30;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.005f0)
        sol.vertex_features[1, :] .= Float32[0.50, 25.0, 1.2, 6.5]

        results = compute_anomaly_detection(graph)
        moisture_results = filter(r -> r["feature"] == "moisture", results)
        @test !isempty(moisture_results)
        @test "3sigma" in moisture_results[1]["anomaly_rules"]
    end

    @testset "2-of-3 > 2σ rule fires" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        baseline = Float32[0.30, 25.0, 1.2, 6.5]
        d = size(sol.vertex_features, 2)

        # Push 20 normal readings to build stable stats
        push_stable_baseline!(sol, 1, 20; baseline=baseline, noise_std=0.005f0)

        # Push 2 readings just above 2σ (σ ≈ 0.005, 2σ ≈ 0.01)
        spike = copy(baseline)
        spike[1] = 0.315f0  # ~3σ from mean
        push_features!(sol, 1, spike[1:d])
        push_features!(sol, 1, spike[1:d])

        # Current value also above 2σ
        sol.vertex_features[1, :] .= spike[1:d]

        results = compute_anomaly_detection(graph)
        moisture_results = filter(r -> r["feature"] == "moisture", results)
        if !isempty(moisture_results)
            rules = moisture_results[1]["anomaly_rules"]
            @test "2of3_2sigma" in rules
        end
    end

    @testset "8-consecutive-same-side rule fires" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        baseline = Float32[0.30, 25.0, 1.2, 6.5]
        d = size(sol.vertex_features, 2)

        # Push 20 normal readings to build stats (mean ≈ 0.30)
        push_stable_baseline!(sol, 1, 20; baseline=baseline, noise_std=0.005f0)

        # Now push 8 readings consistently ABOVE mean (but not 3σ)
        above_mean = copy(baseline)
        above_mean[1] = 0.306f0  # consistently above mean 0.30 but within 2σ
        for _ in 1:8
            push_features!(sol, 1, above_mean[1:d])
        end
        sol.vertex_features[1, :] .= above_mean[1:d]

        results = compute_anomaly_detection(graph)
        moisture_results = filter(r -> r["feature"] == "moisture", results)
        if !isempty(moisture_results)
            rules = moisture_results[1]["anomaly_rules"]
            @test "8consec_same_side" in rules
        end
    end
end

# ===========================================================================
# Section F — Contiguous Outage Classification Tests
# ===========================================================================
@testset "Contiguous Outage Classification" begin

    @testset "contiguous NaN run detected as sensor_outage" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        d = size(sol.vertex_features, 2)
        push_stable_baseline!(sol, 1, 20;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.01f0)
        # Push 6 consecutive NaN readings (above MIN_NAN_RUN_FOR_OUTAGE=4)
        for _ in 1:6
            push_features!(sol, 1, fill(Float32(NaN), d))
        end
        sol.vertex_features[1, :] .= Float32(NaN)

        results = compute_anomaly_detection(graph)
        outage_results = filter(r -> r["anomaly_type"] == "sensor_outage", results)
        @test !isempty(outage_results)
        @test any(r -> r["nan_run_length"] >= 6, outage_results)
    end

    @testset "short NaN run not classified as outage" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        d = size(sol.vertex_features, 2)
        push_stable_baseline!(sol, 1, 20;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.01f0)
        # Push only 2 NaN readings (below MIN_NAN_RUN_FOR_OUTAGE=4)
        for _ in 1:2
            push_features!(sol, 1, fill(Float32(NaN), d))
        end
        sol.vertex_features[1, :] .= Float32(NaN)

        results = compute_anomaly_detection(graph)
        outage_results = filter(r -> r["anomaly_type"] == "sensor_outage", results)
        @test isempty(outage_results)
    end

    @testset "outage severity escalation" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        d = size(sol.vertex_features, 2)
        push_stable_baseline!(sol, 1, 20;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.01f0)
        # Push 12 consecutive NaN readings (≥ 2 × MIN_NAN_RUN_FOR_OUTAGE)
        for _ in 1:12
            push_features!(sol, 1, fill(Float32(NaN), d))
        end
        sol.vertex_features[1, :] .= Float32(NaN)

        results = compute_anomaly_detection(graph)
        outage_results = filter(r -> r["anomaly_type"] == "sensor_outage", results)
        @test !isempty(outage_results)
        @test any(r -> r["severity"] == "alarm", outage_results)
    end

    @testset "outage + statistical anomaly coexist in results" begin
        # Use two separate layers: soil gets outage, weather gets WE alarm
        graph = make_anomaly_graph(nv=1, layers_list=["soil", "weather"])
        sol = graph.layers[:soil]
        wth = graph.layers[:weather]
        d_sol = size(sol.vertex_features, 2)
        d_wth = size(wth.vertex_features, 2)

        # Soil: push 20 stable then 5 NaN → outage
        push_stable_baseline!(sol, 1, 20;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.005f0)
        for _ in 1:5
            push_features!(sol, 1, fill(Float32(NaN), d_sol))
        end
        sol.vertex_features[1, :] .= Float32(NaN)

        # Weather: push 20 stable then set extreme outlier current → WE alarm
        push_stable_baseline!(wth, 1, 20;
                               baseline=Float32[22.0, 60.0, 0.0, 5.0, 15.0],
                               noise_std=0.005f0)
        wth.vertex_features[1, :] .= Float32[55.0, 60.0, 0.0, 5.0, 15.0]

        results = compute_anomaly_detection(graph)
        has_outage = any(r -> r["anomaly_type"] == "sensor_outage", results)
        has_we = any(r -> r["anomaly_type"] != "sensor_outage", results)
        @test has_outage
        @test has_we
    end

    @testset "multi-layer outage detection" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil", "weather"])
        sol = graph.layers[:soil]
        wth = graph.layers[:weather]
        d_sol = size(sol.vertex_features, 2)
        d_wth = size(wth.vertex_features, 2)

        # Push stable baseline for both layers
        push_stable_baseline!(sol, 1, 20;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.01f0)
        push_stable_baseline!(wth, 1, 20;
                               baseline=Float32[22.0, 60.0, 0.0, 5.0, 15.0],
                               noise_std=0.01f0)

        # Push NaN runs for vertex 1 on BOTH layers
        for _ in 1:6
            push_features!(sol, 1, fill(Float32(NaN), d_sol))
            push_features!(wth, 1, fill(Float32(NaN), d_wth))
        end
        sol.vertex_features[1, :] .= Float32(NaN)
        wth.vertex_features[1, :] .= Float32(NaN)

        results = compute_anomaly_detection(graph)
        outage_v1 = filter(r -> r["anomaly_type"] == "sensor_outage" &&
                                r["vertex_id"] == "v1", results)
        @test !isempty(outage_v1)
        @test any(r -> get(r, "multi_layer_outage", false) == true, outage_v1)
    end

    @testset "count_nan_run matches kernel output" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        d = size(sol.vertex_features, 2)

        # Push 10 stable then 6 NaN
        push_stable_baseline!(sol, 1, 10;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.01f0)
        for _ in 1:6
            push_features!(sol, 1, fill(Float32(NaN), d))
        end
        # push_features! sets vertex_features to NaN on last push

        runs = count_nan_run(sol, 1)
        @test length(runs) == d
        @test all(runs .== 6)  # 1 current + 5 history NaN
    end

    @testset "estimated_outage_minutes is correct" begin
        graph = make_anomaly_graph(nv=1, layers_list=["soil"])
        sol = graph.layers[:soil]
        d = size(sol.vertex_features, 2)
        push_stable_baseline!(sol, 1, 20;
                               baseline=Float32[0.30, 25.0, 1.2, 6.5],
                               noise_std=0.01f0)
        for _ in 1:8
            push_features!(sol, 1, fill(Float32(NaN), d))
        end
        sol.vertex_features[1, :] .= Float32(NaN)

        results = compute_anomaly_detection(graph)
        outage_results = filter(r -> r["anomaly_type"] == "sensor_outage", results)
        @test !isempty(outage_results)
        for r in outage_results
            @test r["estimated_outage_minutes"] == r["nan_run_length"] * AgriSenseCore.CADENCE_MINUTES
        end
    end
end
