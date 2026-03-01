using Test
using AgriSenseCore
using SparseArrays
using Statistics

# ---------------------------------------------------------------------------
# GPU-specific tests — validates data residency, kernel dispatch, cache,
# and model parity between GPU and CPU paths.
# Conditional: skipped entirely when CUDA is not functional.
# ---------------------------------------------------------------------------

if !AgriSenseCore.HAS_CUDA
    @testset "GPU Tests (SKIPPED — no CUDA)" begin
        @test_skip "CUDA not available"
    end
else

using CUDA
using CUDA.CUSPARSE
using KernelAbstractions
CUDA.allowscalar(true)

# ============================================================================
# Helper: build a multi-layer graph with realistic features
# ============================================================================
function make_gpu_test_graph(; nv::Int=6)
    vertices = [Dict{String,Any}("id" => "v$i", "type" => "sensor") for i in 1:nv]
    edges = [
        Dict{String,Any}("id"=>"e-soil-1",  "layer"=>"soil",
             "vertex_ids"=>["v$i" for i in 1:3], "metadata"=>Dict{String,Any}()),
        Dict{String,Any}("id"=>"e-soil-2",  "layer"=>"soil",
             "vertex_ids"=>["v$i" for i in 4:nv], "metadata"=>Dict{String,Any}()),
        Dict{String,Any}("id"=>"e-irr-1",   "layer"=>"irrigation",
             "vertex_ids"=>["v$i" for i in 1:3], "metadata"=>Dict{String,Any}()),
        Dict{String,Any}("id"=>"e-irr-2",   "layer"=>"irrigation",
             "vertex_ids"=>["v$i" for i in 4:nv], "metadata"=>Dict{String,Any}()),
        Dict{String,Any}("id"=>"e-weather-1","layer"=>"weather",
             "vertex_ids"=>["v$i" for i in 1:nv], "metadata"=>Dict{String,Any}()),
        Dict{String,Any}("id"=>"e-npk-1",   "layer"=>"npk",
             "vertex_ids"=>["v$i" for i in 1:3], "metadata"=>Dict{String,Any}()),
        Dict{String,Any}("id"=>"e-npk-2",   "layer"=>"npk",
             "vertex_ids"=>["v$i" for i in 4:nv], "metadata"=>Dict{String,Any}()),
        Dict{String,Any}("id"=>"e-crop-1",  "layer"=>"crop_requirements",
             "vertex_ids"=>["v$i" for i in 1:nv], "metadata"=>Dict{String,Any}()),
        Dict{String,Any}("id"=>"e-vis-1",   "layer"=>"vision",
             "vertex_ids"=>["v$i" for i in 1:nv], "metadata"=>Dict{String,Any}()),
        Dict{String,Any}("id"=>"e-light-1", "layer"=>"lighting",
             "vertex_ids"=>["v$i" for i in 1:nv], "metadata"=>Dict{String,Any}()),
    ]

    config = Dict{String,Any}(
        "farm_id" => "gpu-test-farm",
        "farm_type" => "greenhouse",
        "active_layers" => ["soil","irrigation","weather","npk",
                            "crop_requirements","vision","lighting"],
        "zones" => [Dict{String,Any}("id"=>"z1","name"=>"GPU Zone",
                    "zone_type"=>"greenhouse","area_m2"=>200.0,"soil_type"=>"loam")],
        "models" => Dict{String,Any}("irrigation"=>true,"nutrients"=>true,
                         "yield_forecast"=>true,"anomaly_detection"=>true),
        "vertices" => vertices,
        "edges" => edges,
    )
    profile = FarmProfile(config)
    graph = build_hypergraph(profile, config["vertices"], config["edges"])

    # Populate with realistic feature values
    for v in 1:nv
        graph.layers[:soil].vertex_features[v, 1] = Float32(0.10 + 0.05*v)  # moisture
        graph.layers[:soil].vertex_features[v, 2] = Float32(20.0 + v)        # temperature
        graph.layers[:weather].vertex_features[v, 1] = Float32(22.0 + v)     # temp
        graph.layers[:weather].vertex_features[v, 2] = Float32(60.0)         # humidity
        graph.layers[:weather].vertex_features[v, 3] = Float32(0.0)          # precip
        graph.layers[:weather].vertex_features[v, 5] = Float32(15.0)         # solar_rad
        graph.layers[:npk].vertex_features[v, 1] = Float32(50.0 + 5*v)       # N
        graph.layers[:npk].vertex_features[v, 2] = Float32(30.0 + 3*v)       # P
        graph.layers[:npk].vertex_features[v, 3] = Float32(40.0 + 4*v)       # K
        graph.layers[:crop_requirements].vertex_features[v, 1] = Float32(5.0)  # target yield
        graph.layers[:crop_requirements].vertex_features[v, 2] = Float32(0.5)  # growth progress
        graph.layers[:crop_requirements].vertex_features[v, 3] = Float32(80.0) # N target
        graph.layers[:crop_requirements].vertex_features[v, 4] = Float32(60.0) # P target
        graph.layers[:crop_requirements].vertex_features[v, 5] = Float32(70.0) # K target
        graph.layers[:lighting].vertex_features[v, 2] = Float32(18.0)         # DLI
        graph.layers[:vision].vertex_features[v, 1] = Float32(0.8)            # canopy
        graph.layers[:vision].vertex_features[v, 3] = Float32(0.1)            # anomaly_score
    end
    return graph
end

# ============================================================================
@testset "GPU Data Residency" begin

    @testset "build_hypergraph creates GPU arrays" begin
        graph = make_gpu_test_graph()
        for (name, layer) in graph.layers
            @test layer.incidence isa CuSparseMatrixCSR{Float32, Int32}
            @test layer.vertex_features isa CuMatrix{Float32}
            @test layer.feature_history isa CuArray{Float32, 3}
        end
    end

    @testset "to_cpu brings all arrays back" begin
        graph = make_gpu_test_graph()
        cpu_graph = to_cpu(graph)
        for (name, layer) in cpu_graph.layers
            @test layer.incidence isa SparseMatrixCSC{Float32, Int32}
            @test layer.vertex_features isa Matrix{Float32}
            @test layer.feature_history isa Array{Float32, 3}
        end
    end

    @testset "to_gpu → to_cpu round-trip preserves values" begin
        graph = make_gpu_test_graph()
        cpu1 = to_cpu(graph)
        gpu2 = to_gpu(cpu1)
        cpu2 = to_cpu(gpu2)
        for (name, layer) in cpu1.layers
            @test ensure_cpu(layer.vertex_features) ≈ ensure_cpu(cpu2.layers[name].vertex_features)
            B1 = ensure_cpu(layer.incidence)
            B2 = ensure_cpu(cpu2.layers[name].incidence)
            @test B1 ≈ B2
        end
    end

    @testset "ensure_cpu returns plain Array from CuArray" begin
        gpu_arr = CUDA.ones(Float32, 5)
        cpu_arr = ensure_cpu(gpu_arr)
        @test cpu_arr isa Vector{Float32}
        @test all(cpu_arr .== 1.0f0)
    end

    @testset "ensure_cpu is no-op for Array" begin
        arr = ones(Float32, 5)
        @test ensure_cpu(arr) === arr
    end
end

# ============================================================================
@testset "GPU Graph Cache" begin

    @testset "cache lifecycle" begin
        clear_cache!()
        @test isempty(GRAPH_CACHE)

        graph = make_gpu_test_graph()
        cache_graph!("test-1", graph)
        @test get_cached_graph("test-1") !== nothing
        @test get_cached_graph("nonexistent") === nothing

        evict_graph!("test-1")
        @test get_cached_graph("test-1") === nothing

        cache_graph!("test-2", graph)
        cache_graph!("test-3", graph)
        clear_cache!()
        @test isempty(GRAPH_CACHE)
    end

    @testset "bridge build_graph caches on GPU" begin
        clear_cache!()
        config = Dict{String,Any}(
            "farm_id" => "cached-farm",
            "farm_type" => "greenhouse",
            "active_layers" => ["soil"],
            "zones" => [Dict{String,Any}("id"=>"z1","name"=>"Z",
                        "zone_type"=>"greenhouse","area_m2"=>50.0,"soil_type"=>"loam")],
            "models" => Dict{String,Any}("irrigation"=>true,"nutrients"=>false,
                             "yield_forecast"=>false,"anomaly_detection"=>false),
            "vertices" => [Dict{String,Any}("id"=>"v1"),Dict{String,Any}("id"=>"v2")],
            "edges" => [Dict{String,Any}("id"=>"e1","layer"=>"soil",
                        "vertex_ids"=>["v1","v2"],"metadata"=>Dict{String,Any}())],
        )
        state = build_graph(config)
        cached = get_cached_graph("cached-farm")
        @test cached !== nothing
        @test cached.layers[:soil].vertex_features isa CuMatrix{Float32}
        clear_cache!()
    end
end

# ============================================================================
@testset "GPU push_features! & get_history" begin

    @testset "push_features! writes to GPU arrays" begin
        graph = make_gpu_test_graph()
        layer = graph.layers[:soil]
        nv = graph.n_vertices
        d = size(layer.vertex_features, 2)

        # Push new features for vertex 1
        new_feat = Float32[0.99, 99.0, 99.0, 99.0]
        push_features!(layer, 1, new_feat)

        # Verify current snapshot updated (on GPU)
        cpu_vf = ensure_cpu(layer.vertex_features)
        @test cpu_vf[1, 1] ≈ 0.99f0
        @test cpu_vf[1, 2] ≈ 99.0f0

        # History should have one entry
        @test layer.history_length == 1
        @test layer.history_head == 2

        hist = get_history(layer, 1)
        @test size(hist, 1) == d
        @test size(hist, 2) == 1
        @test hist[1, 1] ≈ 0.99f0
    end

    @testset "multiple pushes advance ring buffer" begin
        graph = make_gpu_test_graph()
        layer = graph.layers[:soil]

        for t in 1:5
            push_features!(layer, 1, Float32[t*0.1f0, 0.0f0, 0.0f0, 0.0f0])
        end
        @test layer.history_length == 5
        hist = get_history(layer, 1)
        @test size(hist, 2) == 5
        # Oldest first
        @test hist[1, 1] ≈ 0.1f0
        @test hist[1, 5] ≈ 0.5f0
    end
end

# ============================================================================
@testset "GPU Topology Mutations" begin

    @testset "add_hyperedge! CPU round-trip preserves GPU residency" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        old_ne = size(graph.layers[:soil].incidence, 2)

        add_hyperedge!(graph, :soil, "new-edge", ["v1", "v2", "v3"])

        layer = graph.layers[:soil]
        @test size(layer.incidence, 2) == old_ne + 1
        @test layer.incidence isa CuSparseMatrixCSR{Float32, Int32}
        @test layer.vertex_features isa CuMatrix{Float32}
    end

    @testset "remove_hyperedge! preserves GPU residency" begin
        graph = make_gpu_test_graph()
        ne_before = size(graph.layers[:soil].incidence, 2)
        remove_hyperedge!(graph, :soil, "e-soil-1")
        layer = graph.layers[:soil]
        @test size(layer.incidence, 2) == ne_before - 1
        @test layer.incidence isa CuSparseMatrixCSR{Float32, Int32}
    end

    @testset "add_vertex! extends GPU arrays" begin
        graph = make_gpu_test_graph()
        nv_before = graph.n_vertices
        new_idx = add_vertex!(graph, "v_new")
        @test new_idx == nv_before + 1
        @test graph.n_vertices == nv_before + 1
        for (_, layer) in graph.layers
            @test size(layer.vertex_features, 1) == nv_before + 1
            @test layer.vertex_features isa CuMatrix{Float32}
            @test layer.incidence isa CuSparseMatrixCSR{Float32, Int32}
        end
    end
end

# ============================================================================
@testset "GPU SpMM Aggregation" begin

    @testset "aggregate_by_edge mean" begin
        graph = make_gpu_test_graph()
        layer = graph.layers[:soil]
        result = aggregate_by_edge(layer; reduce=mean)
        @test result isa Matrix{Float32}
        ne = size(ensure_cpu(layer.incidence), 2)
        d = size(ensure_cpu(layer.vertex_features), 2)
        @test size(result) == (ne, d)
    end

    @testset "aggregate_by_edge sum" begin
        graph = make_gpu_test_graph()
        layer = graph.layers[:npk]
        result = aggregate_by_edge(layer; reduce=sum)
        @test result isa Matrix{Float32}
        @test size(result, 1) == size(ensure_cpu(layer.incidence), 2)
    end

    @testset "multi_layer_features concatenates on device" begin
        graph = make_gpu_test_graph()
        X = multi_layer_features(graph, [:soil, :npk, :weather])
        @test X isa CuMatrix{Float32}
        expected_cols = AgriSenseCore.LAYER_FEATURE_DIMS[:soil] +
                        AgriSenseCore.LAYER_FEATURE_DIMS[:npk] +
                        AgriSenseCore.LAYER_FEATURE_DIMS[:weather]
        @test size(X) == (graph.n_vertices, expected_cols)
        # Values should be accessible after ensure_cpu
        X_cpu = ensure_cpu(X)
        @test X_cpu[1, 1] ≈ ensure_cpu(graph.layers[:soil].vertex_features)[1, 1]
    end
end

# ============================================================================
@testset "GPU Model Parity — Irrigation" begin

    @testset "GPU irrigation matches expected behaviour" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        # Set moisture very low → should trigger irrigation
        for v in 1:nv
            graph.layers[:soil].vertex_features[v, 1] = Float32(0.08)
        end
        results = compute_irrigation_schedule(graph, Dict{String,Any}(), 1)
        @test !isempty(results)
        @test any(r["irrigate"] for r in results)
    end

    @testset "GPU irrigation high moisture suppresses" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        for v in 1:nv
            graph.layers[:soil].vertex_features[v, 1] = Float32(0.35)
        end
        results = compute_irrigation_schedule(graph, Dict{String,Any}(), 1)
        irrigate_flags = [r["irrigate"] for r in results]
        @test count(.!irrigate_flags) >= length(irrigate_flags) ÷ 2
    end
end

# ============================================================================
@testset "GPU Model Parity — Nutrients" begin

    @testset "GPU nutrient report detects deficits" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        # Low NPK → big deficit
        for v in 1:nv
            graph.layers[:npk].vertex_features[v, 1] = Float32(10.0)
            graph.layers[:npk].vertex_features[v, 2] = Float32(5.0)
            graph.layers[:npk].vertex_features[v, 3] = Float32(8.0)
        end
        results = compute_nutrient_report(graph)
        @test !isempty(results)
        for r in results
            @test haskey(r, "nitrogen_deficit")
            @test r["nitrogen_deficit"] > 0.0
            @test haskey(r, "urgency")
        end
    end

    @testset "GPU nutrient report — sufficient NPK → low severity" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        for v in 1:nv
            graph.layers[:npk].vertex_features[v, 1] = Float32(100.0)
            graph.layers[:npk].vertex_features[v, 2] = Float32(100.0)
            graph.layers[:npk].vertex_features[v, 3] = Float32(100.0)
        end
        results = compute_nutrient_report(graph)
        for r in results
            @test r["severity_score"] ≈ 0.0 atol=0.01
            @test r["urgency"] == "low"
        end
    end
end

# ============================================================================
@testset "GPU Model Parity — Yield" begin

    @testset "GPU yield forecast basic" begin
        graph = make_gpu_test_graph()
        results = compute_yield_forecast(graph)
        @test !isempty(results)
        for r in results
            @test haskey(r, "yield_estimate_kg_m2")
            @test r["yield_estimate_kg_m2"] >= 0.0
            @test haskey(r, "stress_factors")
        end
    end

    @testset "stress coefficients stay in [0, 1]" begin
        graph = make_gpu_test_graph()
        ks, kn, kl, kw = compute_stress_coefficients(graph)
        for arr in [ks, kn, kl, kw]
            cpu_arr = ensure_cpu(arr)
            @test all(0.0f0 .<= cpu_arr .<= 1.0f0)
        end
    end

    @testset "nutrient_stress_kernel! produces correct values" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        # Set NPK == required → Kn should be 1
        for v in 1:nv
            graph.layers[:npk].vertex_features[v, 1] = Float32(80.0)
            graph.layers[:npk].vertex_features[v, 2] = Float32(60.0)
            graph.layers[:npk].vertex_features[v, 3] = Float32(70.0)
        end
        _, kn, _, _ = compute_stress_coefficients(graph)
        kn_cpu = ensure_cpu(kn)
        @test all(kn_cpu .≈ 1.0f0)
    end

    @testset "weather_stress_kernel! piecewise ranges" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        # Set temps across ranges: 0, 10, 22, 35, 45
        temps = Float32[0.0, 10.0, 22.0, 35.0, 45.0, 22.0]
        for v in 1:min(nv, length(temps))
            graph.layers[:weather].vertex_features[v, 1] = temps[v]
        end
        _, _, _, kw = compute_stress_coefficients(graph)
        kw_cpu = ensure_cpu(kw)
        @test kw_cpu[1] ≈ 0.0f0           # T=0 < 5 → 0
        @test kw_cpu[2] ≈ 0.5f0           # T=10, (10-5)/10 = 0.5
        @test kw_cpu[3] ≈ 1.0f0           # T=22, in [15,30]
        @test kw_cpu[4] ≈ 0.5f0           # T=35, (40-35)/10 = 0.5
        @test kw_cpu[5] ≈ 0.0f0           # T=45 > 40 → 0
    end
end

# ============================================================================
@testset "GPU Model Parity — Anomaly" begin

    @testset "anomaly detection with GPU history" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        layer = graph.layers[:soil]

        # Push enough history for anomaly detection (>= 8)
        for t in 1:10
            for v in 1:nv
                feat = Float32[0.25 + 0.001*t, 22.0, 1.0, 6.5]
                push_features!(layer, v, feat)
            end
        end

        # Now inject a massive anomaly on v1
        push_features!(layer, 1, Float32[0.99, 22.0, 1.0, 6.5])

        results = compute_anomaly_detection(graph)
        # Should detect the moisture spike on v1
        soil_anomalies = filter(r -> r["layer"] == "soil" && r["vertex_id"] == "v1", results)
        @test !isempty(soil_anomalies)
        # New fields should be present
        for r in soil_anomalies
            @test haskey(r, "anomaly_rules")
            @test r["anomaly_rules"] isa Vector{String}
            @test haskey(r, "timestamp_start")
            @test haskey(r, "timestamp_end")
        end
    end

    @testset "no anomaly when insufficient history" begin
        graph = make_gpu_test_graph()
        # Fresh graph with no history pushes
        results = compute_anomaly_detection(graph)
        @test isempty(results)
    end
end

# ============================================================================
@testset "GPU Serialize / Deserialize round-trip" begin

    @testset "serialize_graph produces CPU arrays" begin
        graph = make_gpu_test_graph()
        state = serialize_graph(graph)
        @test state isa Dict{String,Any}
        for (name, ld) in state["layers"]
            @test ld["vertex_features"] isa Matrix{Float32}
            @test ld["feature_history"] isa Array{Float32, 3}
            @test ld["incidence_rows"] isa Vector{Int32}
        end
    end

    @testset "deserialize → GPU promotion via bridge" begin
        clear_cache!()
        graph = make_gpu_test_graph()
        state = serialize_graph(graph)

        # Deserialize produces CPU graph
        cpu_graph = deserialize_graph(state)
        for (_, layer) in cpu_graph.layers
            @test layer.vertex_features isa Matrix{Float32}
        end

        # _get_graph promotes to GPU and caches
        gpu_graph = AgriSenseCore._get_graph(state)
        for (_, layer) in gpu_graph.layers
            @test layer.vertex_features isa CuMatrix{Float32}
        end
        clear_cache!()
    end
end

# ============================================================================
@testset "GPU Backend Detection" begin

    @testset "array_backend dispatches correctly" begin
        gpu_arr = CUDA.ones(Float32, 5)
        cpu_arr = ones(Float32, 5)
        @test array_backend(gpu_arr) isa CUDABackend
        @test array_backend(cpu_arr) isa CPU
    end

    @testset "launch_kernel! runs on correct backend" begin
        # Test with a simple kernel from the codebase
        nv = 10
        out = CUDA.zeros(Float32, nv)
        temp = CUDA.fill(25.0f0, nv)
        solar = CUDA.fill(15.0f0, nv)
        launch_kernel!(hargreaves_et0_kernel!, CUDABackend(), nv, out, temp, solar)
        out_cpu = ensure_cpu(out)
        @test all(out_cpu .> 0.0f0)
    end
end

# ============================================================================
@testset "GPU Synthetic Parity" begin

    @testset "generate_soil_data GPU contract and shape" begin
        cpu = AgriSenseCore.generate_soil_data(8, 96 * 3; seed=77, use_gpu=false)
        gpu = AgriSenseCore.generate_soil_data(8, 96 * 3; seed=77, use_gpu=true)

        @test size(gpu["moisture"]) == size(cpu["moisture"])
        @test size(gpu["missing_mask"]) == size(cpu["missing_mask"])
        @test gpu["moisture"] isa Matrix{Float32}
        @test gpu["missing_mask"] isa BitMatrix
    end

    @testset "GPU weather statistically bounded vs CPU" begin
        cpu = AgriSenseCore.generate_weather_data(3, 96 * 7; seed=91, use_gpu=false)
        gpu = AgriSenseCore.generate_weather_data(3, 96 * 7; seed=91, use_gpu=true)

        cpu_temp = cpu["temperature"][.!isnan.(cpu["temperature"])]
        gpu_temp = gpu["temperature"][.!isnan.(gpu["temperature"])]

        @test !isempty(cpu_temp)
        @test !isempty(gpu_temp)

        μ_cpu = mean(cpu_temp)
        μ_gpu = mean(gpu_temp)
        σ_cpu = std(cpu_temp)
        σ_gpu = std(gpu_temp)

        @test abs(μ_cpu - μ_gpu) <= 2.0
        @test abs(σ_cpu - σ_gpu) <= 2.0
    end

    @testset "generate_vision_data GPU contract and shape" begin
        cpu = AgriSenseCore.generate_vision_data(2, 10, 96; seed=111, use_gpu=false)
        gpu = AgriSenseCore.generate_vision_data(2, 10, 96; seed=111, use_gpu=true)

        @test size(gpu["canopy_coverage_pct"]) == size(cpu["canopy_coverage_pct"])
        @test size(gpu["confidence"]) == size(cpu["confidence"])
        @test size(gpu["missing_mask"]) == size(cpu["missing_mask"])
        @test gpu["canopy_coverage_pct"] isa Matrix{Float32}
        @test gpu["confidence"] isa Matrix{Float32}
        @test gpu["missing_mask"] isa BitMatrix
    end

    @testset "GPU generate_synthetic returns CPU-safe arrays" begin
        result = AgriSenseCore.generate(:greenhouse, 14, 123)
        soil = result["layers"]["soil"]
        vision = result["layers"]["vision"]
        @test soil["moisture"] isa Matrix{Float32}
        @test soil["temperature"] isa Matrix{Float32}
        @test soil["missing_mask"] isa BitMatrix
        @test vision["canopy_coverage_pct"] isa Matrix{Float32}
        @test vision["confidence"] isa Matrix{Float32}
        @test vision["missing_mask"] isa BitMatrix
    end
end

# ============================================================================
@testset "GPU Outage Injection & Classification" begin

    @testset "outage_stamp_kernel! stamps correct positions on GPU" begin
        n_steps, n_channels = 100, 4
        mask_gpu = CuArray(fill(false, n_steps, n_channels))
        vals_gpu = CUDA.ones(Float32, n_steps, n_channels)

        # Known outage events: ch=1 start=10 dur=5, ch=3 start=50 dur=8
        channels = CuArray(Int32[1, 3])
        starts   = CuArray(Int32[10, 50])
        durations = CuArray(Int32[5, 8])

        launch_kernel!(outage_stamp_kernel!, CUDABackend(), 2,
                       mask_gpu, vals_gpu, channels, starts, durations, Int32(n_steps))

        mask_cpu = ensure_cpu(mask_gpu)
        vals_cpu = ensure_cpu(vals_gpu)

        # Verify ch=1, positions 10:14 stamped
        @test all(mask_cpu[10:14, 1])
        @test all(isnan.(vals_cpu[10:14, 1]))
        @test !mask_cpu[9, 1]
        @test !mask_cpu[15, 1]

        # Verify ch=3, positions 50:57 stamped
        @test all(mask_cpu[50:57, 3])
        @test all(isnan.(vals_cpu[50:57, 3]))
        @test !mask_cpu[49, 3]
        @test !mask_cpu[58, 3]

        # Unstamped channels untouched
        @test !any(mask_cpu[:, 2])
        @test !any(mask_cpu[:, 4])
    end

    @testset "nan_run_kernel! runs on GPU and matches CPU helper" begin
        graph = make_gpu_test_graph()
        layer = graph.layers[:soil]
        nv = graph.n_vertices
        d = size(ensure_cpu(layer.vertex_features), 2)

        # Push 10 stable readings for all vertices (10*6=60 < buf_size=96)
        for t in 1:10
            for v in 1:nv
                push_features!(layer, v, Float32[0.25, 22.0, 1.0, 6.5])
            end
        end

        # Push 6 NaN readings for vertex 1 only (total = 66 < 96)
        for _ in 1:6
            push_features!(layer, 1, fill(Float32(NaN), d))
        end

        # Verify current state
        @test layer.history_length == 66
        vf_cpu = ensure_cpu(layer.vertex_features)
        @test all(isnan.(vf_cpu[1, :]))  # v1 should be NaN from last push

        # Run CPU helper on a CPU copy
        cpu_graph = to_cpu(graph)
        cpu_layer = cpu_graph.layers[:soil]
        cpu_runs = count_nan_run(cpu_layer, 1)
        @test all(cpu_runs .== 6)  # 1 current + 5 history

        # Run GPU kernel directly
        backend = array_backend(layer.vertex_features)
        ndrange = nv * d
        buf_size = Int32(size(layer.feature_history, 3))
        head_val = Int32(mod1(layer.history_head - 1, buf_size))
        head_arr = CuArray(Int32[head_val])
        valid_len_arr = CuArray(Int32[layer.history_length])
        nan_runs_gpu = CUDA.zeros(Int32, nv, d)

        launch_kernel!(nan_run_kernel!, backend, ndrange,
                       nan_runs_gpu, layer.vertex_features, layer.feature_history,
                       head_arr, valid_len_arr, buf_size, Int32(d))

        nan_runs_cpu = ensure_cpu(nan_runs_gpu)
        @test all(nan_runs_cpu[1, :] .== 6)
    end

    @testset "GPU outage injection produces same events as CPU" begin
        cpu = AgriSenseCore.generate_soil_data(6, 96 * 5;
                    seed=77, use_gpu=false, outage_prob=0.05f0)
        gpu = AgriSenseCore.generate_soil_data(6, 96 * 5;
                    seed=77, use_gpu=true, outage_prob=0.05f0)

        # Events are sampled on CPU in both paths, so they must be identical
        @test cpu["outage_events"] == gpu["outage_events"]
        # Outage masks must match (stamp positions are deterministic)
        @test cpu["outage_mask"] == gpu["outage_mask"]
        # Shapes must match
        @test size(cpu["moisture"]) == size(gpu["moisture"])
    end
end

# ============================================================================
@testset "GPU Anomaly Detection After Buffer Wrap" begin

    @testset "GPU anomaly detection correct after buffer wrap" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        layer = graph.layers[:soil]
        d = size(ensure_cpu(layer.vertex_features), 2)
        buf_size = size(layer.feature_history, 3)  # DEFAULT_HISTORY_SIZE = 96

        # Push exactly buf_size readings to fill the buffer
        for t in 1:buf_size
            for v in 1:nv
                push_features!(layer, v, Float32[0.25 + 0.001*t, 22.0, 1.0, 6.5])
            end
        end
        @test layer.history_length == buf_size

        # Push 10 more to wrap the buffer
        for t in 1:10
            for v in 1:nv
                push_features!(layer, v, Float32[0.25 + 0.001*(buf_size + t), 22.0, 1.0, 6.5])
            end
        end
        @test layer.history_length == buf_size  # still capped
        @test layer.history_head != 1            # has wrapped

        # Inject a massive outlier on vertex 1
        push_features!(layer, 1, Float32[0.99, 22.0, 1.0, 6.5])

        results = compute_anomaly_detection(graph)
        soil_v1 = filter(r -> r["layer"] == "soil" && r["vertex_id"] == "v1", results)
        @test !isempty(soil_v1)
        # Should fire 3σ alarm on moisture
        moisture_v1 = filter(r -> r["feature"] == "moisture", soil_v1)
        @test !isempty(moisture_v1)
        @test any(r -> r["severity"] == "alarm", moisture_v1)
    end
end

# ============================================================================
# Phase 13 — GPU Mask Infrastructure Tests
# ============================================================================
@testset "GPU Phase 13 — Mask Transfer & Write" begin

    @testset "GPU mask arrays transfer correctly" begin
        graph = make_gpu_test_graph()
        for (_, layer) in graph.layers
            @test layer.feature_history_mask isa CuArray{Bool, 3}
            @test layer.vertex_features_mask isa CuArray{Bool, 2}
            @test size(layer.feature_history_mask) == size(layer.feature_history)
            @test size(layer.vertex_features_mask) == size(layer.vertex_features)
        end
    end

    @testset "GPU → CPU → GPU round-trip preserves masks" begin
        graph = make_gpu_test_graph()
        layer = graph.layers[:soil]
        push_features!(layer, 1, Float32[0.3, 25.0, 1.2, 6.5])
        push_features!(layer, 1, Float32[NaN, 24.0, NaN, 6.0])

        cpu_graph = to_cpu(graph)
        gpu_graph2 = to_gpu(cpu_graph)

        cpu_fhm = ensure_cpu(cpu_graph.layers[:soil].feature_history_mask)
        gpu_fhm = ensure_cpu(gpu_graph2.layers[:soil].feature_history_mask)
        @test cpu_fhm == gpu_fhm

        cpu_vfm = ensure_cpu(cpu_graph.layers[:soil].vertex_features_mask)
        gpu_vfm = ensure_cpu(gpu_graph2.layers[:soil].vertex_features_mask)
        @test cpu_vfm == gpu_vfm
    end

    @testset "GPU push_features! writes correct mask for valid/NaN" begin
        graph = make_gpu_test_graph()
        layer = graph.layers[:soil]

        # Push valid data
        push_features!(layer, 1, Float32[0.5, 22.0, 1.0, 6.5])
        vfm = ensure_cpu(layer.vertex_features_mask)
        @test all(vfm[1, :])  # all valid

        # Push NaN data
        push_features!(layer, 2, Float32[NaN, 22.0, NaN, 6.5])
        vfm2 = ensure_cpu(layer.vertex_features_mask)
        @test vfm2[2, 1] == false
        @test vfm2[2, 2] == true
        @test vfm2[2, 3] == false
        @test vfm2[2, 4] == true
    end

    @testset "GPU get_history return_mask works" begin
        graph = make_gpu_test_graph()
        layer = graph.layers[:soil]
        push_features!(layer, 1, Float32[1.0, 2.0, 3.0, 4.0])
        push_features!(layer, 1, Float32[NaN, 2.0, NaN, 4.0])

        data, mask = get_history(layer, 1; return_mask=true)
        @test data isa Matrix{Float32}
        @test mask isa Matrix{Bool}
        @test size(data) == size(mask)
        @test mask[1, 1] == true   # first push, feature 1 valid
        @test mask[1, 2] == false  # second push, feature 1 NaN
    end
end

@testset "GPU Phase 13 — NaN-Aware Model Parity" begin

    @testset "GPU anomaly detection ignores NaN in stats" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        layer = graph.layers[:soil]
        d = size(ensure_cpu(layer.vertex_features), 2)

        # Push 10 valid readings
        for t in 1:10
            for v in 1:nv
                push_features!(layer, v, Float32[0.25 + 0.001*t, 22.0, 1.0, 6.5])
            end
        end
        # Push 5 NaN readings
        for _ in 1:5
            for v in 1:nv
                push_features!(layer, v, fill(Float32(NaN), d))
            end
        end
        # Inject outlier on v1
        push_features!(layer, 1, Float32[0.99, 22.0, 1.0, 6.5])

        results = compute_anomaly_detection(graph)
        # Despite NaN history, outlier on v1 should still be detected
        soil_v1 = filter(r -> r["layer"] == "soil" && r["vertex_id"] == "v1" &&
                              r["anomaly_type"] != "sensor_outage", results)
        @test !isempty(soil_v1)
    end

    @testset "GPU weather stress NaN → Kw=1.0" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        for v in 1:nv
            graph.layers[:weather].vertex_features[v, 1] = Float32(NaN)
        end
        _, _, _, kw = compute_stress_coefficients(graph)
        kw_cpu = ensure_cpu(kw)
        @test all(kw_cpu .≈ 1.0f0)
    end

    @testset "GPU nutrient stress NaN → worst-case Kn" begin
        graph = make_gpu_test_graph()
        nv = graph.n_vertices
        for v in 1:nv
            graph.layers[:npk].vertex_features[v, 1] = Float32(NaN)
            graph.layers[:npk].vertex_features[v, 2] = Float32(NaN)
            graph.layers[:npk].vertex_features[v, 3] = Float32(NaN)
        end
        _, kn, _, _ = compute_stress_coefficients(graph)
        kn_cpu = ensure_cpu(kn)
        @test !any(isnan.(kn_cpu))
        @test all(kn_cpu .< 0.5f0)  # NaN → 0 → max deficit → low Kn
    end
end

# ============================================================================
@testset "Adapt.adapt_structure on GPU" begin
    using Adapt

    @testset "Adapt.adapt_structure preserves GPU-resident graph" begin
        # Build graph that's already GPU-resident
        graph = make_gpu_test_graph()

        # Adapted with identity (nothing) should preserve all GPU arrays
        adapted = Adapt.adapt_structure(nothing, graph)

        # CPU-only fields must remain plain types
        @test adapted.farm_id isa String
        @test adapted.farm_id == graph.farm_id
        @test adapted.vertex_index isa Dict{String,Int}
        @test adapted.n_vertices == graph.n_vertices

        # Each layer's GPU arrays should still be CuArrays
        for (name, lyr) in adapted.layers
            orig = graph.layers[name]
            @test lyr.vertex_features isa CuArray{Float32, 2}
            @test lyr.feature_history isa CuArray{Float32, 3}
            @test lyr.incidence isa CuSparseMatrixCSR{Float32, Int32}
            # CPU-passthrough fields inside layers
            @test lyr.vertex_ids isa Vector{String}
            @test lyr.edge_ids isa Vector{String}
            @test lyr.edge_metadata isa Vector{Dict{String,Any}}
            # Data preserved
            @test ensure_cpu(lyr.vertex_features) ≈ ensure_cpu(orig.vertex_features)
        end
    end

    @testset "Adapt.adapt_structure(nothing, gpu_graph) → to_cpu round-trip" begin
        graph = make_gpu_test_graph()

        # Set known values
        for v in 1:graph.n_vertices
            graph.layers[:soil].vertex_features[v, 1] = Float32(v) * 0.1f0
        end

        adapted = Adapt.adapt_structure(nothing, graph)
        cpu_back = to_cpu(adapted)

        @test cpu_back.farm_id == graph.farm_id
        @test cpu_back.n_vertices == graph.n_vertices
        for (name, lyr) in cpu_back.layers
            orig = graph.layers[name]
            @test ensure_cpu(lyr.vertex_features) ≈ ensure_cpu(orig.vertex_features)
            B1 = ensure_cpu(lyr.incidence)
            B2 = ensure_cpu(orig.incidence)
            @test B1 ≈ B2
            @test lyr.vertex_ids == orig.vertex_ids
            @test lyr.edge_ids == orig.edge_ids
        end
    end

    @testset "Config types default Adapt passthrough on GPU" begin
        # After removing @adapt_structure, these should be identity under any adaptor
        zc = ZoneConfig("z1", "Zone 1", :greenhouse, 100.0, "loam")
        mc = ModelConfig(true, true, true, true)
        fp = FarmProfile("farm-gpu", :greenhouse, Set([:soil, :irrigation]),
                          [zc], mc)

        # Default Adapt.adapt with nothing returns identical objects
        @test Adapt.adapt(nothing, zc) === zc
        @test Adapt.adapt(nothing, mc) === mc
        @test Adapt.adapt(nothing, fp) === fp
    end
end

end  # if HAS_CUDA
