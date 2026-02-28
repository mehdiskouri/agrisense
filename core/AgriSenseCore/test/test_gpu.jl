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
            @test layer.incidence isa CuSparseMatrixCSC{Float32, Int32}
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
        @test layer.incidence isa CuSparseMatrixCSC{Float32, Int32}
        @test layer.vertex_features isa CuMatrix{Float32}
    end

    @testset "remove_hyperedge! preserves GPU residency" begin
        graph = make_gpu_test_graph()
        ne_before = size(graph.layers[:soil].incidence, 2)
        remove_hyperedge!(graph, :soil, "e-soil-1")
        layer = graph.layers[:soil]
        @test size(layer.incidence, 2) == ne_before - 1
        @test layer.incidence isa CuSparseMatrixCSC{Float32, Int32}
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
            @test layer.incidence isa CuSparseMatrixCSC{Float32, Int32}
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

end  # if HAS_CUDA
