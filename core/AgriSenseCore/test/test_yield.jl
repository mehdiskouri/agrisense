using Test
using AgriSenseCore
using SparseArrays
using Statistics

# Helper: build a graph with all layers needed for yield forecasting
function make_yield_graph(;
    nv::Int=4,
    soil_moisture::Vector{Float32}=Float32[0.30, 0.30, 0.30, 0.30],
    temperature::Vector{Float32}=Float32[22.0, 22.0, 22.0, 22.0],
    dli::Vector{Float32}=Float32[20.0, 20.0, 20.0, 20.0],
    target_yield::Vector{Float32}=Float32[5.0, 5.0, 5.0, 5.0],
    growth_progress::Vector{Float32}=Float32[0.5, 0.5, 0.5, 0.5],
    npk_current::Matrix{Float32}=Float32[80 60 70; 80 60 70; 80 60 70; 80 60 70],
    npk_required::Matrix{Float32}=Float32[80 60 70; 80 60 70; 80 60 70; 80 60 70],
)
    vertices = [Dict{String,Any}("id" => "v$i", "type" => "sensor") for i in 1:nv]
    edges = [
        Dict{String,Any}("id" => "e-soil-1", "layer" => "soil",
             "vertex_ids" => ["v$i" for i in 1:nv],
             "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-weather-1", "layer" => "weather",
             "vertex_ids" => ["v$i" for i in 1:nv],
             "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-lighting-1", "layer" => "lighting",
             "vertex_ids" => ["v$i" for i in 1:nv],
             "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-npk-1", "layer" => "npk",
             "vertex_ids" => ["v$i" for i in 1:nv],
             "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-crop-1", "layer" => "crop_requirements",
             "vertex_ids" => ["v$i" for i in 1:nv],
             "metadata" => Dict{String,Any}()),
    ]

    config = Dict{String,Any}(
        "farm_id" => "yield-test",
        "farm_type" => "greenhouse",
        "active_layers" => ["soil", "weather", "lighting", "npk", "crop_requirements"],
        "zones" => [],
        "models" => Dict("irrigation" => false, "nutrients" => false,
                         "yield_forecast" => true, "anomaly_detection" => false),
        "vertices" => vertices,
        "edges" => edges,
    )
    profile = FarmProfile(config)
    graph = build_hypergraph(profile, config["vertices"], config["edges"])
    graph = to_cpu(graph)  # tests validate correctness on CPU; GPU tested in test_gpu.jl

    # Populate features
    for v in 1:nv
        # Soil: moisture, temp, conductivity, pH
        graph.layers[:soil].vertex_features[v, 1] = soil_moisture[v]

        # Weather: temp, humidity, precip, wind_speed, solar_rad
        graph.layers[:weather].vertex_features[v, 1] = temperature[v]

        # Lighting: par, dli, spectrum_index
        graph.layers[:lighting].vertex_features[v, 2] = dli[v]

        # NPK current
        graph.layers[:npk].vertex_features[v, 1:3] .= npk_current[v, :]

        # Crop requirements: target_yield, growth_progress, n_target, p_target, k_target
        graph.layers[:crop_requirements].vertex_features[v, 1] = target_yield[v]
        graph.layers[:crop_requirements].vertex_features[v, 2] = growth_progress[v]
        graph.layers[:crop_requirements].vertex_features[v, 3:5] .= npk_required[v, :]
    end

    return graph
end

# ---------------------------------------------------------------------------
@testset "Yield Forecast" begin

    @testset "perfect conditions yield equals target" begin
        graph = make_yield_graph(
            soil_moisture=Float32[0.30, 0.30, 0.30, 0.30],  # good (>wilting, <capacity)
            temperature=Float32[22.0, 22.0, 22.0, 22.0],     # optimal range 15-30
            dli=Float32[20.0, 20.0, 20.0, 20.0],             # optimal DLI
            target_yield=Float32[5.0, 5.0, 5.0, 5.0],
        )
        results = compute_yield_forecast(graph)
        @test length(results) >= 1

        for r in results
            # All stress factors should be ~1.0
            sf = r["stress_factors"]
            @test sf["Ks"] > 0.7
            @test sf["Kn"] > 0.9
            @test sf["Kl"] > 0.9
            @test sf["Kw"] > 0.9
            # Yield should be close to target
            @test r["yield_estimate_kg_m2"] ≈ 5.0 atol=1.5
        end
    end

    @testset "water stress reduces yield and Ks" begin
        # Low moisture → water stress
        graph = make_yield_graph(
            soil_moisture=Float32[0.10, 0.10, 0.10, 0.10],  # below wilting point
        )
        results = compute_yield_forecast(graph)
        @test length(results) >= 1

        for r in results
            @test r["stress_factors"]["Ks"] < 0.5  # significant water stress
            @test r["yield_estimate_kg_m2"] < 5.0    # yield reduced
        end
    end

    @testset "temperature stress reduces Kw" begin
        # Very cold temperature → weather stress
        graph = make_yield_graph(
            temperature=Float32[3.0, 3.0, 3.0, 3.0],  # well below optimal
        )
        results = compute_yield_forecast(graph)
        for r in results
            @test r["stress_factors"]["Kw"] < 0.5
        end
    end

    @testset "multiple stresses compound multiplicatively" begin
        # Perfect conditions
        graph_perfect = make_yield_graph()
        # Water + temperature stress
        graph_stressed = make_yield_graph(
            soil_moisture=Float32[0.10, 0.10, 0.10, 0.10],
            temperature=Float32[3.0, 3.0, 3.0, 3.0],
        )
        results_perfect = compute_yield_forecast(graph_perfect)
        results_stressed = compute_yield_forecast(graph_stressed)

        @test !isempty(results_perfect)
        @test !isempty(results_stressed)

        y_perfect = results_perfect[1]["yield_estimate_kg_m2"]
        y_stressed = results_stressed[1]["yield_estimate_kg_m2"]
        @test y_stressed < y_perfect
    end

    @testset "model_layer reports fao_only without trained residual" begin
        # Reset residual coefficients
        RESIDUAL_COEFFICIENTS[] = nothing

        graph = make_yield_graph()
        results = compute_yield_forecast(graph)
        for r in results
            @test r["model_layer"] == "fao_only"
        end
    end

    @testset "confidence intervals are symmetric around estimate" begin
        graph = make_yield_graph()
        results = compute_yield_forecast(graph)
        for r in results
            est = r["yield_estimate_kg_m2"]
            lo  = r["yield_lower"]
            hi  = r["yield_upper"]
            @test lo < est
            @test hi > est
            @test (est - lo) ≈ (hi - est) atol=0.01
        end
    end

    @testset "stress coefficients computation" begin
        graph = make_yield_graph()
        ks, kn, kl, kw = compute_stress_coefficients(graph)
        @test length(ks) == graph.n_vertices
        @test all(0 .<= ks .<= 1)
        @test all(0 .<= kn .<= 1)
        @test all(0 .<= kl .<= 1)
        @test all(0 .<= kw .<= 1)
    end

    @testset "output has all required keys" begin
        graph = make_yield_graph()
        results = compute_yield_forecast(graph)
        required_keys = ["crop_bed_id", "yield_estimate_kg_m2", "yield_lower",
                         "yield_upper", "confidence", "stress_factors", "model_layer"]
        for r in results
            for k in required_keys
                @test haskey(r, k)
            end
        end
    end

    @testset "no crop_requirements layer returns empty" begin
        config = Dict{String,Any}(
            "farm_id" => "no-crop",
            "farm_type" => "greenhouse",
            "active_layers" => ["soil"],
            "zones" => [],
            "models" => Dict("irrigation" => false, "nutrients" => false,
                             "yield_forecast" => true, "anomaly_detection" => false),
            "vertices" => [Dict{String,Any}("id" => "v1", "type" => "sensor")],
            "edges" => [Dict{String,Any}("id" => "e1", "layer" => "soil",
                        "vertex_ids" => ["v1"],
                        "metadata" => Dict{String,Any}())],
        )
        profile = FarmProfile(config)
        graph = build_hypergraph(profile, config["vertices"], config["edges"])
        results = compute_yield_forecast(graph)
        @test isempty(results)
    end

    @testset "fit_residual_model basic sanity" begin
        X = Float32[1 2; 3 4; 5 6; 7 8]
        y = Float32[1.0, 2.0, 3.0, 4.0]
        β = fit_residual_model(X, y; λ=1.0f0)
        @test length(β) == 2
        # Predictions should be reasonable
        y_pred = X * β
        @test cor(y_pred, y) > 0.9
    end

    @testset "confidence interval contains FAO target under perfect conditions" begin
        RESIDUAL_COEFFICIENTS[] = nothing
        graph = make_yield_graph(
            soil_moisture=Float32[0.30, 0.30, 0.30, 0.30],
            temperature=Float32[22.0, 22.0, 22.0, 22.0],
            dli=Float32[20.0, 20.0, 20.0, 20.0],
            target_yield=Float32[5.0, 5.0, 5.0, 5.0],
        )
        results = compute_yield_forecast(graph)
        for r in results
            # The true FAO estimate should lie within [lower, upper]
            @test r["yield_lower"] <= r["yield_estimate_kg_m2"] <= r["yield_upper"]
            # CI should bracket the known target (5.0) under perfect conditions
            @test r["yield_lower"] <= 5.0 + 0.5  # allow slack
            @test r["yield_upper"] >= 5.0 - 0.5
        end
    end

    @testset "derived features matrix has expected columns" begin
        graph = make_yield_graph()
        # Push some history so cumulative DLI is nonzero
        ll = graph.layers[:lighting]
        for v in 1:graph.n_vertices
            for _ in 1:10
                push_features!(ll, v, Float32[100.0, 18.0, 0.5])
            end
        end
        derived = compute_derived_features(graph)
        # Should have at least 2 columns: cumulative DLI + soil health score
        @test size(derived, 1) == graph.n_vertices
        @test size(derived, 2) >= 2
        # Cumulative DLI should be positive
        @test all(derived[:, 1] .> 0)
        # Soil health score should be in [0, 1]
        @test all(0 .<= derived[:, 2] .<= 1)
    end

    @testset "trained residual uses residual-std confidence interval" begin
        # Build a graph with only crop_requirements so feature count is low (5)
        nv = 8
        vertices = [Dict{String,Any}("id" => "v$i", "type" => "crop_bed") for i in 1:nv]
        edges = [Dict{String,Any}(
            "id" => "e-crop-1", "layer" => "crop_requirements",
            "vertex_ids" => ["v$i" for i in 1:nv],
            "metadata" => Dict{String,Any}(),
        )]

        config = Dict{String,Any}(
            "farm_id" => "yield-train-ci",
            "farm_type" => "greenhouse",
            "active_layers" => ["crop_requirements"],
            "zones" => [],
            "models" => Dict("irrigation" => false, "nutrients" => false,
                             "yield_forecast" => true, "anomaly_detection" => false),
            "vertices" => vertices,
            "edges" => edges,
        )
        profile = FarmProfile(config)
        graph = to_cpu(build_hypergraph(profile, config["vertices"], config["edges"]))

        for v in 1:nv
            graph.layers[:crop_requirements].vertex_features[v, 1] = 5.0f0   # target yield
            graph.layers[:crop_requirements].vertex_features[v, 2] = 0.5f0   # growth progress
            graph.layers[:crop_requirements].vertex_features[v, 3] = 80.0f0  # N target
            graph.layers[:crop_requirements].vertex_features[v, 4] = 60.0f0  # P target
            graph.layers[:crop_requirements].vertex_features[v, 5] = 70.0f0  # K target
        end

        # Deterministic observed outcomes with non-zero residual variance
        actual = Dict{String,Float32}(
            "v1" => 5.1f0, "v2" => 4.9f0, "v3" => 5.3f0, "v4" => 4.7f0,
            "v5" => 5.2f0, "v6" => 4.8f0, "v7" => 5.4f0, "v8" => 4.6f0,
        )

        train_yield_residual!(graph, actual)
        @test RESIDUAL_COEFFICIENTS[] !== nothing
        @test RESIDUAL_STD[] !== nothing

        results = compute_yield_forecast(graph)
        @test !isempty(results)
        for r in results
            @test r["model_layer"] == "fao_plus_residual"
            @test r["confidence"] ≈ 0.95 atol=1e-6
            half = (r["yield_upper"] - r["yield_lower"]) / 2.0
            @test half ≈ Float64(1.96f0 * RESIDUAL_STD[]) atol=1e-3
        end
    end
end

# ===========================================================================
# Phase 13 — NaN Yield Guard Tests
# ===========================================================================
@testset "Phase 13 — NaN Yield Guards" begin

    @testset "NaN moisture → Ks uses safe default (0.25)" begin
        graph = make_yield_graph(
            soil_moisture=Float32[NaN, NaN, NaN, NaN],
        )
        ks, _, _, _ = compute_stress_coefficients(graph)
        @test length(ks) == 4
        # NaN moisture → replaced with 0.25 → Ks should be computed from 0.25
        @test all(0.0f0 .<= ks .<= 1.0f0)  # must be valid, not NaN
        @test !any(isnan.(ks))
    end

    @testset "NaN temperature → Kw defaults to 1.0 (no stress)" begin
        graph = make_yield_graph(
            temperature=Float32[NaN, NaN, NaN, NaN],
        )
        _, _, _, kw = compute_stress_coefficients(graph)
        @test all(kw .≈ 1.0f0)  # NaN temp → 1.0 (no weather stress assumed)
    end

    @testset "NaN NPK → Kn uses worst-case deficit" begin
        nan_npk = Float32[NaN NaN NaN; NaN NaN NaN; NaN NaN NaN; NaN NaN NaN]
        graph = make_yield_graph(npk_current=nan_npk)
        _, kn, _, _ = compute_stress_coefficients(graph)
        @test all(0.0f0 .<= kn .<= 1.0f0)
        @test !any(isnan.(kn))
        # NaN → 0 nutrient → max deficit → Kn should be low
        @test all(kn .< 0.5f0)
    end

    @testset "NaN DLI → Kl defaults to 1.0 (no light stress)" begin
        graph = make_yield_graph(dli=Float32[NaN, NaN, NaN, NaN])
        _, _, kl, _ = compute_stress_coefficients(graph)
        @test all(kl .≈ 1.0f0)  # NaN DLI → optimal DLI → Kl=1.0
    end

    @testset "NaN DLI in history → cumulative DLI treats NaN as 0" begin
        graph = make_yield_graph()
        ll = graph.layers[:lighting]
        for v in 1:graph.n_vertices
            for _ in 1:5
                push_features!(ll, v, Float32[100.0, 18.0, 0.5])  # valid DLI
            end
            for _ in 1:5
                push_features!(ll, v, Float32[100.0, NaN, 0.5])   # NaN DLI
            end
        end
        derived = compute_derived_features(graph)
        @test size(derived, 1) == graph.n_vertices
        @test size(derived, 2) >= 1
        # Cumulative DLI should be 5 × 18.0 = 90.0 (NaN slots → 0)
        @test all(derived[:, 1] .≈ 90.0f0)
    end

    @testset "fit_residual_model filters NaN rows" begin
        # X with some NaN rows
        X = Float32[1 2; NaN 4; 5 6; 7 NaN; 9 10]
        y = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
        β = fit_residual_model(X, y; λ=1.0f0)
        @test length(β) == 2
        @test !any(isnan.(β))  # NaN rows filtered → β should be finite
    end

    @testset "fit_residual_model with NaN y values" begin
        X = Float32[1 2; 3 4; 5 6; 7 8]
        y = Float32[1.0, NaN, 3.0, 4.0]
        β = fit_residual_model(X, y; λ=1.0f0)
        @test length(β) == 2
        @test !any(isnan.(β))
    end

    @testset "fit_residual_model all NaN → zero coefficients" begin
        X = Float32[NaN NaN; NaN NaN; NaN NaN]
        y = Float32[NaN, NaN, NaN]
        β = fit_residual_model(X, y; λ=1.0f0)
        @test length(β) == 2
        @test all(β .== 0.0f0)
    end

    @testset "full yield forecast with NaN inputs does not crash" begin
        RESIDUAL_COEFFICIENTS[] = nothing
        graph = make_yield_graph(
            soil_moisture=Float32[NaN, 0.30, NaN, 0.30],
            temperature=Float32[NaN, 22.0, NaN, 22.0],
            dli=Float32[NaN, 20.0, NaN, 20.0],
        )
        results = compute_yield_forecast(graph)
        @test !isempty(results)
        for r in results
            @test r["yield_estimate_kg_m2"] >= 0.0
            @test !isnan(r["yield_estimate_kg_m2"])
        end
    end
end
