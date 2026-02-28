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
end
