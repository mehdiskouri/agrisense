using Test
using AgriSenseCore

function make_backtest_graph(; farm_id::String="backtest-farm", nv::Int=6)
    vertices = [Dict{String,Any}("id" => "v$i", "type" => "sensor") for i in 1:nv]
    edges = [
        Dict{String,Any}("id" => "e-soil-1", "layer" => "soil", "vertex_ids" => ["v$i" for i in 1:nv], "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-weather-1", "layer" => "weather", "vertex_ids" => ["v$i" for i in 1:nv], "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-light-1", "layer" => "lighting", "vertex_ids" => ["v$i" for i in 1:nv], "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-npk-1", "layer" => "npk", "vertex_ids" => ["v$i" for i in 1:nv], "metadata" => Dict{String,Any}()),
        Dict{String,Any}("id" => "e-crop-1", "layer" => "crop_requirements", "vertex_ids" => ["v$i" for i in 1:nv], "metadata" => Dict{String,Any}()),
    ]
    config = Dict{String,Any}(
        "farm_id" => farm_id,
        "farm_type" => "greenhouse",
        "active_layers" => ["soil", "weather", "lighting", "npk", "crop_requirements"],
        "zones" => [],
        "models" => Dict("irrigation" => false, "nutrients" => false, "yield_forecast" => true, "anomaly_detection" => false),
        "vertices" => vertices,
        "edges" => edges,
    )
    graph = to_cpu(build_hypergraph(FarmProfile(config), config["vertices"], config["edges"]))

    for v in 1:nv
        graph.layers[:soil].vertex_features[v, 1] = 0.30f0
        graph.layers[:soil].vertex_features[v, 2] = 22.0f0
        graph.layers[:soil].vertex_features[v, 3] = 1.2f0
        graph.layers[:soil].vertex_features[v, 4] = 6.5f0

        graph.layers[:weather].vertex_features[v, 1] = 23.0f0
        graph.layers[:weather].vertex_features[v, 2] = 55.0f0
        graph.layers[:weather].vertex_features[v, 3] = 0.0f0
        graph.layers[:weather].vertex_features[v, 4] = 3.0f0
        graph.layers[:weather].vertex_features[v, 5] = 2.0f0

        graph.layers[:lighting].vertex_features[v, 1] = 300.0f0
        graph.layers[:lighting].vertex_features[v, 2] = 20.0f0
        graph.layers[:lighting].vertex_features[v, 3] = 0.5f0

        graph.layers[:npk].vertex_features[v, 1] = 80.0f0
        graph.layers[:npk].vertex_features[v, 2] = 60.0f0
        graph.layers[:npk].vertex_features[v, 3] = 70.0f0

        graph.layers[:crop_requirements].vertex_features[v, 1] = 5.0f0
        graph.layers[:crop_requirements].vertex_features[v, 2] = 0.6f0
        graph.layers[:crop_requirements].vertex_features[v, 3] = 80.0f0
        graph.layers[:crop_requirements].vertex_features[v, 4] = 60.0f0
        graph.layers[:crop_requirements].vertex_features[v, 5] = 70.0f0

        for _ in 1:32
            push_features!(graph.layers[:soil], v, Float32[0.30, 22.0, 1.2, 6.5])
            push_features!(graph.layers[:lighting], v, Float32[300.0, 20.0, 0.5])
        end
    end

    return graph
end

@testset "Yield Backtesting" begin
    @testset "backtest returns fold metrics and weights" begin
        clear_ensemble_weights!()
        graph = make_backtest_graph(farm_id="bt-1")
        payload = backtest_yield_ensemble(graph; n_folds=4, min_history=12)

        @test payload["status"] == "ok"
        @test payload["n_folds"] == 4
        @test length(payload["per_fold_metrics"]) == 4
        @test haskey(payload["aggregate_metrics"], "ensemble")
        @test haskey(payload["aggregate_metrics"], "validation")
        @test haskey(payload["aggregate_metrics"], "test")
        @test haskey(payload, "hyperparameters")
        @test haskey(payload["hyperparameters"], "exp_alpha")
        @test haskey(payload, "temporal_split")
        @test payload["temporal_split"]["train"]["end"] < payload["temporal_split"]["validation"]["end"]

        weights = payload["weights"]
        @test haskey(weights, "fao_single")
        @test haskey(weights, "exp_smoothing")
        @test haskey(weights, "quantile_regression")
        sumw = Float32(weights["fao_single"] + weights["exp_smoothing"] + weights["quantile_regression"])
        @test sumw ≈ 1.0f0 atol=1e-4
    end

    @testset "insufficient history degrades gracefully" begin
        graph = make_backtest_graph(farm_id="bt-short")
        payload = backtest_yield_ensemble(graph; n_folds=4, min_history=90)
        @test payload["status"] == "insufficient_history"
        @test payload["n_folds"] == 0
    end

    @testset "backtest updates ensemble cache" begin
        clear_ensemble_weights!()
        graph = make_backtest_graph(farm_id="bt-2")
        _ = backtest_yield_ensemble(graph; n_folds=3)

        cached = get_ensemble_weights("bt-2")
        @test length(cached) == 3
        @test sum(cached) ≈ 1.0f0 atol=1e-4
    end

    @testset "hyperparameter optimization persists per farm" begin
        clear_ensemble_hyperparams!()
        graph = make_backtest_graph(farm_id="bt-3")
        params = optimize_ensemble_hyperparams!(graph; n_folds=3, min_history=12)
        @test haskey(params, "exp_alpha")
        @test haskey(params, "exp_beta")
        @test haskey(params, "quantile_lambda")

        cached = get_ensemble_hyperparams("bt-3")
        @test cached["exp_alpha"] == params["exp_alpha"]
        @test cached["quantile_lambda"] == params["quantile_lambda"]
    end
end
