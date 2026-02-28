# ---------------------------------------------------------------------------
# Irrigation scheduler — water balance model (GPU-first)
# ---------------------------------------------------------------------------

@kernel function water_balance_kernel!(result, moisture, et0, crop_kc, rainfall,
                                        soil_depth_mm)
    i = @index(Global)
    @inbounds begin
        result[i] = moisture[i] - (et0[i] * crop_kc[i]) / soil_depth_mm + rainfall[i] / soil_depth_mm
    end
end

@kernel function threshold_trigger_kernel!(recommendations, projected_moisture,
                                            wilting_point, field_capacity, valve_capacity)
    i = @index(Global)
    @inbounds begin
        if projected_moisture[i] < wilting_point[i]
            recommendations[i] = min(field_capacity[i] - projected_moisture[i],
                                     valve_capacity[i])
        else
            recommendations[i] = 0.0f0
        end
    end
end

@kernel function hargreaves_et0_kernel!(out, temp, solar_rad)
    i = @index(Global)
    @inbounds begin
        t_range = max(0.3f0 * abs(temp[i]), 2.0f0)
        out[i] = 0.0023f0 * (temp[i] + 17.8f0) * sqrt(t_range) * solar_rad[i]
    end
end

# ---------------------------------------------------------------------------
# Hargreaves ET₀ estimation (scalar CPU version for reference/testing)
# ---------------------------------------------------------------------------

"""
    hargreaves_et0(temp, solar_rad) -> Float32

Simplified Hargreaves equation (FAO Paper 56).
"""
function hargreaves_et0(temp::Float32, solar_rad::Float32)::Float32
    t_range = max(0.3f0 * abs(temp), 2.0f0)
    return 0.0023f0 * (temp + 17.8f0) * sqrt(t_range) * solar_rad
end

"""
    growth_progress_to_kc(gp::Float32) -> Float32

Map growth_progress ∈ [0,1] → crop coefficient Kc ∈ [0.3, 1.2] (FAO linear).
"""
function growth_progress_to_kc(gp::Float32)::Float32
    return 0.3f0 + 0.9f0 * clamp(gp, 0.0f0, 1.0f0)
end

# ---------------------------------------------------------------------------
# Default agronomic parameters
# ---------------------------------------------------------------------------

const DEFAULT_WILTING_POINT  = 0.15f0
const DEFAULT_FIELD_CAPACITY = 0.35f0
const DEFAULT_VALVE_CAPACITY = 0.10f0
const SOIL_DEPTH_MM          = 1000.0f0

"""
    compute_irrigation_schedule(graph, weather_forecast, horizon_days) -> Vector{Dict}

Compute per-zone irrigation recommendations. GPU-accelerated when graph is on GPU.
"""
function compute_irrigation_schedule(graph::LayeredHyperGraph,
                                      weather_forecast::Dict,
                                      horizon_days::Int)::Vector{Dict{String,Any}}
    has_soil = haskey(graph.layers, :soil)
    has_weather = haskey(graph.layers, :weather)
    has_crop = haskey(graph.layers, :crop_requirements)
    has_irrig = haskey(graph.layers, :irrigation)

    (!has_soil || !has_weather) && return Dict{String,Any}[]

    nv = graph.n_vertices
    soil_layer = graph.layers[:soil]
    weather_layer = graph.layers[:weather]
    backend = array_backend(soil_layer.vertex_features)

    # Extract current features — stays on device
    soil_moisture = soil_layer.vertex_features[:, 1]
    temp          = weather_layer.vertex_features[:, 1]
    solar_rad     = weather_layer.vertex_features[:, 5]

    # Crop coefficient from growth progress
    kc_vec = if has_crop
        gp = graph.layers[:crop_requirements].vertex_features[:, 2]
        @. Float32(0.3f0 + 0.9f0 * clamp(gp, 0.0f0, 1.0f0))
    else
        if backend isa CPU
            fill(1.0f0, nv)
        else
            CUDA.ones(Float32, nv)
        end
    end

    # Agronomic parameter vectors — on device
    wilting_point  = backend isa CPU ? fill(DEFAULT_WILTING_POINT, nv)  : CUDA.fill(DEFAULT_WILTING_POINT, nv)
    field_capacity = backend isa CPU ? fill(DEFAULT_FIELD_CAPACITY, nv) : CUDA.fill(DEFAULT_FIELD_CAPACITY, nv)
    valve_capacity = backend isa CPU ? fill(DEFAULT_VALVE_CAPACITY, nv) : CUDA.fill(DEFAULT_VALVE_CAPACITY, nv)

    # Forecast arrays
    precip_raw = weather_layer.vertex_features[:, 3]
    precip_forecast = if haskey(weather_forecast, "precip_forecast")
        Float32.(weather_forecast["precip_forecast"])
    else
        nothing
    end
    et0_forecast = if haskey(weather_forecast, "et0_forecast")
        Float32.(weather_forecast["et0_forecast"])
    else
        nothing
    end

    # Run water balance over horizon
    recommendations = Dict{String,Any}[]
    projected = copy(soil_moisture)  # stays on device

    for day in 1:horizon_days
        # Per-day ET₀ — compute on device
        day_et0 = if et0_forecast !== nothing && day <= length(et0_forecast)
            val = et0_forecast[day]
            backend isa CPU ? fill(val, nv) : CUDA.fill(val, nv)
        else
            out = similar(temp)
            launch_kernel!(hargreaves_et0_kernel!, backend, nv, out, temp, solar_rad)
            out
        end

        # Per-day precip — on device
        precip_val = if precip_forecast !== nothing && day <= length(precip_forecast)
            precip_forecast[day]
        else
            Float32(mean(ensure_cpu(precip_raw)))
        end
        day_precip = backend isa CPU ? fill(precip_val, nv) : CUDA.fill(precip_val, nv)

        # Water balance kernel
        result = similar(projected)
        launch_kernel!(water_balance_kernel!, backend, nv,
                       result, projected, day_et0, kc_vec, day_precip, SOIL_DEPTH_MM)
        projected .= max.(result, 0.0f0)

        # Threshold trigger kernel
        recs = similar(projected)
        launch_kernel!(threshold_trigger_kernel!, backend, nv,
                       recs, projected, wilting_point, field_capacity, valve_capacity)

        # Pull to CPU for Dict building
        proj_cpu = ensure_cpu(projected)
        recs_cpu = ensure_cpu(recs)

        if has_irrig
            irrig_layer = graph.layers[:irrigation]
            B_cpu = ensure_cpu(irrig_layer.incidence)
            ne = size(B_cpu, 2)
            for e in 1:ne
                members = findall(!iszero, @view B_cpu[:, e])
                isempty(members) && continue

                zone_vol = mean(recs_cpu[members])
                zone_moisture = mean(proj_cpu[members])
                do_irrigate = zone_vol > 0.0f0

                zone_id = e <= length(irrig_layer.edge_ids) ?
                          irrig_layer.edge_ids[e] : "zone_$e"

                deficit = max(DEFAULT_WILTING_POINT - zone_moisture, 0.0f0)
                priority = clamp(deficit / (DEFAULT_FIELD_CAPACITY - DEFAULT_WILTING_POINT),
                                 0.0f0, 1.0f0)

                reason = if !do_irrigate
                    "moisture_adequate"
                elseif zone_moisture < DEFAULT_WILTING_POINT
                    "below_wilting_point"
                else
                    "projected_deficit"
                end

                push!(recommendations, Dict{String,Any}(
                    "zone_id" => zone_id,
                    "day" => day,
                    "irrigate" => do_irrigate,
                    "volume_liters" => Float64(zone_vol * 1000.0f0),
                    "priority" => Float64(priority),
                    "projected_moisture" => Float64(zone_moisture),
                    "trigger_reason" => reason,
                ))
            end
        else
            for v in 1:nv
                vid = v <= length(soil_layer.vertex_ids) ?
                      soil_layer.vertex_ids[v] : "v_$v"
                push!(recommendations, Dict{String,Any}(
                    "zone_id" => vid,
                    "day" => day,
                    "irrigate" => recs_cpu[v] > 0.0f0,
                    "volume_liters" => Float64(recs_cpu[v] * 1000.0f0),
                    "priority" => Float64(clamp(recs_cpu[v] / DEFAULT_VALVE_CAPACITY, 0.0, 1.0)),
                    "projected_moisture" => Float64(proj_cpu[v]),
                    "trigger_reason" => recs_cpu[v] > 0.0f0 ? "below_wilting_point" : "moisture_adequate",
                ))
            end
        end

        # Apply irrigation to projected moisture for subsequent days (on device)
        projected .+= recs
    end

    return recommendations
end
