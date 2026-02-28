# ---------------------------------------------------------------------------
# Irrigation scheduler — water balance model (GPU-portable)
# ---------------------------------------------------------------------------

@kernel function water_balance_kernel!(result, moisture, et0, crop_kc, rainfall, irrigation)
    i = @index(Global)
    @inbounds begin
        result[i] = moisture[i] - et0[i] * crop_kc[i] + rainfall[i] + irrigation[i]
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

# ---------------------------------------------------------------------------
# Hargreaves ET₀ estimation
# ---------------------------------------------------------------------------

"""
    hargreaves_et0(temp, solar_rad) -> Float32

Simplified Hargreaves equation (FAO Paper 56):
  ET₀ = 0.0023 × (T_mean + 17.8) × √(T_range) × Ra

`temp` = mean air temperature (°C), `solar_rad` = extraterrestrial radiation proxy (MJ m⁻² d⁻¹).
We approximate T_range as 0.3 × |T_mean| (default when only mean temp is available).
"""
function hargreaves_et0(temp::Float32, solar_rad::Float32)::Float32
    t_range = max(0.3f0 * abs(temp), 2.0f0)  # guard against zero range
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

const DEFAULT_WILTING_POINT  = 0.15f0   # volumetric water content
const DEFAULT_FIELD_CAPACITY = 0.35f0
const DEFAULT_VALVE_CAPACITY = 0.10f0   # max irrigation per day (fraction of soil volume)
const SOIL_DEPTH_MM          = 1000.0f0 # effective root-zone depth in mm

"""
    compute_irrigation_schedule(graph::LayeredHyperGraph, weather_forecast::Dict,
                                 horizon_days::Int) -> Vector{Dict}

Compute per-zone irrigation recommendations for the next `horizon_days`.
Uses the water balance equation on the CPU/GPU backend.
"""
function compute_irrigation_schedule(graph::LayeredHyperGraph,
                                      weather_forecast::Dict,
                                      horizon_days::Int)::Vector{Dict{String,Any}}
    # Required layers
    has_soil = haskey(graph.layers, :soil)
    has_weather = haskey(graph.layers, :weather)
    has_crop = haskey(graph.layers, :crop_requirements)
    has_irrig = haskey(graph.layers, :irrigation)

    (!has_soil || !has_weather) && return Dict{String,Any}[]

    nv = graph.n_vertices
    soil_layer = graph.layers[:soil]
    weather_layer = graph.layers[:weather]

    # Extract current features
    soil_moisture = Float32.(soil_layer.vertex_features[:, 1])  # col 1 = moisture
    temp          = Float32.(weather_layer.vertex_features[:, 1])  # col 1 = temp
    precip        = Float32.(weather_layer.vertex_features[:, 3])  # col 3 = precip
    solar_rad     = Float32.(weather_layer.vertex_features[:, 5])  # col 5 = solar_rad

    # Crop coefficient from growth progress
    kc_vec = if has_crop
        gp = Float32.(graph.layers[:crop_requirements].vertex_features[:, 2])
        growth_progress_to_kc.(gp)
    else
        fill(1.0f0, nv)
    end

    # Agronomic parameters (uniform defaults per vertex)
    wilting_point  = fill(DEFAULT_WILTING_POINT, nv)
    field_capacity = fill(DEFAULT_FIELD_CAPACITY, nv)
    valve_capacity = fill(DEFAULT_VALVE_CAPACITY, nv)

    # Forecast arrays — default to current-day values repeated
    precip_forecast = if haskey(weather_forecast, "precip_forecast")
        fc = Float32.(weather_forecast["precip_forecast"])
        # Pad/truncate to horizon_days × nv
        [length(fc) >= d ? fc[d] : precip[1] for d in 1:horizon_days]
    else
        fill(mean(precip), horizon_days)
    end

    et0_forecast = if haskey(weather_forecast, "et0_forecast")
        Float32.(weather_forecast["et0_forecast"])
    else
        nothing
    end

    # Run water balance over horizon
    recommendations = Dict{String,Any}[]
    projected = copy(soil_moisture)

    for day in 1:horizon_days
        # Per-day ET₀ estimate
        day_et0 = if et0_forecast !== nothing && day <= length(et0_forecast)
            fill(Float32(et0_forecast[day]), nv)
        else
            [hargreaves_et0(temp[v], solar_rad[v]) for v in 1:nv]
        end

        day_precip = fill(Float32(get(precip_forecast, day, 0.0f0)), nv)

        # Water balance step (no irrigation applied yet)
        # ET₀ and precip are in mm/day; convert to volumetric fraction change
        result = similar(projected)
        for i in 1:nv
            result[i] = projected[i] - (day_et0[i] * kc_vec[i]) / SOIL_DEPTH_MM + day_precip[i] / SOIL_DEPTH_MM
        end
        projected .= max.(result, 0.0f0)

        # Trigger check
        recs = similar(projected)
        for i in 1:nv
            if projected[i] < wilting_point[i]
                recs[i] = min(field_capacity[i] - projected[i], valve_capacity[i])
            else
                recs[i] = 0.0f0
            end
        end

        # Build per-zone recommendations using irrigation layer edges
        if has_irrig
            irrig_layer = graph.layers[:irrigation]
            ne = size(irrig_layer.incidence, 2)
            for e in 1:ne
                members = findall(!iszero, @view irrig_layer.incidence[:, e])
                isempty(members) && continue

                zone_vol = mean(recs[members])
                zone_moisture = mean(projected[members])
                do_irrigate = zone_vol > 0.0f0

                zone_id = e <= length(irrig_layer.edge_ids) ?
                          irrig_layer.edge_ids[e] : "zone_$e"

                # Priority: higher deficit → higher priority
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
            # No irrigation layer — one recommendation per vertex
            for v in 1:nv
                vid = v <= length(soil_layer.vertex_ids) ?
                      soil_layer.vertex_ids[v] : "v_$v"
                push!(recommendations, Dict{String,Any}(
                    "zone_id" => vid,
                    "day" => day,
                    "irrigate" => recs[v] > 0.0f0,
                    "volume_liters" => Float64(recs[v] * 1000.0f0),
                    "priority" => Float64(clamp(recs[v] / DEFAULT_VALVE_CAPACITY, 0.0, 1.0)),
                    "projected_moisture" => Float64(projected[v]),
                    "trigger_reason" => recs[v] > 0.0f0 ? "below_wilting_point" : "moisture_adequate",
                ))
            end
        end

        # Apply irrigation to projected moisture for subsequent days
        projected .+= recs
    end

    return recommendations
end
