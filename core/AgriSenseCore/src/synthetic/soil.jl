# ---------------------------------------------------------------------------
# Synthetic soil data generator
# ---------------------------------------------------------------------------

"""
    generate_soil_data(n_sensors, n_steps; seed, use_gpu, dropout_rate,
                       rainfall_mm, irrigation_mm) -> Dict

Generates soil channels in shape `(n_steps, n_sensors)`:
- moisture [0,1]
- temperature [°C]
- conductivity [dS/m]
- ph [0,14]

Missingness is represented as NaN + explicit boolean mask.
"""
function generate_soil_data(n_sensors::Int, n_steps::Int;
                             seed::Int=42,
                             use_gpu::Bool=HAS_CUDA,
                             dropout_rate::Float32=SYNTHETIC_DROPOUT_RATE,
                             rainfall_mm::Union{Nothing,AbstractVector{Float32}}=nothing,
                             irrigation_mm::Union{Nothing,AbstractVector{Float32}}=nothing,
                             cadence_minutes::Int=CADENCE_MINUTES,
                             )::Dict{String,Any}
    n_sensors <= 0 && error("generate_soil_data: n_sensors must be > 0")
    n_steps <= 0 && error("generate_soil_data: n_steps must be > 0")

    t = time_grid(n_steps, cadence_minutes)
    th = to_backend(reshape(t, n_steps, 1); use_gpu=use_gpu)

    rng = MersenneTwister(seed + 100)

    # Per-sensor baselines
    base_m = to_backend(reshape(rand(rng, Float32, n_sensors) .* 0.12f0 .+ 0.22f0, 1, n_sensors); use_gpu=use_gpu)
    decay = to_backend(reshape(rand(rng, Float32, n_sensors) .* 0.05f0 .+ 0.03f0, 1, n_sensors); use_gpu=use_gpu)
    temp_offset = to_backend(reshape(rand(rng, Float32, n_sensors) .* 2.0f0 .- 1.0f0, 1, n_sensors); use_gpu=use_gpu)

    # Diurnal cycles
    ω = Float32(2π / 24)
    diurnal = sin.(ω .* th)
    diurnal_shift = sin.(ω .* th .+ 0.6f0)

    # Rainfall / irrigation impulses (shared forcing)
    rain = if rainfall_mm === nothing
        rrng = MersenneTwister(seed + 101)
        event = rand(rrng, Float32, n_steps) .< 0.06f0
        Float32.(event) .* (rand(rrng, Float32, n_steps) .* 8.0f0)
    else
        Float32.(rainfall_mm)
    end
    irrig = if irrigation_mm === nothing
        irng = MersenneTwister(seed + 102)
        event = rand(irng, Float32, n_steps) .< 0.05f0
        Float32.(event) .* (rand(irng, Float32, n_steps) .* 10.0f0 .+ 4.0f0)
    else
        Float32.(irrigation_mm)
    end
    rain_col = to_backend(reshape(rain, n_steps, 1); use_gpu=use_gpu)
    irrig_col = to_backend(reshape(irrig, n_steps, 1); use_gpu=use_gpu)

    # Correlated sensor noise
    corr_noise = correlated_noise(n_steps, n_sensors;
                                  seed=seed + 103,
                                  use_gpu=use_gpu)

    # Synthetic moisture dynamics (decay + impulses + diurnal + noise)
    trend = (th ./ Float32(max(n_steps, 1))) .* decay
    moisture = base_m .+ 0.05f0 .* diurnal .+ 0.008f0 .* rain_col .+
               0.010f0 .* irrig_col .- trend .+ 0.015f0 .* corr_noise
    moisture = clamp.(moisture, 0.03f0, 0.95f0)

    # Temperature coupled to diurnal cycle and inverse moisture
    temperature = 21.0f0 .+ temp_offset .+ 7.0f0 .* diurnal_shift .+
                  1.2f0 .* (0.45f0 .- moisture) .+
                  0.6f0 .* correlated_noise(n_steps, n_sensors;
                                            seed=seed + 104,
                                            use_gpu=use_gpu)

    conductivity = 1.4f0 .+ 1.1f0 .* (0.5f0 .- moisture) .+
                   0.08f0 .* correlated_noise(n_steps, n_sensors;
                                              seed=seed + 105,
                                              use_gpu=use_gpu)
    conductivity = clamp.(conductivity, 0.1f0, 6.0f0)

    ph = 6.5f0 .+ 0.12f0 .* sin.(ω .* th .+ 1.2f0) .+
         0.05f0 .* correlated_noise(n_steps, n_sensors;
                                    seed=seed + 106,
                                    use_gpu=use_gpu)
    ph = clamp.(ph, 4.5f0, 8.5f0)

    # Shared missingness mask across all soil channels (sensor dropout)
    moisture_nan, mask = apply_dropout_with_mask(Float32.(moisture);
                                                 rate=dropout_rate,
                                                 seed=seed + 107,
                                                 use_gpu=use_gpu)
    temperature_nan = ifelse.(mask, Float32(NaN), Float32.(temperature))
    conductivity_nan = ifelse.(mask, Float32(NaN), Float32.(conductivity))
    ph_nan = ifelse.(mask, Float32(NaN), Float32.(ph))

    return Dict{String,Any}(
        "layer" => "soil",
        "n_sensors" => n_sensors,
        "n_steps" => n_steps,
        "cadence_minutes" => cadence_minutes,
        "moisture" => cpu_plain(moisture_nan),
        "temperature" => cpu_plain(temperature_nan),
        "conductivity" => cpu_plain(conductivity_nan),
        "ph" => cpu_plain(ph_nan),
        "missing_mask" => BitMatrix(cpu_plain(mask)),
        "rainfall_mm" => Vector{Float32}(rain),
        "irrigation_mm" => Vector{Float32}(irrig),
    )
end
