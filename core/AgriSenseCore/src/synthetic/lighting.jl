# ---------------------------------------------------------------------------
# Synthetic lighting / PAR data generator
# ---------------------------------------------------------------------------

"""
    generate_lighting_data(n_sensors, n_steps; ...) -> Dict

Generates greenhouse lighting channels `(n_steps, n_sensors)`:
PAR, cumulative DLI, duty_cycle_pct and spectrum_index.
"""
function generate_lighting_data(n_sensors::Int, n_steps::Int;
                                 seed::Int=42,
                                 use_gpu::Bool=HAS_CUDA,
                                 dropout_rate::Float32=SYNTHETIC_DROPOUT_RATE,
                                 cadence_minutes::Int=CADENCE_MINUTES,
                                 )::Dict{String,Any}
    n_sensors <= 0 && error("generate_lighting_data: n_sensors must be > 0")
    n_steps <= 0 && error("generate_lighting_data: n_steps must be > 0")

    rng = MersenneTwister(seed + 500)
    t = time_grid(n_steps, cadence_minutes)
    th = to_backend(reshape(t, n_steps, 1); use_gpu=use_gpu)
    ω = Float32(2π / 24)

    profile = max.(0.0f0, sin.(ω .* th .- 1.1f0))
    sensor_scale = to_backend(reshape(rand(rng, Float32, n_sensors) .* 120.0f0 .+ 520.0f0, 1, n_sensors); use_gpu=use_gpu)

    par = sensor_scale .* profile .+
          18.0f0 .* correlated_noise(n_steps, n_sensors; seed=seed + 501, use_gpu=use_gpu)
    par = clamp.(par, 0.0f0, 1200.0f0)

    # Approximate DLI accumulation (running over horizon)
    step_seconds = Float32(cadence_minutes * 60)
    umol_to_mol = 1.0f-6
    dli = cumsum(par .* (step_seconds * umol_to_mol); dims=1)

    duty = clamp.(100.0f0 .* par ./ max.(sensor_scale, 1.0f0), 0.0f0, 100.0f0)
    spectrum_idx = clamp.(0.5f0 .+ 0.1f0 .* sin.(ω .* th .+ 0.4f0) .+
                          0.03f0 .* correlated_noise(n_steps, n_sensors; seed=seed + 502, use_gpu=use_gpu),
                          0.0f0, 1.0f0)

    par_nan, mask = apply_dropout_with_mask(Float32.(par);
                                            rate=dropout_rate,
                                            seed=seed + 503,
                                            use_gpu=use_gpu)
    dli_nan = ifelse.(mask, Float32(NaN), Float32.(dli))
    duty_nan = ifelse.(mask, Float32(NaN), Float32.(duty))
    spectrum_nan = ifelse.(mask, Float32(NaN), Float32.(spectrum_idx))

    return Dict{String,Any}(
        "layer" => "lighting",
        "n_sensors" => n_sensors,
        "n_steps" => n_steps,
        "cadence_minutes" => cadence_minutes,
        "par_umol" => cpu_plain(par_nan),
        "dli_cumulative" => cpu_plain(dli_nan),
        "duty_cycle_pct" => cpu_plain(duty_nan),
        "spectrum_index" => cpu_plain(spectrum_nan),
        "missing_mask" => BitMatrix(cpu_plain(mask)),
    )
end
