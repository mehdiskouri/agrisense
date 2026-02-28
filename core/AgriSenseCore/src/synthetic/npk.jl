# ---------------------------------------------------------------------------
# Synthetic NPK data generator
# ---------------------------------------------------------------------------

"""
    generate_npk_data(n_zones, n_weeks; ...) -> Dict

Outputs weekly nutrient channels `(n_weeks, n_zones)` with drift + fertilization jumps.
"""
function generate_npk_data(n_zones::Int, n_weeks::Int;
                            seed::Int=42,
                            use_gpu::Bool=HAS_CUDA,
                            dropout_rate::Float32=0.01f0,
                            )::Dict{String,Any}
    n_zones <= 0 && error("generate_npk_data: n_zones must be > 0")
    n_weeks <= 0 && error("generate_npk_data: n_weeks must be > 0")

    rng = MersenneTwister(seed + 300)
    week = to_backend(reshape(Float32.(0:n_weeks-1), n_weeks, 1); use_gpu=use_gpu)

    n0 = to_backend(reshape(rand(rng, Float32, n_zones) .* 25.0f0 .+ 70.0f0, 1, n_zones); use_gpu=use_gpu)
    p0 = to_backend(reshape(rand(rng, Float32, n_zones) .* 18.0f0 .+ 45.0f0, 1, n_zones); use_gpu=use_gpu)
    k0 = to_backend(reshape(rand(rng, Float32, n_zones) .* 22.0f0 .+ 55.0f0, 1, n_zones); use_gpu=use_gpu)

    dn = to_backend(reshape(rand(rng, Float32, n_zones) .* 0.45f0 .+ 0.20f0, 1, n_zones); use_gpu=use_gpu)
    dp = to_backend(reshape(rand(rng, Float32, n_zones) .* 0.30f0 .+ 0.10f0, 1, n_zones); use_gpu=use_gpu)
    dk = to_backend(reshape(rand(rng, Float32, n_zones) .* 0.35f0 .+ 0.12f0, 1, n_zones); use_gpu=use_gpu)

    fert_interval = 4
    week_cpu = Float32.(0:n_weeks-1)
    fert_mask = to_backend(reshape(Float32.((mod.(Int.(week_cpu), fert_interval) .== 0)), n_weeks, 1); use_gpu=use_gpu)
    fert_scale = to_backend(reshape(rand(rng, Float32, n_zones) .* 12.0f0 .+ 8.0f0, 1, n_zones); use_gpu=use_gpu)

    noise = correlated_noise(n_weeks, n_zones; seed=seed + 301, use_gpu=use_gpu)

    nitrogen = n0 .- week .* dn .+ fert_mask .* fert_scale .+ 1.5f0 .* noise
    phosphorus = p0 .- week .* dp .+ fert_mask .* (0.7f0 .* fert_scale) .+ 1.2f0 .* noise
    potassium = k0 .- week .* dk .+ fert_mask .* (0.85f0 .* fert_scale) .+ 1.3f0 .* noise
    organic_matter = 3.0f0 .+ 0.04f0 .* sin.(Float32(2π / 26) .* week) .+
                     0.03f0 .* correlated_noise(n_weeks, n_zones; seed=seed + 302, use_gpu=use_gpu)

    nitrogen = clamp.(nitrogen, 2.0f0, 240.0f0)
    phosphorus = clamp.(phosphorus, 1.0f0, 180.0f0)
    potassium = clamp.(potassium, 2.0f0, 220.0f0)
    organic_matter = clamp.(organic_matter, 0.5f0, 12.0f0)

    n_nan, mask = apply_dropout_with_mask(Float32.(nitrogen);
                                          rate=dropout_rate,
                                          seed=seed + 303,
                                          use_gpu=use_gpu)
    p_nan = ifelse.(mask, Float32(NaN), Float32.(phosphorus))
    k_nan = ifelse.(mask, Float32(NaN), Float32.(potassium))
    om_nan = ifelse.(mask, Float32(NaN), Float32.(organic_matter))

    return Dict{String,Any}(
        "layer" => "npk",
        "n_zones" => n_zones,
        "n_weeks" => n_weeks,
        "nitrogen_mg_kg" => cpu_plain(n_nan),
        "phosphorus_mg_kg" => cpu_plain(p_nan),
        "potassium_mg_kg" => cpu_plain(k_nan),
        "organic_matter_pct" => cpu_plain(om_nan),
        "missing_mask" => BitMatrix(cpu_plain(mask)),
    )
end
