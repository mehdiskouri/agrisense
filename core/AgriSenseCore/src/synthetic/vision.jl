# ---------------------------------------------------------------------------
# Synthetic vision / CV event generator
# ---------------------------------------------------------------------------

"""
    generate_vision_data(n_cameras, n_beds, n_steps; ...) -> Dict

Generates per-bed CV outcomes:
- anomaly_code: 0 none, 1 pest, 2 disease
- confidence, canopy_coverage_pct
- camera assignment and stochastic spatial clustering
"""
function generate_vision_data(n_cameras::Int, n_beds::Int, n_steps::Int;
                               seed::Int=42,
                               use_gpu::Bool=HAS_CUDA,
                               dropout_rate::Float32=SYNTHETIC_DROPOUT_RATE,
                               )::Dict{String,Any}
    n_cameras <= 0 && error("generate_vision_data: n_cameras must be > 0")
    n_beds <= 0 && error("generate_vision_data: n_beds must be > 0")
    n_steps <= 0 && error("generate_vision_data: n_steps must be > 0")

    rng = MersenneTwister(seed + 400)

    # camera assignment (one camera can cover multiple beds)
    camera_for_bed = Int32[(mod(i - 1, n_cameras) + 1) for i in 1:n_beds]

    # adjacency: line topology for clustering effects
    adjacency = zeros(Float32, n_beds, n_beds)
    for i in 1:n_beds
        i > 1 && (adjacency[i, i - 1] = 1.0f0)
        i < n_beds && (adjacency[i, i + 1] = 1.0f0)
    end

    # Base anomaly probabilities from PRD guidance
    p_pest = 0.02f0
    p_disease = 0.005f0
    cluster_gain = 0.12f0

    pest = falses(n_steps, n_beds)
    disease = falses(n_steps, n_beds)
    base_pest_rand = rand(rng, Float32, n_steps, n_beds)
    base_dis_rand = rand(rng, Float32, n_steps, n_beds)

    for t in 1:n_steps
        prev = t == 1 ? zeros(Float32, n_beds) : Float32.(pest[t - 1, :] .| disease[t - 1, :])
        neigh = adjacency * prev
        cluster_p = clamp.(p_pest .+ cluster_gain .* neigh, 0.0f0, 0.6f0)
        cluster_d = clamp.(p_disease .+ 0.5f0 * cluster_gain .* neigh, 0.0f0, 0.4f0)

        pest[t, :] .= base_pest_rand[t, :] .< cluster_p
        disease[t, :] .= (base_dis_rand[t, :] .< cluster_d) .& .!pest[t, :]
    end

    anomaly_code = zeros(Int8, n_steps, n_beds)
    anomaly_code[pest] .= 1
    anomaly_code[disease] .= 2

    progress = to_backend(reshape(Float32.(0:n_steps-1) ./ Float32(max(n_steps - 1, 1)), n_steps, 1); use_gpu=use_gpu)
    canopy = 20.0f0 .+ 75.0f0 .* (1.0f0 .- exp.(-3.0f0 .* progress)) .+
             2.5f0 .* correlated_noise(n_steps, n_beds; seed=seed + 401, use_gpu=use_gpu)
    canopy = clamp.(canopy, 1.0f0, 100.0f0)

    confidence = 0.08f0 .+ 0.15f0 .* rand(rng, Float32, n_steps, n_beds)
    confidence[pest] .= 0.65f0 .+ 0.30f0 .* rand(rng, Float32, count(pest))
    confidence[disease] .= 0.70f0 .+ 0.25f0 .* rand(rng, Float32, count(disease))
    confidence = clamp.(confidence, 0.0f0, 1.0f0)

    canopy_nan, mask = apply_dropout_with_mask(Float32.(canopy);
                                               rate=dropout_rate,
                                               seed=seed + 402,
                                               use_gpu=use_gpu)
    conf_nan = ifelse.(mask, Float32(NaN), Float32.(confidence))

    # For anomaly code, set -1 where missing (still plain integer matrix)
    anomaly_out = copy(anomaly_code)
    anomaly_out[cpu_plain(mask)] .= Int8(-1)

    return Dict{String,Any}(
        "layer" => "vision",
        "n_cameras" => n_cameras,
        "n_beds" => n_beds,
        "n_steps" => n_steps,
        "camera_for_bed" => Vector{Int32}(camera_for_bed),
        "anomaly_code" => Matrix{Int8}(anomaly_out),
        "anomaly_legend" => Dict("-1" => "missing", "0" => "none", "1" => "pest", "2" => "disease"),
        "confidence" => cpu_plain(conf_nan),
        "canopy_coverage_pct" => cpu_plain(canopy_nan),
        "missing_mask" => BitMatrix(cpu_plain(mask)),
    )
end
