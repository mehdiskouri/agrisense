# ---------------------------------------------------------------------------
# Synthetic vision / CV event generator
# ---------------------------------------------------------------------------

"""
    generate_vision_data(n_cameras, n_beds, n_steps; seed) -> Dict

Generate synthetic CV inference results with stochastic anomaly events.
"""
function generate_vision_data(n_cameras::Int, n_beds::Int, n_steps::Int;
                               seed::Int=42)::Dict{String,Any}
    # Stub — full implementation in Phase 4
    return Dict{String,Any}("layer" => "vision", "n_cameras" => n_cameras, "n_steps" => n_steps)
end
