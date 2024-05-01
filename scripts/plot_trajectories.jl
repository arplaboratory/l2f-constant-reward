using JSON
using LinearAlgebra
using Plots
using Statistics
using NaNStatistics


checkpoints = Dict(
    "constant_reward_error_integral" => [
        "checkpoints/multirotor_td3/2024_04_30_20_37_30_d-o-a+r+h+c-f+w+e-_000/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_20_38_22_d-o-a+r+h+c-f+w+e-_001/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_20_39_49_d-o-a+r+h+c-f+w+e-_002/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_20_41_41_d-o-a+r+h+c-f+w+e-_003/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_20_43_19_d-o-a+r+h+c-f+w+e-_004/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_20_45_14_d-o-a+r+h+c-f+w+e-_005/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_20_47_16_d-o-a+r+h+c-f+w+e-_006/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_20_49_10_d-o-a+r+h+c-f+w+e-_007/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_20_51_07_d-o-a+r+h+c-f+w+e-_008/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_20_53_13_d-o-a+r+h+c-f+w+e-_009/actor_000000001000000.h5",
    ],
    "constant_reward" => [
        "checkpoints/multirotor_td3/2024_04_30_21_59_47_d-o-a+r+h+c-f+w+e-_000/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_22_00_39_d-o-a+r+h+c-f+w+e-_001/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_22_02_13_d-o-a+r+h+c-f+w+e-_002/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_22_03_52_d-o-a+r+h+c-f+w+e-_003/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_22_05_20_d-o-a+r+h+c-f+w+e-_004/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_22_06_47_d-o-a+r+h+c-f+w+e-_005/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_22_08_26_d-o-a+r+h+c-f+w+e-_006/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_22_10_14_d-o-a+r+h+c-f+w+e-_007/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_22_11_42_d-o-a+r+h+c-f+w+e-_008/actor_000000001000000.h5",
        "checkpoints/multirotor_td3/2024_04_30_22_13_09_d-o-a+r+h+c-f+w+e-_009/actor_000000001000000.h5",
    ]
)

mode_name_lookup = Dict(
    "constant_reward_error_integral" => "Constant Reward & Error Integral",
    "constant_reward" => "Constant Reward"
)

modes = collect(keys(checkpoints))

for mode in modes
    for (i, checkpoint) in enumerate(checkpoints[mode])
        run(`cmake-build-release/src/evaluate_actor_headless_sq_rwd --checkpoint $(checkpoint)`)
        mkpath("data_figures")
        cp("data/evaluate_actor_headless.json", "data_figures/$(mode)_$i.json", force=true)
    end
end



# Load the data

episodes = Dict{String, Any}()

aggregate = Dict{String, Any}()

final_positions = Dict{String, Any}()


for mode in modes
    episodes[mode] = []
    means = []
    stds = []
    current_final_positions = []
    for i in 1:10
        data = JSON.parsefile("data_figures/$(mode)_$i.json")
        push!(episodes[mode], data["episodes"]...)
        # episodes_run = [[[norm(state["position"]) for state in episode["states"]]; fill(NaN, episode_length-length(episode["states"]))] for episode in data["episodes"]]
        # episodes_run_cat = hcat(episodes_run...)
        # push!(means, nanmean(episodes_run_cat, dims=2)[:])
        # push!(stds, nanstd(episodes_run_cat, dims=2)[:])
        push!(current_final_positions, [episode["states"][end]["position"] for episode in data["episodes"]])
    end
    final_positions[mode] = hcat(vcat(current_final_positions...)...)
    # aggregate[mode] = Dict(
    #     "means" => hcat(means...),
    #     "stds" => hcat(stds...)
    # )
end

data_positions = Dict([(mode, [
    [state["position"] for state in episode["states"]]
    for episode in episodes[mode]
]) for mode in modes])

data_positions_cat = Dict([(mode, hcat(vcat(data_positions[mode]...)...)) for mode in modes])

final_positions[modes[1]]

mode_color_lookup = Dict(
    "constant_reward_error_integral" => "#7DB9B6",
    "constant_reward" => "#b8b8b8"
)

begin
    p = plot(grid=false, aspect_ratio=:equal, size=(700, 700))
    for (i, mode) in enumerate(modes)
        for episode in episodes[mode]
            positions = hcat([state["position"] for state in episode["states"]]...)
            plot!(p, positions[1, :], positions[2, :], color=mode_color_lookup[mode], linewidth=0.05, label=nothing)
        end
        # scatter!(p, final_positions[mode][1, :], final_positions[mode][2, :], label=mode, color=colors[i], markersize=1)
    end
    for (i, mode) in enumerate(modes)
        mode_mu = nanmean(data_positions_cat[mode], dims=2)
        mode_cov = cov(data_positions_cat[mode]')
        eigvals, eigvecs = eigen(mode_cov)
        println("Mode cov: $mode_cov")
        axis_lengths = sqrt.(eigvals)  # lengths of the axes of the ellipse

        angle = atan(eigvecs[2, 1], eigvecs[1, 1])
        θ = range(0, stop=2π, length=100)
        ellipse_x = axis_lengths[1] * cos.(θ)
        ellipse_y = axis_lengths[2] * sin.(θ)

        rotated_ellipse_x = cos(angle) * ellipse_x - sin(angle) * ellipse_y
        rotated_ellipse_y = sin(angle) * ellipse_x + cos(angle) * ellipse_y

        final_ellipse_x = rotated_ellipse_x .+ mode_mu[1]
        final_ellipse_y = rotated_ellipse_y .+ mode_mu[2]
        plot!(final_ellipse_x, final_ellipse_y, label=nothing, color=mode_color_lookup[mode], linewidth=4, linestrokewidth=1)
    end
    for (i, mode) in enumerate(modes)
        mode_mu = nanmean(data_positions_cat[mode], dims=2)
        mode_std = nanstd(data_positions_cat[mode], dims=2)
        mode_cov = cov(data_positions_cat[mode]')
        scatter!(p, [mode_mu[1]], [mode_mu[2]], label=mode_name_lookup[mode], color=mode_color_lookup[mode], markersize=5, markerstrokewidth=0.5)
    end
    scatter!(p, [0], [0], label="Target (origin)", color="black", markersize=5)
    xlabel!(p, "x [m]")
    ylabel!(p, "y [m]")
    p
end

begin
    p = plot()
    for mode in modes
        scatter!(p, final_positions[mode][1, :], final_positions[mode][2, :], label=mode, legend=false)
    end
    p
end



aggregate[modes[2]]["means"]



begin
p = plot()
for mode in modes
    mode_mu = nanmean(aggregate[mode]["means"], dims=2)
    mode_std = nanstd(aggregate[mode]["means"], dims=2)
    plot!(p, mode_mu, ribbon=mode_std, label=mode, legend=false)
end
p
end


episode_length = 500



data_rmse_position_error = Dict([(mode, [
    [[norm(state["position"]) for state in episode["states"]]; fill(NaN, episode_length-length(episode["states"]))]
    for episode in episodes[mode]
]) for mode in modes])

data_rmse_position_error_cat = Dict([(mode, hcat(data_rmse_position_error[mode]...)) for mode in modes])

data_rmse_mean = Dict([(mode, nanmean(data_rmse_position_error_cat[mode], dims=2)[:]) for mode in modes])
data_rmse_std = Dict([(mode, nanstd(data_rmse_position_error_cat[mode], dims=2)[:]) for mode in modes])
data_not_nan = Dict([(mode, [sum(isnan.(row)) for row in eachrow(data_rmse_position_error_cat[mode])] ./ size(data_rmse_position_error_cat[mode], 2)) for mode in modes])

begin
p = plot()
for mode in modes
    plot!(p, data_rmse_mean[mode], ribbon=data_rmse_std[mode], label=mode, legend=false)
    p2 = twinx(p)
    plot!(p2, data_not_nan[mode], color=:red, legend=false, label=mode)
end
p
end

num_samples = size(data_positions_cat)
r = range(1, num_samples[2], step=100)
p = plot()
scatter!(p, data_positions_cat[modes[1]][1, r], data_positions_cat[modes[1]][2, r], data_positions_cat[modes[1]][3, r], legend=false, aspect_ratio=:equal, markersize=1)
scatter!(p, data_positions_cat[modes[2]][1, r], data_positions_cat[modes[2]][2, r], data_positions_cat[modes[2]][3, r], legend=false, aspect_ratio=:equal, markersize=1)


std(norm.(data_positions_cat[modes[1]]))
std(norm.(data_positions_cat[modes[2]]))



begin
p = plot()
for rmse_series in data_rmse_position_error
    plot!(p, rmse_series, legend=false)
end
p
end


