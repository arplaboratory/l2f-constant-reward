using JSON
using LinearAlgebra
using Plots
using Statistics
using NaNStatistics


checkpoints_integral_error = [
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
]
mode = "constant_reward_error_integral"

for (i, checkpoint) in enumerate(checkpoints_integral_error)
    run(`cmake-build-release/src/evaluate_actor_headless_sq_rwd --checkpoint $(checkpoint)`)
    mkpath("data_figures")
    cp("data/evaluate_actor_headless.json", "data_figures/$(mode)_$i.json")
end



# Load the data

episodes = []

for i in 1:10
    data = JSON.parsefile("data_figures/$(mode)_$i.json")
    push!(episodes, data["episodes"]...)
end

episodes

episode_length = 500

data_rmse_position_error = [
    [[norm(state["position"]) for state in episode["states"]]; fill(NaN, episode_length-length(episode["states"]))]
    for episode in episodes
]

data_rmse_position_error_cat = hcat(data_rmse_position_error...)

data_rmse_mean = nanmean(data_rmse_position_error_cat, dims=2)[:]
data_rmse_std = nanstd(data_rmse_position_error_cat, dims=2)[:]
data_rmse_position_error_cat
data_not_nan = [sum(isnan.(row)) for row in eachrow(data_rmse_position_error_cat)] ./ size(data_rmse_position_error_cat, 2)

p = plot()
plot!(p, data_rmse_mean, ribbon=data_rmse_std, legend=false)
p2 = twinx(p)
plot!(p2, data_not_nan, color=:red, legend=false)



begin
p = plot()
for rmse_series in data_rmse_position_error
    plot!(p, rmse_series, legend=false)
end
p
end




