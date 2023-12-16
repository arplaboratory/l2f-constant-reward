using DataFrames
using JSON
using CSV
using Statistics
using GLM
using Plots
using GaussianProcesses
using Random
using Optim
using LinearAlgebra

json_data = JSON.parsefile("transferability_assessment/data/data_table.json")
df = DataFrame(json_data)
rename!(df, "checkpoint" => "step")

validation_dir = "transferability_assessment/data/validation_hpc_300000"

validation_data = []
for seed_run in readdir(validation_dir, join=true)
    pattern = r"_([a-z\+\-]+)_0*(\d+)$"
    match_result = match(pattern, seed_run)
    config = match_result.captures[1]
    seed = parse(Int, match_result.captures[2])
    println(config, " ", seed)
    for checkpoint in readdir(seed_run, join=true)
        if endswith(checkpoint, ".json")
            pattern = r"_(\w+).json$"
            match_result = match(pattern, checkpoint)
            step = parse(Int, match_result.captures[1])
            checkpoint_data = JSON.parsefile(checkpoint)
            row = deepcopy(checkpoint_data)
            row["config"] = config
            row["seed"] = seed
            row["step"] = step
            push!(validation_data, row)
        end
    end
end

validation_df = DataFrame(validation_data)
names(validation_df)
# relevant_val_df = validation_df[!, ["config", "step", "seed", "MaxErrorMean(Position, after 200 steps)"]]
relevant_df = df[df.test.=="figure_eight_tracking_normal", :]


joined_df = deepcopy(innerjoin(df, validation_df, on=["config", "step", "seed"]))
regression_df = select(joined_df, Not([:step, :config, :filename, :seed, :test]))
transform!(regression_df, :rmse_without_z => ByRow(x-> isnothing(x) ? NaN : Float64(x)) => :rmse_without_z)
transform!(regression_df, :success => ByRow(Float64) => :success)


for n in names(regression_df)
    println(n)
end

nan_df = deepcopy(regression_df)
replace_nothing_with_nan(x) = isnothing(x) ? NaN : x
replace_nothing_with_nan_string(x) = isnothing(x) ? "nan" : x
for col in names(nan_df)
    if eltype(nan_df[!, col]) <: Union{Number, Nothing}
        transform!(nan_df, col => ByRow(replace_nothing_with_nan) => col)
    else
        transform!(nan_df, col => ByRow(replace_nothing_with_nan_string) => col)
    end
end

missing_df = deepcopy(nan_df)
replace_nan_with_missing(x) = isnan(x) ? missing : x
for col in names(missing_df)
    transform!(missing_df, col => ByRow(replace_nan_with_missing) => col)
end
dropmissing!(missing_df)

CSV.write("transferability_assessment/data/regression_data.csv", nan_df)
CSV.write("transferability_assessment/data/regression_data_clean.csv", missing_df)
missing_df = DataFrame(shuffle(eachrow(missing_df)))

@assert all(missing_df.success .== 1.0)


begin
X = transpose(Array(missing_df[!, 3:end]))
y = Array(missing_df[!, :rmse_without_z])
order = randperm(size(X, 2))
X = X[:, order]
y = y[order]


# rmse_mean = mean(missing_df.rmse_without_z)
# rmse_std = std(missing_df.rmse_without_z)
normalized_df = deepcopy(missing_df)
# zscore_normalize!(col) = (col .-= mean(col)) ./= std(col)
# foreach(zscore_normalize!, eachcol(normalized_df))
train_points = 120
X_train = X[:, 1:train_points]
y_train = y[1:train_points]
X_test = X[:, train_points+1:end]
y_test = y[train_points+1:end]

regularization_term = 1.0
X_pad = vcat(ones(1, size(X_train, 2)), X_train)
weights = pinv(X_pad * transpose(X_pad) + I*regularization_term) * X_pad * y_train

X_test_pad = vcat(ones(1, size(X_test, 2)), X_test)
predicted_rmse = (transpose(weights) * X_test_pad)[:]
begin
mkpath("transferability_assessment/figures")
p = plot()
scatter!(p, y_test, predicted_rmse, xlabel="Actual Position Error", ylabel="Predicted Position Error", legend=false)
savefig(p, "transferability_assessment/figures/linear_regression_correlation_plot_manual.pdf")
display(p)
end
model_json = JSON.json(
    Dict(
        "names" => names(missing_df)[3:end],
        "coefs" => weights
    ),
)
open("transferability_assessment/linear_regression_model_manual.json", "w") do f
    write(f, model_json)
end
end



predictors = sum(Term.(Symbol.(sort(names(normalized_df)[3:end]))))
formula_rmse_without_z = Term(:rmse_without_z) ~ predictors
n_training_samples = size(normalized_df)[1]
model_rmse = lm(formula_rmse_without_z, first(normalized_df, n_training_samples))

coef_df_rmse = DataFrame(
    :name => coefnames(model_rmse),
    :coef => coef(model_rmse)
)

model_rmse

# predicted_rmse = predict(model_rmse, normalized_df[n_training_samples+1:end, :])# .* rmse_std .+ rmse_mean
# actual_rmse = missing_df.rmse_without_z[n_training_samples+1:end]

predicted_rmse = predict(model_rmse, normalized_df[1:n_training_samples, :])# .* rmse_std .+ rmse_mean
actual_rmse = missing_df.rmse_without_z[1:n_training_samples]


begin
mkpath("transferability_assessment/figures")
p = plot()
scatter!(p, actual_rmse, predicted_rmse, xlabel="Actual Position Error", ylabel="Predicted Position Error", legend=false)
savefig(p, "transferability_assessment/figures/linear_regression_correlation_plot.pdf")
display(p)
end

model_coef_names = coefnames(model_rmse)
model_coefs = coef(model_rmse)
model_coefs[1] += rmse_mean
model_coefs[2:end] .*= rmse_std

model_json = JSON.json(
    Dict(
        "names" => model_coef_names,
        "coefs" => model_coefs
    ),
)

open("transferability_assessment/linear_regression_model.json", "w") do f
    write(f, model_json)
end


results_json = open("transferability_assessment/results.json", "r") do f
    read(f, String)
end

results = JSON.parse(results_json)

X

query = [1, [results[n] for n in names(missing_df)[3:end]]...]
pred = dot(weights, query)

results_df = DataFrame([results])
[]
predicted_rmse = predict(model_rmse, results_df)


sort(coef_df_rmse, :coef, by=abs)




train_points = 100
X_train = X[:, 1:train_points]
y_train = y[1:train_points]
X_test = X[:, train_points+1:end]
y_test = y[train_points+1:end]


mZero = MeanZero()
kern = SE(0.0,0.0)

logObsNoise = -1.0
gp = GP(X_train,y_train,mZero,kern,logObsNoise)
optimize!(gp, method=ConjugateGradient())

pred_mean, pred_std = predict_y(gp, X_test);
scatter(pred_mean, y_test, xlabel="Predicted Position Error", ylabel="Actual Position Error", legend=false)

# predictors = sum(Term.(Symbol.(names(normalized_df)[3:end])))
# formula_success = Term(:success) ~ predictors
# model_success = lm(formula_success, normalized_df)

# coef_df_success = DataFrame(
#     :name => coefnames(model_success),
#     :coef => coef(model_success)
# )

# sort(coef_df_success, :coef, by=abs)
