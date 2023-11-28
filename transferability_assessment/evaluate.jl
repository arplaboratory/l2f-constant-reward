using DataFrames
using JSON
using CSV
using Statistics
using GLM

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

@assert all(missing_df.success .== 1.0)


missing_df

normalized_df = deepcopy(missing_df)
zscore_normalize!(col) = (col .-= mean(col)) ./= std(col)
foreach(zscore_normalize!, eachcol(normalized_df))

predictors = sum(Term.(Symbol.(names(normalized_df)[3:end])))
formula_rmse_without_z = Term(:rmse_without_z) ~ predictors
model_rmse = lm(formula_rmse_without_z, normalized_df)

coef_df_rmse = DataFrame(
    :name => coefnames(model_rmse),
    :coef => coef(model_rmse)
)

sort(coef_df_rmse, :coef, by=abs)

predictors = sum(Term.(Symbol.(names(normalized_df)[3:end])))
formula_success = Term(:success) ~ predictors
model_success = lm(formula_success, normalized_df)

coef_df_success = DataFrame(
    :name => coefnames(model_success),
    :coef => coef(model_success)
)

sort(coef_df_success, :coef, by=abs)
