using DataFrames
using JSON
using CSV

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

for n in names(regression_df)
    println(n)
end

nan_df = deepcopy(regression_df)
replace_nothing_with_nan(x) = isnothing(x) ? NaN : x
for col in names(nan_df)
    if eltype(nan_df[!, col]) <: Union{Number, Missing}
        transform!(nan_df, col => ByRow(replace_nothing_with_nan) => col)
    end
end

nan_df


CSV.write("transferability_assessment/data/regression_data.csv", nan_df)

