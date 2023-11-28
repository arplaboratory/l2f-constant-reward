using Plots
using OrderedCollections
using JSON
using ColorSchemes

color_palette = ColorSchemes.tab10

data = OrderedDict()

for hpo_framework_results_file in reverse(sort(readdir("hpo/hpo_results")))
    hpo_framework_results = JSON.parsefile("hpo/hpo_results/" * hpo_framework_results_file)
    hpo_framework = splitext(hpo_framework_results_file)[1]
    data[hpo_framework] = hpo_framework_results

end


keys(data)

data["bayesian_optimization"]

color_palette = map(x -> color_palette[x], 1:size(color_palette))

begin
p = plot()
for (i, (hpo_framework, hpo_framework_results)) in enumerate(data)
    current_color = color_palette[i % length(color_palette) + 1]
    plot!(p, [r[1]["mdp.gamma"] for r in hpo_framework_results], label=nothing, color=current_color, linewidth=3)
end
for (i, (hpo_framework, hpo_framework_results)) in enumerate(data)
    current_color = color_palette[i % length(color_palette) + 1]
    max_index = argmax([r[2] for r in hpo_framework_results])
    scatter!(p, [max_index], [hpo_framework_results[max_index][1]["mdp.gamma"]], label=hpo_framework, color=current_color, markersize=8)
end
ylabel!("γ")
xlabel!("Acquisition step")
display(p)
mkpath("hpo/figures")
savefig(p, "hpo/figures/find_gamma_progression_plot.pdf")
end

begin


begin
p = plot()
for (i, (hpo_framework, hpo_framework_results)) in enumerate(data)
    framework_data = []
    for r in hpo_framework_results
        push!(framework_data, [r[1]["mdp.gamma"], r[2]])
    end
    framework_data = hcat(framework_data...)
    scatter!(p, framework_data[1, :], framework_data[2, :], color=color_palette[i % length(color_palette) + 1], markersize=5, label=hpo_framework)
end
xlabel!("γ")
ylabel!("Episode Lengths")
savefig(p, "hpo/figures/find_gamma_data.pdf")
display(p)
end


end