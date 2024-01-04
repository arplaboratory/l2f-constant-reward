
#include "training.h"

#include <chrono>
#ifdef LEARNING_TO_FLY_HYPERPARAMETER_OPTIMIZATION
#include <CLI/CLI.hpp>
#endif


template <typename T_ABLATION_SPEC>
int run(int argc, char** argv){
    using namespace learning_to_fly::config;

    using CONFIG = learning_to_fly::config::Config<T_ABLATION_SPEC>;
    using TI = typename CONFIG::TI;

    TI base_seed = 0;
#ifdef LEARNING_TO_FLY_HYPERPARAMETER_OPTIMIZATION
    CLI::App app{"Hyperparameter optimization for Learning to Fly in Seconds"};

    TI num_runs = 1;
    app.add_option("-n,--num-runs", num_runs, "Number of runs with different seeds");
    std::string parameters_path, results_path;
    app.add_option("-f,--parameter-file", parameters_path, "Parameter file to load hyperparameters from")->required();
    app.add_option("-r,--result-file", results_path, "Path store the results (JSON)")->required();
    app.add_option("-s,--seed", base_seed, "Base seed (incremented for additional runs)");


    CLI11_PARSE(app, argc, argv);
#else

#ifdef LEARNING_TO_FLY_IN_SECONDS_BENCHMARK
    TI num_runs = 1;
#else
    TI num_runs = 1;
#endif
#endif

    for (TI run_i = 0; run_i < num_runs; run_i++){
        std::cout << "Run " << run_i << "\n";
        auto start = std::chrono::high_resolution_clock::now();
        learning_to_fly::TrainingState<CONFIG> ts;
#ifdef LEARNING_TO_FLY_HYPERPARAMETER_OPTIMIZATION
        ts.parameters_path = parameters_path;
        ts.results_path = results_path;
#else
        std::string model_name = "x500_real";
        ts.parameters_path = std::filesystem::path("parameters") / "output" / (model_name + ".json");
#endif
        learning_to_fly::init(ts, base_seed + run_i);

        for(TI step_i=0; step_i < CONFIG::STEP_LIMIT; step_i++){
            learning_to_fly::step(ts);
        }

        learning_to_fly::destroy(ts);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Training took: " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << "s" << std::endl;
    }
    return 0;
}

struct ABLATION_SPEC: learning_to_fly::config::DEFAULT_ABLATION_SPEC{
    static constexpr bool ENABLE_CURRICULUM = true;
    static constexpr bool USE_INITIAL_REWARD_FUNCTION = true;
    static constexpr bool EXPLORATION_NOISE_DECAY = false;
    static constexpr bool DISTURBANCE = false;
    static constexpr bool OBSERVATION_NOISE = true;
};

int main(int argc, char** argv){
    return run<ABLATION_SPEC>(argc, argv);
}