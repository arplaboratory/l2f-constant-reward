
#include "training.h"

#include <chrono>
#include <CLI/CLI.hpp>


template <typename T_ABLATION_SPEC>
int run(int argc, char** argv){
    using namespace learning_to_fly::config;

    using CONFIG = learning_to_fly::config::Config<T_ABLATION_SPEC>;
    using TI = typename CONFIG::TI;

    TI base_seed = 0;
    CLI::App app{"Hyperparameter optimization for Learning to Fly in Seconds"};

    TI num_runs = 100;
    app.add_option("-n,--num-runs", num_runs, "Number of runs with different seeds");
    std::string parameters_path = "parameters/output/crazyflie.json";
    std::string results_path;
    bool disable_error_integral = false;
    app.add_option("-f,--parameter-file", parameters_path, "Parameter file to load hyperparameters from");
#ifdef LEARNING_TO_FLY_HYPERPARAMETER_OPTIMIZATION
    app.add_option("-r,--result-file", results_path, "Path store the results (JSON)")->required();
#endif
    app.add_option("-s,--seed", base_seed, "Base seed (incremented for additional runs)");
    app.add_flag("--disable-error-integral", disable_error_integral, "Use error integral in the termination condition");


    CLI11_PARSE(app, argc, argv);

    for (TI run_i = 0; run_i < num_runs; run_i++){
        std::cout << "Run " << run_i << "\n";
        auto start = std::chrono::high_resolution_clock::now();
        learning_to_fly::TrainingState<CONFIG> ts;
        ts.parameters_path = parameters_path;
        ts.results_path = results_path;
        learning_to_fly::init(ts, base_seed + run_i);

        if(disable_error_integral){
            ts.env_eval.parameters.mdp.termination.orientation_integral_threshold = 999999999999;
            ts.env_eval.parameters.mdp.termination.position_integral_threshold = 999999999999;
            for(typename CONFIG::ENVIRONMENT& env: ts.validation_envs){
                env.parameters.mdp.termination.orientation_integral_threshold = 999999999999;
                env.parameters.mdp.termination.position_integral_threshold = 999999999999;
            }
            for (auto& env : ts.envs) {
                env.parameters.mdp.termination.orientation_integral_threshold = 999999999999;
                env.parameters.mdp.termination.position_integral_threshold = 999999999999;
            }

            for (auto& env : ts.off_policy_runner.envs) {
                env.parameters.mdp.termination.orientation_integral_threshold = 999999999999;
                env.parameters.mdp.termination.position_integral_threshold = 999999999999;
            }
        }

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
    static constexpr bool ENABLE_CURRICULUM = false;
    static constexpr bool USE_INITIAL_REWARD_FUNCTION = true;
    static constexpr bool EXPLORATION_NOISE_DECAY = false;
    static constexpr bool DISTURBANCE = true;
    static constexpr bool OBSERVATION_NOISE = false;
    static constexpr bool DOMAIN_RANDOMIZATION = false;
};

int main(int argc, char** argv){
    return run<ABLATION_SPEC>(argc, argv);
}