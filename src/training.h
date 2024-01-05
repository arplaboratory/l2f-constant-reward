#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <learning_to_fly/simulator/operations_cpu.h>

#include "config/config.h"

#include <rl_tools/rl/algorithms/td3/loop.h>


#include "training_state.h"

#include "init/load_config.h"

#include "steps/checkpoint.h"
#include "steps/critic_reset.h"
#include "steps/curriculum.h"
#include "steps/log_reward.h"
#include "steps/logger.h"
#include "steps/trajectory_collection.h"
#include "steps/validation.h"

#include "helpers.h"

#include <filesystem>
#include <fstream>


namespace learning_to_fly{

    template <typename T_CONFIG>
    void init(TrainingState<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using ABLATION_SPEC = typename CONFIG::ABLATION_SPEC;
        ts.env_parameters_base = parameters::environment<T, TI, ABLATION_SPEC>::parameters;
        ts.env_parameters_base_eval = parameters::environment<T, TI, config::template ABLATION_SPEC_EVAL<ABLATION_SPEC>>::parameters;
        _init::load_config(ts);

        for (auto& env : ts.envs) {
            env.parameters = ts.env_parameters_base;
        }
        ts.env_eval.parameters = ts.env_parameters_base_eval;
        TI effective_seed = CONFIG::BASE_SEED + seed;
        ts.run_name = helpers::run_name<ABLATION_SPEC, CONFIG>(effective_seed);
        rlt::construct(ts.device, ts.device.logger, std::string("logs"), ts.run_name);

        rlt::set_step(ts.device, ts.device.logger, 0);
        rlt::add_scalar(ts.device, ts.device.logger, "loop/seed", effective_seed);
        rlt::rl::algorithms::td3::loop::init(ts, effective_seed);
        ts.off_policy_runner.parameters = CONFIG::off_policy_runner_parameters;

        for(typename CONFIG::ENVIRONMENT& env: ts.validation_envs){
            env.parameters = ts.env_parameters_base;
        }
        rlt::malloc(ts.device, ts.validation_actor_buffers);
        rlt::init(ts.device, ts.task, ts.validation_envs, ts.rng_validation);

        // info

        std::cout << "Environment Info: \n";
        std::cout << "\t" << "Observation dim: " << CONFIG::ENVIRONMENT::OBSERVATION_DIM << std::endl;
        std::cout << "\t" << "Observation dim privileged: " << CONFIG::ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED << std::endl;
        std::cout << "\t" << "Action dim: " << CONFIG::ENVIRONMENT::ACTION_DIM << std::endl;
    }

    template <typename CONFIG>
    void step(TrainingState<CONFIG>& ts){
        using TI = typename CONFIG::TI;
        using T = typename CONFIG::T;
        if(ts.step % 10000 == 0){
            std::cout << "Step: " << ts.step << std::endl;
        }
        steps::logger(ts);
        if constexpr(!CONFIG::BENCHMARK) {
            steps::log_reward(ts);
        }
        steps::checkpoint(ts);
        if constexpr(!CONFIG::BENCHMARK){
            steps::validation(ts);
        }
        steps::curriculum(ts);
//        steps::critic_reset(ts);
        rlt::rl::algorithms::td3::loop::step(ts);
        steps::trajectory_collection(ts);
    }
    template <typename CONFIG>
    void destroy(TrainingState<CONFIG>& ts){
        {
            rlt::reset(ts.device, ts.task, ts.rng_validation);
            bool completed = false;
            while(!completed){
                completed = rlt::step(ts.device, ts.task, ts.actor_critic.actor, ts.validation_actor_buffers, ts.rng_validation);
            }
            auto results = rlt::analyse_json(ts.device, ts.task, typename TrainingState<CONFIG>::SPEC::METRICS{});
            std::filesystem::path results_directory = std::string("results/") + ts.run_name;
            try {
                std::filesystem::create_directories(results_directory);
            }
            catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Error creating directories: " << e.what() << '\n';
            }
            {
                std::ofstream file;
                file.open(results_directory / "validation.json");
                file << results.dump(4);
                file.close();
            }
#ifdef LEARNING_TO_FLY_HYPERPARAMETER_OPTIMIZATION
            {
                std::ofstream file;
                file.open(ts.results_path);
                file << results.dump(4);
                file.close();
            }
#endif
        }
        rlt::rl::algorithms::td3::loop::destroy(ts);
        rlt::destroy(ts.device, ts.task);
        rlt::free(ts.device, ts.validation_actor_buffers);
    }
}
