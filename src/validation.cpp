#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <learning_to_fly/simulator/operations_cpu.h>

#include "config/config.h"

#include <rl_tools/containers/persist.h>
#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

#include <highfive/H5Group.hpp>

#include <rl_tools/rl/utils/validation_analysis.h>

#include <chrono>
#include <CLI/CLI.hpp>



struct ABLATION_SPEC: learning_to_fly::config::DEFAULT_ABLATION_SPEC{
    static constexpr bool ENABLE_CURRICULUM = true;
    static constexpr bool USE_INITIAL_REWARD_FUNCTION = true;
    static constexpr bool EXPLORATION_NOISE_DECAY = false;
    static constexpr bool DISTURBANCE = false;
    static constexpr bool OBSERVATION_NOISE = false;
    static constexpr bool DOMAIN_RANDOMIZATION = false;
};


namespace fs = std::filesystem;

fs::path removeFirstComponent(const fs::path& path) {
    fs::path result;
    auto iter = path.begin();
    if (iter != path.end()) {
        ++iter;
    }
    for (; iter != path.end(); ++iter) {
        result /= *iter;
    }

    return result;
}

int main(int argc, char** argv){
    using namespace learning_to_fly::config;

    using CONFIG = learning_to_fly::config::Config<ABLATION_SPEC>;
    using T = typename CONFIG::T;
    using TI = typename CONFIG::TI;

    rlt::devices::DefaultCPU device;
    auto rng = rlt::random::default_engine(device.random, 0);
    CONFIG::ACTOR_CHECKPOINT_TYPE actor;
    rlt::malloc(device, actor);
    constexpr TI N_VALIDATION_EPISODES = 100;
    using VALIDATION_SPEC = rlt::rl::utils::validation::Specification<T, TI, CONFIG::ENVIRONMENT>;
    using TASK_SPEC = rlt::rl::utils::validation::TaskSpecification<VALIDATION_SPEC, N_VALIDATION_EPISODES, CONFIG::VALIDATION_MAX_EPISODE_LENGTH>;
    rlt::rl::utils::validation::Task<TASK_SPEC> task;
    typename CONFIG::ACTOR_TYPE::template Buffer<N_VALIDATION_EPISODES> validation_actor_buffers;
    typename CONFIG::ENVIRONMENT validation_envs[N_VALIDATION_EPISODES];
    auto base_parameters_eval = parameters::environment<T, TI, learning_to_fly::config::template ABLATION_SPEC_EVAL<ABLATION_SPEC>>::parameters;
    for(auto& env: validation_envs){
        env.parameters = base_parameters_eval;
    }
    rlt::malloc(device, validation_actor_buffers);
    rlt::init(device, task, validation_envs, rng);

    std::filesystem::path checkpoints_300000_dir = "checkpoints_hpc_300000";
    std::filesystem::path validation_300000_dir = "validation_hpc_300000";
    std::filesystem::path checkpoints_3000000_dir = "checkpoints_hpc_3000000";
    std::filesystem::path validation_3000000_dir = "validation_hpc_3000000";
    std::filesystem::path checkpoints_dir = checkpoints_300000_dir;
    std::filesystem::path validation_dir = validation_300000_dir;
    // for each subdirectory in checkpoints_300000_dir
    for(const auto & entry : std::filesystem::directory_iterator(checkpoints_dir)) {
        for(const auto & subentry : std::filesystem::directory_iterator(entry.path())) {
            std::cout << "\t" << subentry.path() << std::endl;
            std::filesystem::path one_checkpoint = subentry.path();
            {
                auto actor_file = HighFive::File(one_checkpoint.string(), HighFive::File::ReadOnly);
                try{
                    rlt::load(device, actor, actor_file.getGroup("actor"));
                }
                catch(...){
                    std::cout << "Error loading checkpoint: " << one_checkpoint << std::endl;
                    break;
                }
                rlt::reset(device, task, rng);
                bool completed = false;
                while(!completed){
                    completed = rlt::step(device, task, actor, validation_actor_buffers, rng);
                }
                auto analysis_json = rlt::analyse_json(device, task, typename CONFIG::METRICS{});

                // make corresponding path (to the checkpoint) in validation_300000_dir
                std::filesystem::path validation_dir_current = validation_dir / removeFirstComponent(one_checkpoint.parent_path());
                if(!std::filesystem::exists(validation_dir_current)){
                    std::filesystem::create_directory(validation_dir_current);
                }
                std::filesystem::path validation_result_path = validation_dir_current / (one_checkpoint.filename().stem().string() + ".json");
                std::cout << "Writing validation result to: " << validation_result_path << std::endl;
                std::ofstream validation_result_file(validation_result_path);
                validation_result_file << analysis_json.dump(4) << std::endl;
                validation_result_file.close();
            }
        }
    }


    return 0;
}
