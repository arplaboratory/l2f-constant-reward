#include <rl_tools/operations/cpu_mux.h>

#include <learning_to_fly/simulator/operations_cpu.h>
#include <learning_to_fly/simulator/ui.h>
#include <rl_tools/nn_models/operations_cpu.h>
#include <rl_tools/nn_models/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include "config/config.h"
//#include "training.h"

#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <highfive/H5File.hpp>
#include <CLI/CLI.hpp>

namespace TEST_DEFINITIONS{
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;
    namespace parameter_set = parameters;
    template <typename BASE_SPEC>
    struct SpecEval: BASE_SPEC{
        static constexpr bool DISTURBANCE = true;
        static constexpr bool OBSERVATION_NOISE = true;
        static constexpr bool ROTOR_DELAY = true;
        static constexpr bool ACTION_HISTORY = BASE_SPEC::ROTOR_DELAY && BASE_SPEC::ACTION_HISTORY;
        static constexpr bool USE_INITIAL_REWARD_FUNCTION = false;
        static constexpr bool INIT_NORMAL = true;
    };
    using EVAL_SPEC = SpecEval<parameters::DefaultAblationSpec>;
    using CONFIG = learning_to_fly::config::Config<EVAL_SPEC>;

    using penv = parameter_set::environment<T, TI, EVAL_SPEC>;
    using ENVIRONMENT = penv::ENVIRONMENT;
    constexpr TI MAX_EPISODE_LENGTH = 500;
    constexpr TI NUM_EPISODES = 100;
}

int main(int argc, char** argv) {
    using namespace TEST_DEFINITIONS;
    CLI::App app;
    std::string arg_run = "", arg_checkpoint = "";
    DEVICE::index_t startup_timeout = 0;
    std::string arg_parameters_path = "parameters/output/crazyflie.json";
    app.add_option("--run", arg_run, "path to the run's directory");
    app.add_option("--checkpoint", arg_checkpoint, "path to the checkpoint");
    app.add_option("--parameters", arg_parameters_path, "parameter file");

    CLI11_PARSE(app, argc, argv);
    DEVICE dev;
    ENVIRONMENT env;
    env.parameters = penv::parameters;
    typename CONFIG::ACTOR_TYPE actor;
    typename CONFIG::ACTOR_TYPE::template Buffer<1> actor_buffer;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    typename ENVIRONMENT::State state, next_state;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    rlt::malloc(dev, env);
    rlt::malloc(dev, actor);
    rlt::malloc(dev, actor_buffer);
    rlt::malloc(dev, action);
    rlt::malloc(dev, observation);

    std::string run = arg_run;
    std::string checkpoint = arg_checkpoint;
    std::filesystem::path actor_run;
    if(run == "" && checkpoint == ""){
        std::filesystem::path actor_checkpoints_dir = std::filesystem::path("checkpoints") / "multirotor_td3";
        std::vector<std::filesystem::path> actor_runs;

        for (const auto& run : std::filesystem::directory_iterator(actor_checkpoints_dir)) {
            if (run.is_directory()) {
                actor_runs.push_back(run.path());
            }
        }
        std::sort(actor_runs.begin(), actor_runs.end());
        actor_run = actor_runs.back();
    }
    else{
        actor_run = run;
    }
    if(checkpoint == ""){
        std::vector<std::filesystem::path> actor_checkpoints;
        for (const auto& checkpoint : std::filesystem::directory_iterator(actor_run)) {
            if (checkpoint.is_regular_file()) {
                if(checkpoint.path().extension() == ".h5" || checkpoint.path().extension() == ".hdf5"){
                    actor_checkpoints.push_back(checkpoint.path());
                }
            }
        }
        std::sort(actor_checkpoints.begin(), actor_checkpoints.end());
        checkpoint = actor_checkpoints.back().string();
    }

    std::cout << "Loading actor from " << checkpoint << std::endl;
    {
        auto data_file = HighFive::File(checkpoint, HighFive::File::ReadOnly);
        rlt::load(dev, actor, data_file.getGroup("actor"));
    }
    if(arg_checkpoint == ""){
        checkpoint = "";
    }
    if(arg_run == ""){
        run = "";
    }

    T reward_acc = 0;
    std::cout << "Loading parameters from: " << arg_parameters_path << std::endl;
    std::ifstream parameters_file(arg_parameters_path);
    if(!parameters_file.is_open()) {
        std::cout << "Could not open parameters file: " << arg_parameters_path << "\n";
        std::abort();
    }
    nlohmann::json parameters_json;
    parameters_file >> parameters_json;
    env.parameters = penv::parameters;
    rlt::load_config(dev, env.parameters, parameters_json);
    nlohmann::json data_episodes;
    for(int episode_i = 0; episode_i < NUM_EPISODES; episode_i++){
        nlohmann::json episode;
        nlohmann::json episode_states;
        nlohmann::json episode_actions;
        rlt::sample_initial_state(dev, env, state, rng);
        for(int step_i = 0; step_i < MAX_EPISODE_LENGTH; step_i++){
            nlohmann::json json_state;
            json_state["position"] = std::vector<T>{state.position[0], state.position[1], state.position[2]};;
            json_state["orientation"] = std::vector<T>{state.orientation[0], state.orientation[1], state.orientation[2], state.orientation[3]};
            json_state["linear_velocity"] = std::vector<T>{state.linear_velocity[0], state.linear_velocity[1], state.linear_velocity[2]};
            json_state["angular_velocity"] = std::vector<T>{state.angular_velocity[0], state.angular_velocity[1], state.angular_velocity[2]};
            episode_states.push_back(json_state);
            rlt::observe(dev, env, state, observation, rng);
            rlt::evaluate(dev, actor, observation, action, actor_buffer);
            rlt::clamp(dev, action, (T)-1, (T)1);
            episode_actions.push_back(std::vector<T>{rlt::get(action, 0, 0), rlt::get(action, 0, 1), rlt::get(action, 0, 2), rlt::get(action, 0, 3)});
            T dt = rlt::step(dev, env, state, action, next_state, rng);
            bool terminated_flag = rlt::terminated(dev, env, state, rng);
            reward_acc += rlt::reward(dev, env, state, action, next_state, rng);
            state = next_state;
            if(terminated_flag){
                break;
            }
        }
        episode["states"] = episode_states;
        episode["actions"] = episode_actions;
        data_episodes.push_back(episode);
    }
    // save data
    nlohmann::json data;
    data["episodes"] = data_episodes;
    std::string data_path = "data/evaluate_actor_headless.json";
    std::ofstream data_file(data_path);
    data_file << data.dump(4);
    data_file.close();
    std::cout << "Saved data to: " << data_path << std::endl;
}

