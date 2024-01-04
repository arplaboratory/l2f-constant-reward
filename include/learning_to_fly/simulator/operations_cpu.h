#ifndef LEARNING_TO_FLY_SIMULATOR_OPERATIONS_CPU_H
#define LEARNING_TO_FLY_SIMULATOR_OPERATIONS_CPU_H
#include "operations_generic.h"

#include <random>
#ifdef RL_TOOLS_ENABLE_JSON
#include <nlohmann/json.hpp>
#endif
namespace rl_tools{
#ifdef RL_TOOLS_ENABLE_JSON
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::Dynamics& parameters, nlohmann::json config){
        if(config.contains("dynamics")){
            if(config["dynamics"].contains("n_rotors")){
                typename SPEC::TI num_rotors = config["dynamics"]["n_rotors"];
                utils::assert_exit(device, num_rotors == SPEC::N, "Rotor number in config file does not match number of rotors in parameters file");
                if(config["dynamics"].contains("rotor_positions")){
                    auto rotor_positions = config["dynamics"]["rotor_positions"];
                    utils::assert_exit(device, rotor_positions.size() == SPEC::N, "Rotor number in config file does not match number of rotors in parameters file");
                }
            }
            else{
                std::cout << "Config file does not contain rotor number, skipping..." << std::endl;
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>& parameters, nlohmann::json config){
        load_config<DEV_SPEC, SPEC>(device, parameters.dynamics, config);
    }
#endif
}

#endif