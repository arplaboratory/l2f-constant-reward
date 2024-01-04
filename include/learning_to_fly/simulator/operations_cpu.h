#ifndef LEARNING_TO_FLY_SIMULATOR_OPERATIONS_CPU_H
#define LEARNING_TO_FLY_SIMULATOR_OPERATIONS_CPU_H
#include "operations_generic.h"

#include <random>
#ifdef RL_TOOLS_ENABLE_JSON
#include <nlohmann/json.hpp>
#endif
#include <string>
namespace rl_tools{
#ifdef RL_TOOLS_ENABLE_JSON
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::Dynamics& parameters, nlohmann::json config){
        if(config.contains("dynamics")){
            if(config["dynamics"].contains("model")){
                std::string config_model = config["dynamics"]["model"];
                utils::assert_exit(device, config_model == rl_tools::rl::environments::multirotor::parameters::registry_name<SPEC>, "Model in config file does not match model in parameters file");
            }
            if(config["dynamics"].contains("n_rotors")){
                typename SPEC::TI num_rotors = config["dynamics"]["n_rotors"];
                utils::assert_exit(device, num_rotors == SPEC::N, "Rotor number in config file does not match number of rotors in parameters file");
                if(config["dynamics"].contains("rotor_positions")){
                    auto rotor_positions = config["dynamics"]["rotor_positions"];
                    utils::assert_exit(device, rotor_positions.size() == SPEC::N, "Rotor number in config file does not match number of rotors in parameters file");
                    for(typename SPEC::TI rotor_i = 0; rotor_i < SPEC::N; rotor_i++){
                        utils::assert_exit(device, rotor_positions[rotor_i].size() == 3, "Positions are 3 dimensional");
                        for(typename SPEC::TI dim_i = 0; dim_i < 3; dim_i++){
                            parameters.rotor_positions[rotor_i][dim_i] = rotor_positions[rotor_i][dim_i];
                        }
                    }
                }
                if(config["dynamics"].contains("rotor_thrust_directions")){
                    auto rotor_thrust_directions = config["dynamics"]["rotor_thrust_directions"];
                    utils::assert_exit(device, rotor_thrust_directions.size() == SPEC::N, "Rotor number in config file does not match number of rotors in parameters file");
                    for(typename SPEC::TI rotor_i = 0; rotor_i < SPEC::N; rotor_i++){
                        utils::assert_exit(device, rotor_thrust_directions[rotor_i].size() == 3, "Positions are 3 dimensional");
                        for(typename SPEC::TI dim_i = 0; dim_i < 3; dim_i++){
                            parameters.rotor_thrust_directions[rotor_i][dim_i] = rotor_thrust_directions[rotor_i][dim_i];
                        }
                    }
                }
                if(config["dynamics"].contains("rotor_torque_directions")){
                    auto rotor_torque_directions = config["dynamics"]["rotor_torque_directions"];
                    utils::assert_exit(device, rotor_torque_directions.size() == SPEC::N, "Rotor number in config file does not match number of rotors in parameters file");
                    for(typename SPEC::TI rotor_i = 0; rotor_i < SPEC::N; rotor_i++){
                        utils::assert_exit(device, rotor_torque_directions[rotor_i].size() == 3, "Positions are 3 dimensional");
                        for(typename SPEC::TI dim_i = 0; dim_i < 3; dim_i++){
                            parameters.rotor_torque_directions[rotor_i][dim_i] = rotor_torque_directions[rotor_i][dim_i];
                        }
                    }
                }
                if(config["dynamics"].contains("rotor_thrust_coefficients")){
                    auto rotor_thrust_coefficients = config["dynamics"]["rotor_thrust_coefficients"];
                    utils::assert_exit(device, rotor_thrust_coefficients.size() == 3, "Please provide three orders (0, 1, 2) of thrust coefficients");
                    for(typename SPEC::TI order_i=0; order_i < 3; order_i++){
                        parameters.rotor_thrust_coefficients[order_i] = rotor_thrust_coefficients[order_i];
                    }
                }
                if(config["dynamics"].contains("rotor_torque_constant")){
                    parameters.rotor_torque_constant = config["dynamics"]["rotor_torque_constant"];
                }
                if(config["dynamics"].contains("mass")){
                    parameters.mass = config["dynamics"]["mass"];
                }
                if(config["dynamics"].contains("gravity")){
                    auto gravity = config["dynamics"]["gravity"];
                    utils::assert_exit(device, gravity.size() == 3, "Gravity is three dimensional");
                    for(typename SPEC::TI dim_i=0; dim_i < 3; dim_i++){
                        parameters.gravity[dim_i] = gravity[dim_i];
                    }
                }
                if(config["dynamics"].contains("J")){
                    auto J = config["dynamics"]["J"];
                    utils::assert_exit(device, J.size() == 3, "The moment of inertia matrix should be 3x3");
                    for(typename SPEC::TI row_i = 0; row_i < 3; row_i++){
                        utils::assert_exit(device, J[row_i].size() == 3, "The moment of inertia matrix should be 3x3");
                        for(typename SPEC::TI col_i = 0; col_i < 3; col_i++){
                            parameters.J[row_i][col_i] = J[row_i][col_i];
                        }
                    }
                }
                if(config["dynamics"].contains("J_inv")){
                    auto J_inv = config["dynamics"]["J_inv"];
                    utils::assert_exit(device, J_inv.size() == 3, "The moment of inertia matrix should be 3x3");
                    for(typename SPEC::TI row_i = 0; row_i < 3; row_i++){
                        utils::assert_exit(device, J_inv[row_i].size() == 3, "The moment of inertia matrix should be 3x3");
                        for(typename SPEC::TI col_i = 0; col_i < 3; col_i++){
                            parameters.J_inv[row_i][col_i] = J_inv[row_i][col_i];
                        }
                    }
                }
                if(config["dynamics"].contains("motor_time_constant")){
                    parameters.motor_time_constant = config["dynamics"]["motor_time_constant"];
                }
                if(config["dynamics"].contains("action_limit")){
                    auto action_limit = config["dynamics"]["action_limit"];
                    if(action_limit.contains("upper_bound")){
                        parameters.action_limit.max = action_limit["upper_bound"];
                    }
                    if(action_limit.contains("lower_bound")){
                        parameters.action_limit.min = action_limit["lower_bound"];
                    }
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