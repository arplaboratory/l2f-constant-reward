#ifdef RL_TOOLS_ENABLE_JSON
#include <nlohmann/json.hpp>
#endif

#include <filesystem>
#include <fstream>

#include <learning_to_fly/simulator/operations_cpu.h>

namespace learning_to_fly{
    namespace _init{
        template <typename T_CONFIG>
        void load_config(TrainingState<T_CONFIG>& ts){
            std::filesystem::path parameters_path;
            parameters_path = ts.parameters_path;
            std::ifstream parameters_file(parameters_path);
            if(!parameters_file.is_open()) {
                std::cout << "Could not open parameters file: " << parameters_path << "\n";
                std::abort();
            }

#ifdef RL_TOOLS_ENABLE_JSON
            nlohmann::json parameters_json;
            parameters_file >> parameters_json;
            rlt::load_config(ts.device, ts.env_parameters_base, parameters_json);
            rlt::load_config(ts.device, ts.env_parameters_base_eval, parameters_json);
            if(parameters_json.contains("mdp")){
                auto mdp_json = parameters_json["mdp"];
                if(parameters_json["mdp"].contains("gamma")){
                    ts.actor_critic.gamma = mdp_json["gamma"];
                }
                if(parameters_json["mdp"].contains("ignore_termination")){
                    rlt::utils::assert_exit(ts.device, parameters_json["mdp"]["ignore_termination"] == decltype(ts.actor_critic)::SPEC::PARAMETERS::IGNORE_TERMINATION, "ignore termination should match the constexpr value");
                }
                if(mdp_json.contains("curriculum")){
                    auto curriculum_json = mdp_json["curriculum"];

                    if(curriculum_json.contains("position")){
                        ts.curriculum.position.factor = curriculum_json["position"]["factor"];
                        ts.curriculum.position.limit = curriculum_json["position"]["limit"];
                    }
                    if(curriculum_json.contains("orientation")){
                        ts.curriculum.orientation.factor = curriculum_json["orientation"]["factor"];
                        ts.curriculum.orientation.limit = curriculum_json["orientation"]["limit"];
                    }
                    if(curriculum_json.contains("linear_velocity")){
                        ts.curriculum.linear_velocity.factor = curriculum_json["linear_velocity"]["factor"];
                        ts.curriculum.linear_velocity.limit = curriculum_json["linear_velocity"]["limit"];
                    }
                    if(curriculum_json.contains("linear_acceleration")){
                        ts.curriculum.linear_acceleration.factor = curriculum_json["linear_acceleration"]["factor"];
                        ts.curriculum.linear_acceleration.limit = curriculum_json["linear_acceleration"]["limit"];
                    }
                    if(curriculum_json.contains("angular_acceleration")){
                        ts.curriculum.angular_acceleration.factor = curriculum_json["angular_acceleration"]["factor"];
                        ts.curriculum.angular_acceleration.limit = curriculum_json["angular_acceleration"]["limit"];
                    }
                    if(curriculum_json.contains("action")){
                        ts.curriculum.action.factor = curriculum_json["action"]["factor"];
                        ts.curriculum.action.limit = curriculum_json["action"]["limit"];
                    }
                    if(curriculum_json.contains("init_position")){
                        ts.curriculum.init_position.factor = curriculum_json["init_position"]["factor"];
                        ts.curriculum.init_position.limit = curriculum_json["init_position"]["limit"];
                    }
                    if(curriculum_json.contains("init_orientation")){
                        ts.curriculum.init_orientation.factor = curriculum_json["init_orientation"]["factor"];
                        ts.curriculum.init_orientation.limit = curriculum_json["init_orientation"]["limit"];
                    }
                    if(curriculum_json.contains("init_linear_velocity")){
                        ts.curriculum.init_linear_velocity.factor = curriculum_json["init_linear_velocity"]["factor"];
                        ts.curriculum.init_linear_velocity.limit = curriculum_json["init_linear_velocity"]["limit"];
                    }
                    if(curriculum_json.contains("init_angular_velocity")){
                        ts.curriculum.init_angular_velocity.factor = curriculum_json["init_angular_velocity"]["factor"];
                        ts.curriculum.init_angular_velocity.limit = curriculum_json["init_angular_velocity"]["limit"];
                    }
                    if(curriculum_json.contains("termination_position")){
                        ts.curriculum.termination_position.factor = curriculum_json["termination_position"]["factor"];
                        ts.curriculum.termination_position.limit = curriculum_json["termination_position"]["limit"];
                    }
                    if(curriculum_json.contains("termination_linear_velocity")){
                        ts.curriculum.termination_linear_velocity.factor = curriculum_json["termination_linear_velocity"]["factor"];
                        ts.curriculum.termination_linear_velocity.limit = curriculum_json["termination_linear_velocity"]["limit"];
                    }
                    if(curriculum_json.contains("termination_angular_velocity")){
                        ts.curriculum.termination_angular_velocity.factor = curriculum_json["termination_angular_velocity"]["factor"];
                        ts.curriculum.termination_angular_velocity.limit = curriculum_json["termination_angular_velocity"]["limit"];
                    }
                }
            }
            if(parameters_json.contains("off_policy_runner")){
                if(parameters_json["off_policy_runner"].contains("exploration_noise")){
                    ts.off_policy_runner.parameters.exploration_noise = parameters_json["off_policy_runner"]["exploration_noise"];
                }
            }
            if(parameters_json.contains("td3")){
                if(parameters_json["td3"].contains("target_action_noise_std")){
                    ts.actor_critic.target_next_action_noise_std = parameters_json["td3"]["target_action_noise_std"];
                }
                if(parameters_json["td3"].contains("target_action_noise_clip")){
                    ts.actor_critic.target_next_action_noise_clip = parameters_json["td3"]["target_action_noise_clip"];
                }
            }

#endif
        }
    }
}