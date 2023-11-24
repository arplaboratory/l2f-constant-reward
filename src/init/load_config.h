#ifdef RL_TOOLS_ENABLE_JSON
#include <nlohmann/json.hpp>
#endif

#include <filesystem>
#include <fstream>

namespace learning_to_fly{
    namespace _init{
        template <typename T_CONFIG>
        void load_config(TrainingState<T_CONFIG>& ts){
            std::filesystem::path parameters_path = std::filesystem::path("parameters") / "test.json";
            std::ifstream parameters_file(parameters_path);
            if(parameters_file.is_open()){
#ifdef RL_TOOLS_ENABLE_JSON
                nlohmann::json parameters_json;
                parameters_file >> parameters_json;
                if(parameters_json.contains("mdp")){
                    auto mdp_json = parameters_json["mdp"];
                    if(mdp_json.contains("reward")){
                        auto reward_json = mdp_json["reward"];
                        rlt::utils::assert_exit(ts.device, reward_json.contains("type"), "Parameters file does not contain reward type");
                        rlt::utils::assert_exit(ts.device, reward_json["type"] == "Squared", "Parameters file reward type is not Squared");
                        
                        auto& reward_params = ts.env_parameters_base.mdp.reward;
                        auto& reward_params_eval = ts.env_parameters_base_eval.mdp.reward;

                        if(reward_json.contains("scale")){
                            reward_params.scale = reward_json["scale"];
                            reward_params_eval.scale = reward_json["scale"];
                        }
                        if(reward_json.contains("constant")){
                            reward_params.constant = reward_json["constant"];
                            reward_params_eval.constant = reward_json["constant"];
                        }
                        if(reward_json.contains("termination_penalty")){
                            reward_params.termination_penalty = reward_json["termination_penalty"];
                            reward_params_eval.termination_penalty = reward_json["termination_penalty"];
                        }
                        if(reward_json.contains("position")){
                            reward_params.position = reward_json["position"];
                            reward_params_eval.position = reward_json["position"];
                        }
                        if(reward_json.contains("orientation")){
                            reward_params.orientation = reward_json["orientation"];
                            reward_params_eval.orientation = reward_json["orientation"];
                        }
                        if(reward_json.contains("linear_velocity")){
                            reward_params.linear_velocity = reward_json["linear_velocity"];
                            reward_params_eval.linear_velocity = reward_json["linear_velocity"];
                        }
                        if(reward_json.contains("angular_velocity")){
                            reward_params.angular_velocity = reward_json["angular_velocity"];
                            reward_params_eval.angular_velocity = reward_json["angular_velocity"];
                        }
                        if(reward_json.contains("linear_acceleration")){
                            reward_params.linear_acceleration = reward_json["linear_acceleration"];
                            reward_params_eval.linear_acceleration = reward_json["linear_acceleration"];
                        }
                        if(reward_json.contains("angular_acceleration")){
                            reward_params.angular_acceleration = reward_json["angular_acceleration"];
                            reward_params_eval.angular_acceleration = reward_json["angular_acceleration"];
                        }
                        if(reward_json.contains("action_baseline")){
                            reward_params.action_baseline = reward_json["action_baseline"];
                            reward_params_eval.action_baseline = reward_json["action_baseline"];
                        }
                        if(reward_json.contains("action")){
                            reward_params.action = reward_json["action"];
                            reward_params_eval.action = reward_json["action"];
                        }
                    }
                }

                std::cout << "Gamma: " << parameters_json["gamma"] << "\n";
#endif
            }
        }
    }
}