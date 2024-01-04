#ifdef RL_TOOLS_ENABLE_JSON
#include <nlohmann/json.hpp>
#endif

#include <filesystem>
#include <fstream>

namespace learning_to_fly{
    namespace _init{
        template <typename T_CONFIG>
        void load_config(TrainingState<T_CONFIG>& ts){
            std::filesystem::path parameters_path;
#ifdef LEARNING_TO_FLY_HYPERPARAMETER_OPTIMIZATION
            parameters_path = ts.parameters_path;
#else
            parameters_path = std::filesystem::path("parameters") / "default.json";
#endif
            std::ifstream parameters_file(parameters_path);
            if(!parameters_file.is_open()) {
                std::cout << "Could not open parameters file: " << parameters_path << "\n";
                std::abort();
            }

#ifdef RL_TOOLS_ENABLE_JSON
            nlohmann::json parameters_json;
            parameters_file >> parameters_json;
            if(parameters_json.contains("mdp")){
                if(parameters_json["mdp"].contains("gamma")){
                    ts.actor_critic.gamma = parameters_json["mdp"]["gamma"];
                }
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
                    if(reward_json.contains("calculate_action_baseline")){
                        reward_params.calculate_action_baseline = reward_json["calculate_action_baseline"];
                        reward_params_eval.calculate_action_baseline = reward_json["calculate_action_baseline"];
                    }
                    if(reward_json.contains("action")){
                        reward_params.action = reward_json["action"];
                        reward_params_eval.action = reward_json["action"];
                    }
                }
                if(mdp_json.contains("init")) {
                    auto init_json = mdp_json["init"];
                    auto& init_params = ts.env_parameters_base.mdp.init;
                    if(init_json.contains("max_position")){
                        init_params.max_position = init_json["max_position"];
                    }
                    if(init_json.contains("max_orientation")){
                        init_params.max_angle = init_json["max_angle"];
                    }
                    if(init_json.contains("max_linear_velocity")){
                        init_params.max_linear_velocity = init_json["max_linear_velocity"];
                    }
                    if(init_json.contains("max_angular_velocity")){
                        init_params.max_angular_velocity = init_json["max_angular_velocity"];
                    }
                }
                if(mdp_json.contains("termination")) {
                    auto termination_json = mdp_json["termination"];
                    auto& termination_params = ts.env_parameters_base.mdp.termination;
                    if(termination_json.contains("enabled")){
                        termination_params.enabled = termination_json["enabled"];
                    }
                    if(termination_json.contains("position_threshold")){
                        termination_params.position_threshold = termination_json["position_threshold"];
                    }
                    if(termination_json.contains("linear_velocity_threshold")){
                        termination_params.linear_velocity_threshold = termination_json["linear_velocity_threshold"];
                    }
                    if(termination_json.contains("angular_velocity_threshold")){
                        termination_params.angular_velocity_threshold = termination_json["angular_velocity_threshold"];
                    }
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