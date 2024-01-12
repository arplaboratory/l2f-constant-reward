namespace learning_to_fly {
    namespace steps {
        template <typename CONFIG>
        void curriculum(TrainingState<CONFIG>& ts){
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            if constexpr(CONFIG::ABLATION_SPEC::ENABLE_CURRICULUM == true) {
                if(ts.step != 0 && ts.step % 100000 == 0 && ts.step != (CONFIG::STEP_LIMIT - 1)){
                    rlt::add_scalar(ts.device, ts.device.logger, "td3/gamma", ts.actor_critic.gamma);
                    rlt::add_scalar(ts.device, ts.device.logger, "td3/target_next_action_noise_std", ts.actor_critic.target_next_action_noise_std);
                    rlt::add_scalar(ts.device, ts.device.logger, "td3/target_next_action_noise_clip", ts.actor_critic.target_next_action_noise_clip);
                    rlt::add_scalar(ts.device, ts.device.logger, "off_policy_runner/exploration_noise", ts.off_policy_runner.parameters.exploration_noise);


                    for(auto& env : ts.off_policy_runner.envs){
                        {
                            T position_weight = env.parameters.mdp.reward.position;
                            position_weight *= ts.curriculum.position.factor;
                            T position_weight_limit = ts.curriculum.position.limit;
                            position_weight = position_weight > position_weight_limit ? position_weight_limit : position_weight;
                            env.parameters.mdp.reward.position = position_weight;
                            rlt::add_scalar(ts.device, ts.device.logger, "reward_function/position_weight", position_weight);
                        }
                        {
                            T orientation_weight = env.parameters.mdp.reward.orientation;
                            orientation_weight *= ts.curriculum.orientation.factor;
                            T orientation_weight_limit = ts.curriculum.orientation.limit;
                            orientation_weight = orientation_weight > orientation_weight_limit ? orientation_weight_limit : orientation_weight;
                            env.parameters.mdp.reward.orientation = orientation_weight;
                            rlt::add_scalar(ts.device, ts.device.logger, "reward_function/orientation_weight", orientation_weight);
                        }
                        {
                            T linear_velocity_weight = env.parameters.mdp.reward.linear_velocity;
                            linear_velocity_weight *= ts.curriculum.linear_velocity.factor;
                            T linear_velocity_weight_limit = ts.curriculum.linear_velocity.limit;
                            linear_velocity_weight = linear_velocity_weight > linear_velocity_weight_limit ? linear_velocity_weight_limit : linear_velocity_weight;
                            env.parameters.mdp.reward.linear_velocity = linear_velocity_weight;
                            rlt::add_scalar(ts.device, ts.device.logger, "reward_function/linear_velocity_weight", linear_velocity_weight);
                        }
                        {
                            T linear_acceleration_weight = env.parameters.mdp.reward.linear_acceleration;
                            linear_acceleration_weight *= ts.curriculum.linear_acceleration.factor;
                            T linear_acceleration_weight_limit = ts.curriculum.linear_acceleration.limit;
                            linear_acceleration_weight = linear_acceleration_weight > linear_acceleration_weight_limit ? linear_acceleration_weight_limit : linear_acceleration_weight;
                            env.parameters.mdp.reward.linear_acceleration = linear_acceleration_weight;
                            rlt::add_scalar(ts.device, ts.device.logger, "reward_function/linear_acceleration_weight", linear_acceleration_weight);
                        }
                        {
                            T angular_acceleration_weight = env.parameters.mdp.reward.angular_acceleration;
                            angular_acceleration_weight *= ts.curriculum.angular_acceleration.factor;
                            T angular_acceleration_weight_limit = ts.curriculum.angular_acceleration.limit;
                            angular_acceleration_weight = angular_acceleration_weight > angular_acceleration_weight_limit ? angular_acceleration_weight_limit : angular_acceleration_weight;
                            env.parameters.mdp.reward.angular_acceleration = angular_acceleration_weight;
                            rlt::add_scalar(ts.device, ts.device.logger, "reward_function/angular_acceleration_weight", angular_acceleration_weight);
                        }
                        {
                            T action_weight = env.parameters.mdp.reward.action;
                            action_weight *= ts.curriculum.action.factor;
                            T action_weight_limit = ts.curriculum.action.limit;
                            action_weight = action_weight > action_weight_limit ? action_weight_limit : action_weight;
                            env.parameters.mdp.reward.action = action_weight;
                            rlt::add_scalar(ts.device, ts.device.logger, "reward_function/action_weight", action_weight);
                        }
                        {
                            T init_max_position = env.parameters.mdp.init.max_position;
                            init_max_position *= ts.curriculum.init_max_position.factor;
                            T init_max_position_limit = ts.curriculum.init_max_position.limit;
                            init_max_position = init_max_position > init_max_position_limit ? init_max_position_limit : init_max_position;
                            env.parameters.mdp.init.max_position = init_max_position;
                            rlt::add_scalar(ts.device, ts.device.logger, "reward_function/init_max_position", init_max_position);
                        }
                        {
                            T init_guidance = env.parameters.mdp.init.guidance;
                            init_guidance *= ts.curriculum.init_guidance.factor;
                            T init_guidance_limit = ts.curriculum.init_guidance.limit;
                            init_guidance = init_guidance < init_guidance_limit ? init_guidance_limit : init_guidance;
                            env.parameters.mdp.init.guidance = init_guidance;
                            rlt::add_scalar(ts.device, ts.device.logger, "reward_function/init_guidance", init_guidance);
                        }

                    }
                    if constexpr(CONFIG::ABLATION_SPEC::RECALCULATE_REWARDS == true){
                        auto start = std::chrono::high_resolution_clock::now();
                        rlt::recalculate_rewards(ts.device, ts.off_policy_runner.replay_buffers[0], ts.off_policy_runner.envs[0], ts.rng);
                        auto end = std::chrono::high_resolution_clock::now();
//                        std::cout << "recalculate_rewards: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
                    }
                }
            }
            if(CONFIG::ABLATION_SPEC::EXPLORATION_NOISE_DECAY == true){
                if(ts.step % 100000 == 0 && ts.step >= 000000){
                    ts.actor_critic.gamma *= ts.curriculum.gamma.factor;
                    ts.actor_critic.gamma = rlt::math::min(ts.device.math, ts.actor_critic.gamma, ts.curriculum.gamma.limit);
                    ts.off_policy_runner.parameters.exploration_noise *= ts.curriculum.exploration_noise.factor;
                    ts.off_policy_runner.parameters.exploration_noise = rlt::math::max(ts.device.math, ts.off_policy_runner.parameters.exploration_noise, ts.curriculum.exploration_noise.limit);
                    ts.actor_critic.target_next_action_noise_std *= ts.curriculum.target_next_action_noise_std.factor;
                    ts.actor_critic.target_next_action_noise_std = rlt::math::max(ts.device.math, ts.actor_critic.target_next_action_noise_std, ts.curriculum.target_next_action_noise_std.limit);
                    ts.actor_critic.target_next_action_noise_clip *= ts.curriculum.target_next_action_noise_clip.factor;
                    ts.actor_critic.target_next_action_noise_clip = rlt::math::max(ts.device.math, ts.actor_critic.target_next_action_noise_clip, ts.curriculum.target_next_action_noise_clip.limit);
                }
            }
        }
    }
}

