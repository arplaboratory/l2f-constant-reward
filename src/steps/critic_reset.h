namespace learning_to_fly {
    namespace steps {
        template <typename CONFIG>
        void critic_reset(TrainingState<CONFIG>& ts){
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            if(ts.step == 500000) {
                std::cout << "Resetting critic" << std::endl;
                rlt::init_weights(ts.device, ts.actor_critic.critic_1, ts.rng);
                rlt::init_weights(ts.device, ts.actor_critic.critic_2, ts.rng);
                rlt::reset_optimizer_state(ts.device, ts.actor_critic.critic_optimizers[0], ts.actor_critic.critic_1);
                rlt::reset_optimizer_state(ts.device, ts.actor_critic.critic_optimizers[1], ts.actor_critic.critic_2);
                for(TI step_i=0; step_i < 100; step_i++){
                    for(TI critic_i=0; critic_i < 2; critic_i++){
                        target_action_noise(ts.device, ts.actor_critic, ts.critic_training_buffers.target_next_action_noise, ts.rng);
                        gather_batch(ts.device, ts.off_policy_runner, ts.critic_batch, ts.rng);
                        rlt::train_critic(ts.device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch, ts.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers);
                    }
                }
            }
//            if(ts.step == 600000){
//                std::cout << "Resetting actor" << std::endl;
//                rlt::init_weights(ts.device, ts.actor_critic.actor, ts.rng);
//                rlt::reset_optimizer_state(ts.device, ts.actor_critic.actor_optimizer, ts.actor_critic.actor);
//            }
        }
    }
}

