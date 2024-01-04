def transform(data):
    if(env.parameters.mdp.reward.calculate_action_baseline){
    utils::assert_exit(device, env.current_dynamics.thrust_constants[1] == 0, "linear thrust coefficient not handled yet");
    T hover_thrust = env.current_dynamics.mass * (-1) * env.current_dynamics.gravity[2];
    env.parameters.mdp.reward.action_baseline = math::sqrt(device.math, (hover_thrust / 4 - env.current_dynamics.thrust_constants[0]) / env.current_dynamics.thrust_constants[2]);
    //            env.parameters.mdp.reward.action_baseline *= 0.8;
    }
