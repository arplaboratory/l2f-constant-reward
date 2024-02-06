import l2f


device = l2f.Device()
rng = l2f.RNG()
env = l2f.Environment()
l2f.init(device, env)
state, next_state = [l2f.State() for _ in range(2)]
action = l2f.Action()
observation, next_observation = [l2f.Observation() for _ in range(2)]

l2f.sample_initial_state(device, env, state, rng)

action_value = 0.5
action.motor_command[0] = action_value
action.motor_command[1] = action_value
action.motor_command[2] = action_value
action.motor_command[3] = action_value
l2f.step(device, env, state, action, next_state, rng)
l2f.observe(device, env, state, observation, rng)
l2f.observe(device, env, next_state, next_observation, rng)

print("     observation: ", observation.observation)
print("Next observation: ", next_observation.observation)