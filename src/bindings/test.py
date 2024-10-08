import l2f
import json
from websockets.sync.client import connect
from collections import deque
import time

import numpy as np

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    
    xx2 = 2 * x * x
    yy2 = 2 * y * y
    zz2 = 2 * z * z
    xy2 = 2 * x * y
    xz2 = 2 * x * z
    yz2 = 2 * y * z
    wx2 = 2 * w * x
    wy2 = 2 * w * y
    wz2 = 2 * w * z
    
    R = np.array([[1 - yy2 - zz2, xy2 - wz2, xz2 + wy2],
                  [xy2 + wz2, 1 - xx2 - zz2, yz2 - wx2],
                  [xz2 - wy2, yz2 + wx2, 1 - xx2 - yy2]])
    
    return R


with open('../../parameters/output/crazyflie.json', 'r') as f:
    parameters_json = json.load(f)

parameters_json["mdp"]["reward"]["type"] = "absolute"
# parameters_json["dynamics"]["rotor_torque_constant"] *= 1.0
parameters_json["integration"]["dt"] /= 1
parameters_json["dynamics"]["motor_time_constant"] /= 1

device = l2f.Device()
rng = l2f.RNG()
env = l2f.Environment()
l2f.init(device, env)
l2f.load_config(device, env, json.dumps(parameters_json))
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


model = None
with open('ui_model.json', 'r') as f:
    model = json.load(f)



def add_drone_message(namespace, id):
    add_drone_data = {
        "display_options": {
            "displayActions": True,
            "displayGlobalCoordinateSystem": True,
            "displayIMUCoordinateSystem": True
        },
        "id": id,
        "origin": [
            0,
            0,
            0
        ],
        "model": model
    }
    return {
        "channel": "addDrone",
        "namespace": namespace,
        "data": add_drone_data
    }
def set_state_message(namespace, id, position, orientation, action=None):
    R = quaternion_to_rotation_matrix(orientation)
    return {
        "channel": "setDroneState",
        "namespace": namespace,
        "data": {
            "id": id,
            "data": {
                "pose": {
                    "position": position,
                    "orientation": R.tolist()
                },
                "rotor_states": [
                    {"power": 0},
                    {"power": 0},
                    {"power": 0},
                    {"power": 0}
                ] if action is None else [
                    {"power": action[0]},
                    {"power": action[1]},
                    {"power": action[2]},
                    {"power": action[3]}
                ]
            }
        }
    }





import torch
import h5py


class ActorLearningToFly:
    checkpoint = h5py.File("../../checkpoints/multirotor_td3/2024_02_19_17_17_22_d+o+a+r+h+c+f+w+e+_000/actor_000000001000000.h5", "r")

    checkpoint["actor"]["layers"]["0"]["weights"]["parameters"][:]

    W_in = checkpoint["actor"]["layers"]["0"]["weights"]["parameters"][:]
    b_in = checkpoint["actor"]["layers"]["0"]["biases"]["parameters"][:]
    act_in = np.tanh

    W_1 = checkpoint["actor"]["layers"]["1"]["weights"]["parameters"][:]
    b_1 = checkpoint["actor"]["layers"]["1"]["biases"]["parameters"][:]
    act_1 = np.tanh

    W_out = checkpoint["actor"]["layers"]["2"]["weights"]["parameters"][:]
    b_out = checkpoint["actor"]["layers"]["2"]["biases"]["parameters"][:]
    act_out = np.tanh


    layers = [
        {"W": W_in, "b": b_in, "act": act_in},
        {"W": W_1, "b": b_1, "act": act_1},
        {"W": W_out, "b": b_out, "act": act_out}
    ]

    def forward(self, x):
        x = x.detach().numpy()
        for layer in self.layers:
            x = layer["act"](x @ layer["W"].T + layer["b"])
        return torch.from_numpy(x)[0]

class ActorSimulationOptimization:
    checkpoint = torch.load("/home/jonas/phd/projects/learning_to_fly/ral_rebuttal/checkpoint/torch_save/model.pt")
    observation_mean = checkpoint["obs_oms.mean"]
    observation_std = checkpoint["obs_oms.std"]

    W_in = checkpoint["pi.net.0.weight"]
    b_in = checkpoint["pi.net.0.bias"]
    act_in = torch.nn.ReLU()

    W_1 = checkpoint["pi.net.2.weight"]
    b_1 = checkpoint["pi.net.2.bias"]
    act_1 = torch.nn.ReLU()

    W_out = checkpoint["pi.net.4.weight"]
    b_out = checkpoint["pi.net.4.bias"]
    act_out = torch.nn.Identity()


    layers = [
        {"W": W_in, "b": b_in, "act": act_in},
        {"W": W_1, "b": b_1, "act": act_1},
        {"W": W_out, "b": b_out, "act": act_out}
    ]

    def forward(self, x):
        x = (x - self.observation_mean) / self.observation_std
        for layer in self.layers:
            x = layer["act"](x @ layer["W"].T + layer["b"])
        return x
    
    def write_hdf5(self, filename):
        with h5py.File(filename, "w") as f:
            actor = f.create_group("actor")
            observation_distribution = actor.create_group("observation_distribution")
            observation_distribution.create_dataset("mean", data=self.observation_mean.reshape(1, -1))
            observation_distribution.create_dataset("std", data=self.observation_std.reshape(1, -1))
            layers = actor.create_group("layers")
            for i, layer in enumerate(self.layers):
                layer_group = layers.create_group(f"{i}")
                weights = layer_group.create_group("weights")
                weights.create_dataset("parameters", data=layer["W"])
                biases = layer_group.create_group("biases")
                biases.create_dataset("parameters", data=layer["b"].reshape(1, -1))

class ActorSim2MultiReal:
    checkpoint = torch.load("best_000032303_33078272_reward_3.374.pth", map_location=torch.device('cpu'))

    model = checkpoint["model"]
    model.keys()

    W_in = model["actor_encoder.self_encoder.0.weight"]
    b_in = model["actor_encoder.self_encoder.0.bias"]
    act_in = torch.nn.Tanh()

    W_1 = model["actor_encoder.self_encoder.2.weight"]
    b_1 = model["actor_encoder.self_encoder.2.bias"]
    act_1 = torch.nn.Tanh()

    W_2 = model["actor_encoder.feed_forward.0.weight"]
    b_2 = model["actor_encoder.feed_forward.0.bias"]
    act_2 = torch.nn.Tanh()

    W_out = model["action_parameterization.distribution_linear.weight"]
    b_out = model["action_parameterization.distribution_linear.bias"]
    act_out = torch.nn.Tanh()

    layers = [
        {"W": W_in, "b": b_in, "act": act_in},
        {"W": W_1, "b": b_1, "act": act_1},
        {"W": W_2, "b": b_2, "act": act_2},
        {"W": W_out, "b": b_out, "act": act_out}
    ]

    def forward(self, x):
        for layer in self.layers:
            x = layer["act"](x @ layer["W"].T + layer["b"])
        return x

# baseline = "sim2multireal"
baseline = "simulation_optimization"
# baseline = "LearningToFly"
observation_history_length = 2
sim_opt_target_position = np.array([0, 0, 1.0])

actor_sim2multi_real = ActorSim2MultiReal()
actor_simulation_optimization = ActorSimulationOptimization()
actor_simulation_optimization.write_hdf5("actor_simulation_optimization.h5")
actor_learning_to_fly = ActorLearningToFly()
actor = actor_sim2multi_real if baseline == "sim2multireal" else (actor_simulation_optimization if baseline == "simulation_optimization" else actor_learning_to_fly)

def thrust2rpm(thrust):
    return (((thrust + 1)/2) ** 0.5) * 2 - 1

with connect("ws://localhost:8000/backend") as websocket:
    message = json.loads(websocket.recv())
    namespace = message["data"]["namespace"]
    id = "default"
    print(f"received: {message}")
    websocket.send(json.dumps(add_drone_message(namespace, id)))
    while True:
        # l2f.sample_initial_state(device, env, state, rng)
        l2f.initial_state(device, env, state)
        action_history = deque([torch.zeros(4) for _ in range(observation_history_length)], maxlen=observation_history_length)
        observation_history = None
        for step_i in range(1000):
            l2f.observe(device, env, state, observation, rng)

            position_real = state.position
            orientation_real = state.orientation
            position = observation.observation[:3]
            orientation_quaternion = state.orientation #observation.observation[3:3+4]
            if orientation_quaternion[0] < 0:
                orientation_quaternion = -np.array(orientation_quaternion) # the policy does not seem to like quaternions with a negative real part
            orientation_matrix = quaternion_to_rotation_matrix(orientation_quaternion)
            linear_velocity = observation.observation[3+9:3+9+3]
            angular_velocity = observation.observation[3+9+3:3+9+3+3]
            if baseline == "sim2multireal":
                flat_observation = torch.Tensor(np.array([*position, *linear_velocity, *orientation_matrix.ravel(), *angular_velocity]))
            elif baseline == "simulation_optimization":
                quaternion_xyzw = np.array([*orientation_quaternion[1:], orientation_quaternion[0]])
                flat_observation_now = torch.Tensor(np.array([*(position + sim_opt_target_position), *quaternion_xyzw, *linear_velocity, *angular_velocity]))
                if observation_history is None:
                    observation_history = deque([flat_observation_now for _ in range(observation_history_length)], maxlen=observation_history_length)
                else:
                    observation_history.append(flat_observation_now)
                flat_observation = torch.concat([torch.concat((fo, fa)) for fo, fa in zip(observation_history, action_history)])
            elif baseline == "LearningToFly":
                flat_observation = torch.Tensor(np.array(observation.observation).copy())
            flat_action = actor.forward(flat_observation).detach().numpy()
            action_history.append(torch.from_numpy(flat_action))
            flat_action = np.clip(flat_action, -1, 1)
            action_factor = 1
            flat_action = (flat_action + 1) / 2 * action_factor * 2 - action_factor
            print(f"flat_action: {flat_action}")
            # action.motor_command = [thrust2rpm(t) for t in flat_action]
            action.motor_command = [t for t in flat_action]
            websocket.send(json.dumps(set_state_message(namespace, id, position_real, orientation_real, action=action.motor_command)))
            dt = l2f.step(device, env, state, action, next_state, rng)
            state = next_state
            next_state = l2f.State()

            time.sleep(dt * 1.0)

print("Done")