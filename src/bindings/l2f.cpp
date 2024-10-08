#include <rl_tools/operations/cpu.h>
#include <learning_to_fly/simulator/operations_cpu.h>

#include "../config/parameters.h"

#ifdef RL_TOOLS_ENABLE_JSON
// for loading the config
#include <nlohmann/json.hpp>
#include <learning_to_fly/simulator/operations_cpu.h>
#endif

namespace rlt = rl_tools;

#ifndef TEST
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

using DEVICE = rlt::devices::DefaultCPU;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

struct ABLATION_SPEC: parameters::DefaultAblationSpec{
    static constexpr bool OBSERVATION_NOISE = false;
};

using CONFIG = parameters::environment<T, TI, ABLATION_SPEC>;
using ENVIRONMENT = typename CONFIG::ENVIRONMENT;
using PARAMETERS = typename CONFIG::ENVIRONMENT::PARAMETERS;


void init(DEVICE &device, rlt::rl::environments::Multirotor<typename ENVIRONMENT::SPEC>& env){
    rlt::malloc(device, env);
    rlt::init(device, env);
    env.parameters = CONFIG::parameters;
}

struct State{
    std::array<T, 3> position;
    std::array<T, 4> orientation;
    std::array<T, 3> linear_velocity;
    std::array<T, 3> angular_velocity;
    ENVIRONMENT::State state;
};

void sync(State& state, typename ENVIRONMENT::State& state_real){
    state.position[0] = state_real.position[0];
    state.position[1] = state_real.position[1];
    state.position[2] = state_real.position[2];
    state.orientation[0] = state_real.orientation[0];
    state.orientation[1] = state_real.orientation[1];
    state.orientation[2] = state_real.orientation[2];
    state.orientation[3] = state_real.orientation[3];
    state.linear_velocity[0] = state_real.linear_velocity[0];
    state.linear_velocity[1] = state_real.linear_velocity[1];
    state.linear_velocity[2] = state_real.linear_velocity[2];
    state.angular_velocity[0] = state_real.angular_velocity[0];
    state.angular_velocity[1] = state_real.angular_velocity[1];
    state.angular_velocity[2] = state_real.angular_velocity[2];
}
struct Action{
    std::array<T, ENVIRONMENT::ACTION_DIM> motor_command;
};

struct Observation{
    std::array<T, ENVIRONMENT::OBSERVATION_DIM> observation;
};

#ifdef RL_TOOLS_ENABLE_JSON
void load_config(DEVICE& device, ENVIRONMENT& env, std::string config_string){
    nlohmann::json parameters_json = nlohmann::json::parse(config_string);
    rlt::load_config(device, env.parameters, parameters_json);
}
#endif

T step(DEVICE& device, ENVIRONMENT& env, State& state, Action action, State& next_state, RNG& rng){
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> motor_commands;
    rlt::malloc(device, motor_commands);
    for(TI action_i=0; action_i < 4; action_i++){
        set(motor_commands, 0, action_i, action.motor_command[action_i]);
    }
    T dt = rlt::step(device, env, state.state, motor_commands, next_state.state, rng);
    sync(next_state, next_state.state);
    return dt;
}
void initial_state(DEVICE& device, ENVIRONMENT& env, State& state){
    rlt::initial_state(device, env, state.state);
    sync(state, state.state);
}
void sample_initial_state(DEVICE& device, ENVIRONMENT& env, State& state, RNG& rng){
    rlt::sample_initial_state(device, env, state.state, rng);
    sync(state, state.state);
}
void observe(DEVICE& device, ENVIRONMENT& env, State& state, Observation& observation, RNG& rng){
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_matrix;
    rlt::malloc(device, observation_matrix);
    rlt::observe(device, env, state.state, observation_matrix, rng);
    for(TI observation_i=0; observation_i < ENVIRONMENT::OBSERVATION_DIM; observation_i++){
        observation.observation[observation_i] = get(observation_matrix, 0, observation_i);
    }
}

class Calculator {
public:
    Calculator() = default;
    int add(int a, int b) { return a + b; }
    int subtract(int a, int b) { return a - b; }
};

#ifndef TEST
PYBIND11_MODULE(l2f, m) {
    // Optional: m.doc() = "Documentation string for the module"; // Module documentation
    py::class_<DEVICE>(m, "Device")
        .def(py::init<>());
    py::class_<RNG>(m, "RNG")
        .def(py::init<>());
    py::class_<PARAMETERS>(m, "Parameters")
        .def(py::init<>());
    py::class_<ENVIRONMENT>(m, "Environment")
        .def(py::init<>())
        .def_readwrite("parameters", &ENVIRONMENT::parameters);
    py::class_<State>(m, "State")
        .def(py::init<>())
        .def_readwrite("position", &State::position)
        .def_readwrite("orientation", &State::orientation);
    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("motor_command", &Action::motor_command);
    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        .def_readwrite("observation", &Observation::observation);


    m.def("init", &init, "Init environement");
    m.def("step", &step, "Simulate one step");
    m.def("initial_state", &initial_state, "Reset to default state");
    m.def("sample_initial_state", &sample_initial_state, "Reset to random state");
    m.def("observe", &observe, "Observe state");
#ifdef RL_TOOLS_ENABLE_JSON
    m.def("load_config", &load_config, "Load config");
#endif
}
#else
int main(){
    DEVICE device;
    ENVIRONMENT env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    env.parameters = CONFIG::parameters;

    RNG rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 0);
    ENVIRONMENT::State state, next_state;
    Action action;
    Observation observation, next_observation;
    sample_initial_state(device, env, state, rng);
    for(TI action_i=0; action_i < 4; action_i++){
        action.motor_command[action_i] = 0.5;
    }
    step(device, env, state, action, next_state, rng);
    observe(device, env, state, observation, rng);
    observe(device, env, next_state, next_observation, rng);
    for(TI observation_i=0; observation_i < ENVIRONMENT::OBSERVATION_DIM; observation_i++){
        std::cout << observation.observation[observation_i] << " " << next_observation.observation[observation_i] << std::endl;
    }
    return 0;
}

#endif