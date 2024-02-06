#include <rl_tools/operations/cpu.h>
#include <learning_to_fly/simulator/operations_cpu.h>

#include "../config/parameters.h"

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

using CONFIG = parameters::environment<T, TI, parameters::DefaultAblationSpec>;
using ENVIRONMENT = typename CONFIG::ENVIRONMENT;


void init(DEVICE &device, rlt::rl::environments::Multirotor<typename ENVIRONMENT::SPEC>& env){
    rlt::malloc(device, env);
    rlt::init(device, env);
    env.parameters = CONFIG::parameters;
}

struct Action{
    std::array<T, ENVIRONMENT::ACTION_DIM> motor_command;
};

struct Observation{
    std::array<T, ENVIRONMENT::OBSERVATION_DIM> observation;
};

void step(DEVICE& device, ENVIRONMENT& env, typename ENVIRONMENT::State& state, Action action, typename ENVIRONMENT::State& next_state, RNG& rng){
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> motor_commands;
    rlt::malloc(device, motor_commands);
    for(TI action_i=0; action_i < 4; action_i++){
        set(motor_commands, 0, action_i, action.motor_command[action_i]);
    }
    rlt::step(device, env, state, motor_commands, next_state, rng);
}
void initial_state(DEVICE& device, ENVIRONMENT& env, typename ENVIRONMENT::State& state){
    rlt::initial_state(device, env, state);
}
void sample_initial_state(DEVICE& device, ENVIRONMENT& env, typename ENVIRONMENT::State& state, RNG& rng){
    rlt::sample_initial_state(device, env, state, rng);
}
void observe(DEVICE& device, ENVIRONMENT& env, typename ENVIRONMENT::State& state, Observation& observation, RNG& rng){
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_matrix;
    rlt::malloc(device, observation_matrix);
    rlt::observe(device, env, state, observation_matrix, rng);
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
    py::class_<ENVIRONMENT::PARAMETERS>(m, "Parameters")
        .def(py::init<>());
    py::class_<ENVIRONMENT>(m, "Environment")
        .def(py::init<>())
        .def_readwrite("parameters", &ENVIRONMENT::parameters);
    py::class_<ENVIRONMENT::State>(m, "State")
        .def(py::init<>());
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