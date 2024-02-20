#include <rl_tools/operations/cpu.h>

#include "actor_simulation_optimization.h"

#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;
using T = actor::MODEL::T;

int main(int argc, char** argv){
    DEVICE device;

    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, actor::MODEL::INPUT_DIM>> observation;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, actor::MODEL::OUTPUT_DIM>> action;
    typename actor::MODEL::Buffer<1> buffer;

    rlt::malloc(device, observation);
    rlt::malloc(device, action);
    rlt::malloc(device, buffer);

    rlt::set_all(device, observation, 0);
    for(TI input_i=0; input_i<actor::MODEL::INPUT_DIM; input_i++){
        T value = rlt::get(observation, 0, input_i);
        T mean = rlt::get(observation_mean::container, 0, input_i);
        T std = rlt::get(observation_std::container, 0, input_i);
        rlt::set(observation, 0, input_i, (value - mean) / std);
    }
    rlt::evaluate(device, actor::model, observation, action, buffer);

    rlt::print(device, action);

    
    return 0;
}