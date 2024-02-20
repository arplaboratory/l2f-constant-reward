#include <rl_tools/operations/cpu.h>

#include <rl_tools/containers/persist.h>
#include <rl_tools/containers/persist_code.h>

#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/parameters/persist_code.h>

#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/operations_generic.h>
#include <rl_tools/nn_models/sequential/model.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

namespace rlt = rl_tools;

#include <CLI/CLI.hpp>


// this file imports an hdf5 model and outputs a code (.h) export of it
// make sure that the compile-time model structure matches the structure in the hdf5 file


using DEVICE = rlt::devices::DefaultCPU;
using PARAMETER_TYPE = rlt::nn::parameters::Plain;
using T = float;
using TI = typename DEVICE::index_t;

namespace builder{ // to not pollute the global namespace we define the model in a namespace
    using namespace rlt::nn_models::sequential::interface; 
    struct ACTOR{
        static constexpr TI HIDDEN_DIM = 50;
        static constexpr TI BATCH_SIZE = 1;
        static constexpr TI OBSERVATION_DIM = 34;
        static constexpr TI ACTION_DIM = 4;
        static constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::RELU;
        static constexpr auto OUTPUT_ACTIVATION_FUNCTION = rlt::nn::activation_functions::IDENTITY;
        using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, OBSERVATION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Input>;
        using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Normal>;
        using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
        using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, ACTION_DIM, OUTPUT_ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Output>;
        using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

        using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
    };
}

using ACTOR = builder::ACTOR::MODEL;



#include <filesystem>

int main(int argc, char** argv){
    CLI::App app{"Learning to fly"};
    std::string model_path = "../../../src/bindings/actor_simulation_optimization.h5";
    std::string model_output_path = "../../../src/bindings/actor_simulation_optimization.h";
    app.add_option("-m,--model", model_path, "Path to the model file");//->required();
    CLI11_PARSE(app, argc, argv);

    DEVICE device;
    ACTOR model;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ACTOR::INPUT_DIM>> observation_mean;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ACTOR::INPUT_DIM>> observation_std;

    rlt::malloc(device, observation_mean);
    rlt::malloc(device, observation_std);


    //current cwd
    std::filesystem::path cwd = std::filesystem::current_path();
    rlt::malloc(device, model);
    HighFive::File file(model_path, HighFive::File::ReadOnly);
    auto actor_group = file.getGroup("actor");
    auto observation_distribution_group = actor_group.getGroup("observation_distribution");
    rlt::load(device, observation_mean, observation_distribution_group, "mean");
    rlt::load(device, observation_std, observation_distribution_group, "std");
    std::cout << "Current path is " << cwd << std::endl;
    std::cout << "Trying to load from: " << model_path << std::endl;
    rlt::load(device, model, actor_group);
    auto code_split = rlt::save_code_split(device, model, "actor");

    auto observation_mean_split = rlt::save_split(device, observation_mean, "observation_mean");
    auto observation_std_split = rlt::save_split(device, observation_std, "observation_std");

    auto headers = code_split.header + observation_mean_split.header + observation_std_split.header;
    auto body = code_split.body + observation_mean_split.body + observation_std_split.body;

    std::cout << "Trying to save to: " << model_output_path << std::endl;
    std::ofstream out(model_output_path);
    out << headers << body;
    out.close();
    // [0.4171, 0.4969, 0.4487, 0.2470]

    return 0; 
}