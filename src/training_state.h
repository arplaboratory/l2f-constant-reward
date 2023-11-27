#include <queue>
#include <vector>
#include <mutex>
#ifdef LEARNING_TO_FLY_HYPERPARAMETER_OPTIMIZATION
#include <filesystem>
#endif

namespace learning_to_fly{
    template <typename T_CONFIG>
    struct TrainingState: rlt::rl::algorithms::td3::loop::TrainingState<T_CONFIG>{
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using ABLATION_SPEC = typename CONFIG::ABLATION_SPEC;
        std::string run_name;
        std::queue<std::vector<typename CONFIG::ENVIRONMENT::State>> trajectories;
        std::mutex trajectories_mutex;
        std::vector<typename CONFIG::ENVIRONMENT::State> episode;
        // validation
        rlt::rl::utils::validation::Task<typename CONFIG::TASK_SPEC> task;
        typename CONFIG::ENVIRONMENT validation_envs[CONFIG::VALIDATION_N_EPISODES];
        typename CONFIG::ACTOR_TYPE::template DoubleBuffer<CONFIG::VALIDATION_N_EPISODES> validation_actor_buffers;
        typename CONFIG::ENVIRONMENT_PARAMETERS env_parameters_base;
        typename CONFIG::ENVIRONMENT_PARAMETERS env_parameters_base_eval;
        struct Curriculum{
            struct Schedule{
                T factor;
                T limit;
            };
            Schedule position = {1.2, 40};
            Schedule orientation = {1.0, 100};
            Schedule linear_velocity = {1.4, 1};
            Schedule action = {1.4, 1};
        };
        Curriculum curriculum;
#ifdef LEARNING_TO_FLY_HYPERPARAMETER_OPTIMIZATION
        std::filesystem::path parameters_path;
        std::filesystem::path results_path;
#endif
    };
}
