#include <rl_tools/rl/utils/validation.h>
#include <learning_to_fly/simulator/metrics.h>
namespace learning_to_fly{
    namespace config_builder{ // to not pollute ::config with "using namespace"
        using namespace rlt::rl::utils::validation;
        using namespace rlt::rl::utils::validation::set;
        using namespace rlt::rl::utils::validation::metrics;
        using namespace rlt::rl::utils::validation::metrics::multirotor;
        template <typename T_SUPER_CONFIG>
        struct Validation: T_SUPER_CONFIG{
            using SUPER = T_SUPER_CONFIG;
            using T = typename SUPER::T;
            using TI = typename SUPER::TI;
            using ENVIRONMENT = typename SUPER::ENVIRONMENT;
            using VALIDATION_SPEC = rlt::rl::utils::validation::Specification<T, TI, ENVIRONMENT>;
            static constexpr TI VALIDATION_N_EPISODES = 10;
            static constexpr TI VALIDATION_MAX_EPISODE_LENGTH = SUPER::ENVIRONMENT_STEP_LIMIT_EVALUATION;
            using TASK_SPEC = TaskSpecification<VALIDATION_SPEC, VALIDATION_N_EPISODES, VALIDATION_MAX_EPISODE_LENGTH>;
            using ADDITIONAL_METRICS = Component<
            rlt::rl::utils::validation::metrics::SettlingFractionPosition<TI, 200>,
            Component<MaxErrorMean<POSITION, TI, 100>,
            Component<MaxErrorStd <POSITION, TI, 100>,
            Component<MaxErrorMean<POSITION, TI, 200>,
            Component<MaxErrorStd <POSITION, TI, 200>,
            Component<MaxErrorMean<ANGLE, TI, 100>,
            Component<MaxErrorStd <ANGLE, TI, 200>,
            Component<MaxErrorMean<LINEAR_VELOCITY, TI, 100>,
            Component<MaxErrorStd <LINEAR_VELOCITY, TI, 100>,
            Component<MaxErrorMean<LINEAR_VELOCITY, TI, 200>,
            Component<MaxErrorStd <LINEAR_VELOCITY, TI, 200>,
            Component<MaxErrorMean<ANGULAR_VELOCITY, TI, 100>,
            Component<MaxErrorStd <ANGULAR_VELOCITY, TI, 100>,
            Component<MaxErrorMean<ANGULAR_VELOCITY, TI, 200>,
            Component<MaxErrorStd <ANGULAR_VELOCITY, TI, 200>,
            Component<MaxErrorMean<ANGULAR_ACCELERATION, TI,   0>,
            Component<MaxErrorStd <ANGULAR_ACCELERATION, TI,   0>,
            Component<MaxErrorMean<ANGULAR_ACCELERATION, TI, 100>,
            Component<MaxErrorStd <ANGULAR_ACCELERATION, TI, 100>,
            Component<MaxErrorMean<ANGULAR_ACCELERATION, TI, 200>,
            Component<MaxErrorStd <ANGULAR_ACCELERATION, TI, 200>,
//            Component<MaxErrorMean<ACTION, TI,   0>,
//            Component<MaxErrorStd <ACTION, TI,   0>,
//            Component<MaxErrorMean<ACTION, TI, 100>,
//            Component<MaxErrorStd <ACTION, TI, 100>,
//            Component<MaxErrorMean<ACTION, TI, 200>,
//            Component<MaxErrorStd <ACTION, TI, 200>,
            FinalComponent>>>>>>>>>>>>>>>>>>>>>;
            using METRICS = DefaultMetrics<ADDITIONAL_METRICS>;
        };
    }
    namespace config{
        template <typename T_SUPER_CONFIG>
        using Validation = config_builder::Validation<T_SUPER_CONFIG>;
    }
}