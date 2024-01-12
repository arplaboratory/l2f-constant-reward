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
            using ADDITIONAL_METRICS = set::Component<
            rlt::rl::utils::validation::metrics::SettlingFractionPosition<TI, 200>,
            set::Component<MeanErrorMean<POSITION, TI, 100>,
            set::Component<MaxErrorMean<POSITION, TI, 100>,
            set::Component<MaxErrorStd <POSITION, TI, 100>,
            set::Component<MeanErrorMean<POSITION, TI, 200>,
            set::Component<MaxErrorMean<POSITION, TI, 200>,
            set::Component<MaxErrorStd <POSITION, TI, 200>,
            set::Component<MeanErrorMean<POSITION, TI, 300>,
            set::Component<MaxErrorMean<POSITION, TI, 300>,
            set::Component<MaxErrorStd <POSITION, TI, 300>,
            set::Component<MeanErrorMean<POSITION, TI, 400>,
            set::Component<MaxErrorMean<POSITION, TI, 400>,
            set::Component<MaxErrorStd <POSITION, TI, 400>,
            set::Component<MeanErrorMean<ANGLE, TI, 100>,
            set::Component<MaxErrorMean<ANGLE, TI, 100>,
            set::Component<MaxErrorStd <ANGLE, TI, 100>,
            set::Component<MeanErrorMean<LINEAR_VELOCITY, TI, 100>,
            set::Component<MaxErrorMean<LINEAR_VELOCITY, TI, 100>,
            set::Component<MaxErrorStd <LINEAR_VELOCITY, TI, 100>,
            set::Component<MeanErrorMean<LINEAR_VELOCITY, TI, 200>,
            set::Component<MaxErrorMean<LINEAR_VELOCITY, TI, 200>,
            set::Component<MaxErrorStd <LINEAR_VELOCITY, TI, 200>,
            set::Component<MeanErrorMean<ANGULAR_VELOCITY, TI, 100>,
            set::Component<MaxErrorMean<ANGULAR_VELOCITY, TI, 100>,
            set::Component<MaxErrorStd <ANGULAR_VELOCITY, TI, 100>,
            set::Component<MeanErrorMean<ANGULAR_VELOCITY, TI, 200>,
            set::Component<MaxErrorMean<ANGULAR_VELOCITY, TI, 200>,
            set::Component<MaxErrorStd <ANGULAR_VELOCITY, TI, 200>,
            set::Component<MeanErrorMean<ANGULAR_ACCELERATION, TI,   0>,
            set::Component<MaxErrorMean<ANGULAR_ACCELERATION, TI,   0>,
            set::Component<MaxErrorStd <ANGULAR_ACCELERATION, TI,   0>,
            set::Component<MeanErrorMean<ANGULAR_ACCELERATION, TI, 100>,
            set::Component<MaxErrorMean<ANGULAR_ACCELERATION, TI, 100>,
            set::Component<MaxErrorStd <ANGULAR_ACCELERATION, TI, 100>,
            set::Component<MeanErrorMean<ANGULAR_ACCELERATION, TI, 200>,
            set::Component<MaxErrorMean<ANGULAR_ACCELERATION, TI, 200>,
            set::Component<MaxErrorStd <ANGULAR_ACCELERATION, TI, 200>,
            set::Component<MeanErrorMean<ACTION, TI,   0>,
            set::Component<MaxErrorMean<ACTION, TI,   0>,
            set::Component<MaxErrorStd <ACTION, TI,   0>,
            set::Component<MeanErrorMean<ACTION, TI, 100>,
            set::Component<MaxErrorMean<ACTION, TI, 100>,
            set::Component<MaxErrorStd <ACTION, TI, 100>,
            set::Component<MeanErrorMean<ACTION, TI, 200>,
            set::Component<MaxErrorMean<ACTION, TI, 200>,
            set::Component<MaxErrorStd <ACTION, TI, 200>,
            set::Component<MeanErrorMean<ACTION_RELATIVE, TI,   0>,
            set::Component<MaxErrorMean<ACTION_RELATIVE, TI,   0>,
            set::Component<MaxErrorStd <ACTION_RELATIVE, TI,   0>,
            set::Component<MeanErrorMean<ACTION_RELATIVE, TI, 100>,
            set::Component<MaxErrorMean<ACTION_RELATIVE, TI, 100>,
            set::Component<MaxErrorStd <ACTION_RELATIVE, TI, 100>,
            set::Component<MeanErrorMean<ACTION_RELATIVE, TI, 200>,
            set::Component<MaxErrorMean<ACTION_RELATIVE, TI, 200>,
            set::Component<MaxErrorStd <ACTION_RELATIVE, TI, 200>,
            FinalComponent>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;
            using METRICS = DefaultMetrics<ADDITIONAL_METRICS>;
        };
    }
    namespace config{
        template <typename T_SUPER_CONFIG>
        using Validation = config_builder::Validation<T_SUPER_CONFIG>;
    }
}