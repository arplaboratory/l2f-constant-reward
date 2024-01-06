
#include "../../multirotor.h"

#include <rl_tools/math/operations_generic.h>

namespace rl_tools::rl::environments::multirotor::parameters::termination{
    template<typename SPEC>
    constexpr typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::MDP::Termination classic = {
            true,           // enable
            0.6,            // position
            10,         // linear velocity
            10 // angular velocity
    };
    template<typename SPEC>
    constexpr typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::MDP::Termination fast_learning = {
        true,           // enable
        0.6,            // position
        10,         // linear velocity
        40 // angular velocity
    };
}