
#include "../../multirotor.h"

namespace rl_tools::rl::environments::multirotor::parameters::dynamics{
    namespace x500{
        template<typename T, typename TI, typename REWARD_FUNCTION>
        constexpr typename ParametersBase <T, TI, TI(4), REWARD_FUNCTION>::Dynamics sim = {
            // Rotor positions
            {
                {
                    +0.176776695296636,
                    -0.176776695296636,
                    0

                },
                {
                    -0.176776695296636,
                    +0.176776695296636,
                    0

                },
                {
                    +0.176776695296636,
                    +0.176776695296636,
                    0
                },
                {
                    -0.176776695296636,
                    -0.176776695296636,
                    0
                },
            },
            // Rotor thrust directions
            {
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
            },
            // Rotor torque directions
            {
                {0, 0, -1},
                {0, 0, -1},
                {0, 0, +1},
                {0, 0, +1},
            },
            // thrust constants
            {
                0,
                0,
                8.74
            },
            // torque constant
//            0.025126582278481014,
            0.11697849233439939,
            // mass vehicle
            2.000,
            // gravity
            {0, 0, -9.81},
            // J
            {
                {
                    0.0216666666666666,
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    0.0216666666666666,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000,
                    0.04
                }
            },
            // J_inv
            {
                {
                    46.153846153846295,
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    46.153846153846295,
                    0.0000000000000000000000000000000000000000
                },
                {
                    0.0000000000000000000000000000000000000000,
                    0.0000000000000000000000000000000000000000,
                    25
                }
            },
            // T, RPM time constant
            0.03,
            // action limit
            {0.0, 1.0},
        };

    }
}
