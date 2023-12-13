
#include "../../multirotor.h"

namespace rl_tools::rl::environments::multirotor::parameters::dynamics{
    template<typename T, typename TI, typename REWARD_FUNCTION>
    constexpr typename ParametersBase <T, TI, TI(4), REWARD_FUNCTION>::Dynamics race = {
            // Rotor positions
            {
                    {
                            0.0775,
                            -0.0981,
                            0

                    },
                    {
                            -0.0775,
                            0.0981,
                            0

                    },
                    {
                            0.0775,
                            0.0981,
                            0

                    },
                    {
                            -0.0775,
                            -0.0981,
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
                    5.886
            },
            // torque constant
//            0.025126582278481014,
            0.005964552,
            // mass vehicle
            0.600,
            // gravity
            {0, 0, -9.81},
            // J
            {
                    {
                            0.0004692632566870014,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0004692632566870014,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            0.0007273588879652079
                    }
            },
            // J_inv
            {
                    {
                            2131.16326995376,
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            2131.16326995376,
                            0.0000000000000000000000000000000000000000
                    },
                    {
                            0.0000000000000000000000000000000000000000,
                            0.0000000000000000000000000000000000000000,
                            1374.8371217371216

                    }
            },
            // T, RPM time constant
            0.15,
            // action limit
            {0, 1},
    };
}
