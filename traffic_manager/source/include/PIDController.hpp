//Declaration of class members

#pragma once

#include <vector>
#include "SharedData.hpp"

namespace traffic_manager {

    struct ActuationSignal
    {
        float throttle;
        float brake;
        float steer;
    };
    
    class PIDController {
        public:
            PIDController();

            StateEntry stateUpdate(
                StateEntry previous_state,
                float current_velocity,
                float target_velocity,
                float angular_deviation,
                TimeInstance current_time
            );

            ActuationSignal runStep(
                StateEntry present_state,
                StateEntry previous_state,
                std::vector<float> longitudinal_parameters,
                std::vector<float> lateral_parameters
            );
    };
}
