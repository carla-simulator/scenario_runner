//Declaration of class members

#pragma once
#include <cmath>
#include "carla/rpc/VehicleControl.h"
#include "PipelineCallable.hpp"

namespace traffic_manager{
    
    class ActorPIDCallable: public PipelineCallable
    {
    private:
        float k_v;
        float k_s;
        float target_velocity;
        
    public:
        ActorPIDCallable(float k_v, float k_s, float target_velocity,
            SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue);
        ~ActorPIDCallable();
        PipelineMessage action(PipelineMessage message);
    };
}