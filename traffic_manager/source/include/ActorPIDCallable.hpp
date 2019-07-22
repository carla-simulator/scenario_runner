//Declaration of class members

#pragma once

#include <chrono>
#include <cmath>
#include <vector>
#include "PipelineCallable.hpp"
#include "SharedData.hpp"

namespace traffic_manager{
    
    class ActorPIDCallable: public PipelineCallable
    {
    private:
        std::vector<float> vpid;
        std::vector<float> spid;
        float target_velocity;
        SharedData* shared_data;
        
    public:
        ActorPIDCallable(
            float target_velocity,
            SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue,
            SharedData* shared_data,
            std::vector<float> vpid,
            std::vector<float> spid
        );
        ~ActorPIDCallable();
        PipelineMessage action(PipelineMessage message);
    };
}
