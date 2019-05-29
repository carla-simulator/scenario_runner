//Declaration of FeederCallable class members
#pragma once

#include "carla/client/Actor.h"
#include "PipelineCallable.hpp"

namespace traffic_manager{

    class Feedercallable: public PipelineCallable
    {
        public:
        Feedercallable(
            SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue,
            SharedData* shared_data);
        ~Feedercallable();

        PipelineMessage action (PipelineMessage message);
    };
}