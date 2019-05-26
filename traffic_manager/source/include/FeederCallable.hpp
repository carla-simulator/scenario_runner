//Declaration of FeederCallable class members
#pragma once

#include "carla/client/Actor.h"
#include "PipelineCallable.hpp"
#include "RegisteredActorMessage.hpp"

namespace traffic_manager{

    class Feedercallable: public PipelineCallable
    {
        public:
        Feedercallable(
            SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue,
            RegisteredActorMessage* reg_actor);
        ~Feedercallable();

        PipelineMessage action (PipelineMessage message);
    };
}