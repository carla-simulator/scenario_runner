//Declaration of ActorStateCallable class members
#pragma once

#include "carla/client/Actor.h"
#include "carla/client/ActorList.h"
#include "PipelineCallable.hpp"

namespace traffic_manager{

class ActorStateCallable: public PipelineCallable
{

public:
    ActorStateCallable(
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue);
    ~ActorStateCallable();

    PipelineMessage action(PipelineMessage message);
};

}