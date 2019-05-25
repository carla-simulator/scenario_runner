//Declaration of class for reading actor state
#pragma once

#include "PipelineStage.hpp"
#include <vector>
#include "carla/Memory.h"
#include "carla/geom/Transform.h"
#include "carla/client/ActorList.h"
#include "carla/client/Actor.h"
#include "ActorStateCallable.hpp"
#include "ActorStateMessage.hpp"

namespace traffic_manager {
    
class ActorStateStage: public PipelineStage
{
private:
    ActorStateMessage* actorstate_msg;

public:
    ActorStateStage(int output_buffer_size,
        std::queue<PipelineMessage>* input_queue,
        std::queue<PipelineMessage>* output_queue);
    ~ActorStateStage();
    void createPipelineCallables();
};

} //namespace traffic_manager