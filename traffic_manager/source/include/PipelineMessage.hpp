// Declaration for a common base class to all messages between pipeline stages
#pragma once

#include "carla/client/Actor.h"

namespace traffic_manager {

class PipelineMessage
{
private:
   int actor_id;
   carla::SharedPtr<carla::client::Actor> actor;
public:
    PipelineMessage();
    virtual ~PipelineMessage();
    int getActorID();
    void setActor(carla::SharedPtr<carla::client::Actor> actor);
    carla::SharedPtr<carla::client::Actor> getActor();
    
};

}