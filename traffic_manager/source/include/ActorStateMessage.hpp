//Declaration of ActorStateMessage class members

#pragma once
#include "PipelineMessage.hpp"

namespace traffic_manager{

    class ActorStateMessage: public PipelineMessage
    {
    private:
        carla::geom::Transform actor_transform;
        
    public:
        ActorStateMessage();
        ~ActorStateMessage();
        void setActorTransform(carla::geom::Transform actor_transform);
        carla::geom::Transform getActorTransform();    
    };
}