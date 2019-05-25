// Declaration of RegisteredActorMessage class
#pragma once
#include "PipelineMessage.hpp"

namespace traffic_manager
{
    class RegisteredActorMessage: public PipelineMessage
    {
        public:
        std::vector<carla::SharedPtr<carla::client::Actor>> shared_actor_list;
        RegisteredActorMessage();
        ~RegisteredActorMessage();
        void addActor(carla::SharedPtr<carla::client::Actor> actor);
        void removeActor(carla::SharedPtr<carla::client::Actor> actor);
    };
}