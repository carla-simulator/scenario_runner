// Definition of RegisteredActorMessage class

#include "RegisteredActorMessage.hpp"

namespace traffic_manager
{
    RegisteredActorMessage::RegisteredActorMessage(){}
    
    RegisteredActorMessage::~RegisteredActorMessage(){}

    void RegisteredActorMessage::addActor(carla::SharedPtr<carla::client::Actor> actor){
        shared_actor_list.push_back(actor);
    }
    void RegisteredActorMessage::removeActor(carla::SharedPtr<carla::client::Actor> actor){
        shared_actor_list.erase(std::remove(shared_actor_list.begin(), shared_actor_list.end(), actor), shared_actor_list.end());
    }
}