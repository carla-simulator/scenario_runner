// Definition of RegisteredActorMessage class

#include "RegisteredActorMessage.hpp"

namespace traffic_manager
{
    RegisteredActorMessage::RegisteredActorMessage()
    {
        
    }
    RegisteredActorMessage::~RegisteredActorMessage(){}

    void RegisteredActorMessage::addActorID(int actor_id){
        shared_actor_list.push_back(actor_id);
    }
    void RegisteredActorMessage::removeActorID(int actor_id){
        shared_actor_list.erase(std::remove(shared_actor_list.begin(), shared_actor_list.end(), actor_id), shared_actor_list.end());
    }
}