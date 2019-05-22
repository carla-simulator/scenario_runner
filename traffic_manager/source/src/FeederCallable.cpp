//definition of FeederCallable class members

#include "FeederCallable.hpp"

namespace traffic_manager
{
    
    Feedercallable::Feedercallable():PipelineCallable(NULL,NULL,inmutex,outmutex,20){}
    Feedercallable::~Feedercallable(){}
    PipelineMessage Feedercallable::action (PipelineMessage message)
    {
        RegisteredActorMessage* reg_actor;
        for (std::vector<int>::iterator it = reg_actor->shared_actor_list.begin(); it != reg_actor->shared_actor_list.end(); it++)
        {
            message.setActorID(*it);
            writeQueue(message);
        }
    }
}