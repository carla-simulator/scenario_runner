//definition of FeederCallable class members

#include "FeederCallable.hpp"

namespace traffic_manager
{
    Feedercallable::Feedercallable(RegisteredActorMessage* reg_actor):PipelineCallable(NULL,NULL,inmutex,outmutex,20)
    {
        this-> reg_actor = reg_actor;
    }
    Feedercallable::~Feedercallable(){}
    PipelineMessage Feedercallable::action (PipelineMessage message)
    {   
        while(true)
        {        
            for (std::vector<int>::iterator it = reg_actor->shared_actor_list.begin(); it != reg_actor->shared_actor_list.end(); it++)
            {
                message.setActorID(*it);
                writeQueue(message);
            }
        }
        PipelineMessage empty_message;
        return empty_message;
    }
}