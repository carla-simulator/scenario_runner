//definition of FeederCallable class members

#include "FeederCallable.hpp"

namespace traffic_manager
{
    Feedercallable::Feedercallable(
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue,
        RegisteredActorMessage* shared_data):
        PipelineCallable(input_queue,output_queue, shared_data){}
    Feedercallable::~Feedercallable(){}

    PipelineMessage Feedercallable::action (PipelineMessage message)
    {
        auto reg_actors = (RegisteredActorMessage*) shared_data; 
        while(true)
        {
            for (auto actor: reg_actors->shared_actor_list)
            {
                message.setActor(actor);
                writeQueue(message);
            }
        }
        PipelineMessage empty_message;
        return empty_message;
    }
}