//definition of FeederCallable class members

#include "FeederCallable.hpp"

namespace traffic_manager
{
    Feedercallable::Feedercallable(
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue,
        SharedData* shared_data):
        PipelineCallable(input_queue,output_queue, shared_data){}
    Feedercallable::~Feedercallable(){}

    PipelineMessage Feedercallable::action (PipelineMessage message)
    {
        while(true)
        {
            for (auto actor: shared_data->registered_actors)
            {
                message.setActor(actor);
                writeQueue(message);
            }
        }
        PipelineMessage empty_message;
        return empty_message;
    }
}