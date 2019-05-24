//definition of FeederCallable class members

#include "FeederCallable.hpp"
#include "carla/client/Actor.h"
namespace traffic_manager
{
    Feedercallable::Feedercallable(
        RegisteredActorMessage* reg_actor,
        std::queue<PipelineMessage>* input_queue,
        std::queue<PipelineMessage>* output_queue,
        std::mutex& read_mutex,
        std::mutex& write_mutex,
        int output_buffer_size):
        reg_actor(reg_actor),
        PipelineCallable(input_queue,output_queue,read_mutex,write_mutex,output_buffer_size)
    {
        
    }
    Feedercallable::~Feedercallable(){}
    PipelineMessage Feedercallable::action (PipelineMessage message)
    {   
        while(true)
        {     
            for (std::vector<carla::SharedPtr<carla::client::Actor>>::iterator it = reg_actor->shared_actor_list.begin(); it != reg_actor->shared_actor_list.end(); it++)
            {
                message.setActor(*it);
                writeQueue(message);
            }
        }
        PipelineMessage empty_message;
        return empty_message;
    }
}