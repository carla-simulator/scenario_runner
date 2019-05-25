// Member Defination for Class ActorStateCallable

#include "ActorStateCallable.hpp"

namespace traffic_manager {

    ActorStateCallable::ActorStateCallable(ActorStateMessage* actorstate_msg,
        std::queue<PipelineMessage>* input_queue,
        std::queue<PipelineMessage>* output_queue,
        std::mutex& read_mutex,
        std::mutex& write_mutex,
        int output_buffer_size):
        
        actorstate_msg(actorstate_msg),
        PipelineCallable(input_queue,output_queue,read_mutex,write_mutex,output_buffer_size){}

    ActorStateCallable::~ActorStateCallable(){}

    PipelineMessage ActorStateCallable::action(PipelineMessage message)
    {
        message.getActor()->GetTransform();
        return message;
    }
}