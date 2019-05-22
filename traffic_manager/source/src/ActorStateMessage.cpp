//Definition of ActorStateMessage class members

#include "ActorStateMessage.hpp"

namespace traffic_manager{

    ActorStateMessage::ActorStateMessage(PipelineMessage in_message, PipelineMessage out_message){
        this->in_message = in_message;
        this->out_message = out_message;
    }
    ActorStateMessage::~ActorStateMessage(){}
    // PipelineMessage ActorStateMessage::getStageMessage(PipelineMessage in_message)
    // {

    // }
}