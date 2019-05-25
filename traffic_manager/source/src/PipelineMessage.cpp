// Member definitions for class PipelineMessage

#include "PipelineMessage.hpp"

namespace traffic_manager {

    PipelineMessage::PipelineMessage(){}
    PipelineMessage::~PipelineMessage(){}


    int PipelineMessage::getActorID(){
        return actor_id;
    }
    void PipelineMessage::setActor( carla::SharedPtr<carla::client::Actor> actor)
    {
        this->actor = actor;
        this->actor_id = actor->GetId();
    }
    
    carla::SharedPtr<carla::client::Actor> PipelineMessage::getActor()
    {
        return actor;
    }
}
