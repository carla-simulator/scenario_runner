// Member definitions for class PipelineMessage

#include "PipelineMessage.hpp"

namespace traffic_manager {

    PipelineMessage::PipelineMessage(){}
    PipelineMessage::~PipelineMessage(){}


    int PipelineMessage::getActorID(){
        return actor_id;
    }
    void PipelineMessage::setActorID(int actor_id){
        this->actor_id = actor_id;
    }

}
