// Member Defination for Class ActorStateMessage

#include "ActorStateMessage.hpp"

namespace traffic_manager {

    ActorStateMessage::ActorStateMessage(carla::SharedPtr<carla::client::ActorList> _actor_list)
    {
        this->_actor_list = _actor_list;
    }

    ActorStateMessage::~ActorStateMessage(){}

    carla::geom::Transform ActorStateMessage::getActorTransform(int actor_id){
        std::vector<carla::geom::Transform>::iterator it;
        it = std::find (all_actor_transform.begin(), all_actor_transform.end(), *actor_id);
        if(it != all_actor_transform.end())
        {
            this->actor_transform = *it;
            return actor_transform;
        }
        //_actor_list->Find(actor_id)->GetTransform();
    }

    void ActorStateMessage::setActorTransform(carla::geom::Transform actor_transform){
        this->actor_transform = actor_transform;
    }
    
    PipelineMessage action(PipelineMessage message)
    {
        return actor_transform;
    }
}