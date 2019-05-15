// Member Defination for Class ActorStateMessage

#include "ActorStateMessage.hpp"

namespace traffic_manager {

    ActorStateMessage::ActorStateMessage(carla::SharedPtr<carla::client::ActorList> _actor_list){
        this->_actor_list = _actor_list;
    }
    ActorStateMessage::~ActorStateMessage(){}

    // void ActorStateMessage::setActorList(carla::SharedPtr<carla::client::ActorList> actor_list) {
    //     this->actor_list =actor_list;
    // }
    // carla::SharedPtr<carla::client::ActorList> ActorStateMessage::getActorList(){
    //     return actor_list;
    // }

    carla::geom::Transform ActorStateMessage::getActorTransform(int actor_id){
        return actor_transform;
        //_actor_list->Find(actor_id)->GetTransform();
    }

    void ActorStateMessage::setActorTransform(carla::geom::Transform actor_transform){
        this->actor_transform = actor_transform;
    }
   
    
}