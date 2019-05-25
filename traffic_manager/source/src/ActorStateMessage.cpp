//Definition of ActorStateMessage class members

#include "ActorStateMessage.hpp"

namespace traffic_manager{

    ActorStateMessage::ActorStateMessage(){}

    ActorStateMessage::~ActorStateMessage(){}

    void ActorStateMessage::setActorTransform(carla::geom::Transform actor_transform)
    {
        this->actor_transform = actor_transform;
    }
    carla::geom::Transform ActorStateMessage::getActorTransform()
    {
        return actor_transform;
    }
}