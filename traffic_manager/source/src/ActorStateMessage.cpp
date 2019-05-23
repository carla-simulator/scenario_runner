//Definition of ActorStateMessage class members

#include "ActorStateMessage.hpp"

namespace traffic_manager{

    ActorStateMessage::ActorStateMessage(
        carla::geom::Transform _actor_transform):_actor_transform(_actor_transform){}

    ActorStateMessage::~ActorStateMessage(){}
}