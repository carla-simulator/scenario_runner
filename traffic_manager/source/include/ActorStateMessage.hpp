#include "PipelineMessage.hpp"
#include <carla/geom/Transform.h>

namespace traffic_manager{

class ActorStateMessage:PipelineMessage
{
private:
    carla::geom::Transform actor_transform;
    int actor_id;
public:    
    ActorStateMessage(/* args */);
    ~ActorStateMessage();
    int getActorID();
    void setActorID( int actor_id);
    carla::geom::Transform getActorTransform();
    void setActorTransform(carla::geom::Transform actor_transform);
};

}