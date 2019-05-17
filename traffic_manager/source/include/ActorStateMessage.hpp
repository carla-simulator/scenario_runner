#include "PipelineMessage.hpp"
#include "carla/client/Actor.h"
#include "carla/client/ActorList.h"
#include "ActorReadState.hpp"

namespace traffic_manager{

class ActorStateMessage:public PipelineCallable
{
private:
    carla::geom::Transform actor_transform;
    carla::SharedPtr<carla::client::ActorList> _actor_list;
public:    
    ActorStateMessage(carla::SharedPtr<carla::client::ActorList> _actor_list);
    ~ActorStateMessage();
    carla::geom::Transform getActorTransform(int actor_id);
    void setActorTransform(carla::geom::Transform actor_transform);
    PipelineMessage action(PipelineMessage message);
};

}