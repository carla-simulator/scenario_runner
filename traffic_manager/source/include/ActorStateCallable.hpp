//Declaration of ActorStateCallable class members

#include "carla/client/Actor.h"
#include "carla/client/ActorList.h"
#include "ActorReadStage.hpp"
#include "PipelineCallable.hpp"

namespace traffic_manager{

class ActorStateCallable: public PipelineCallable
{
private:
    int _actor_id;
    carla::geom::Transform _actor_transform;
    carla::SharedPtr<carla::client::ActorList> _actor_list;
public:
    std::mutex inmutex;
    std::mutex outmutex;
    ActorStateCallable(carla::SharedPtr<carla::client::ActorList> _actor_list, int _actor_id);
    ~ActorStateCallable();
    carla::geom::Transform getActorTransform();
    void setActorTransform(carla::geom::Transform _actor_transform);
    PipelineMessage action(PipelineMessage message);
};

}