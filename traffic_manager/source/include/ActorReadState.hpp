//Declaration of class for reading actor state

#include "PipelineStage.hpp"
#include<vector>
#include "carla/Memory.h"
#include "carla/geom/Location.h"
#include "carla/client/ActorList.h"
#include "carla/client/Actor.h"
namespace traffic_manager {
    
class ActorReadState:PipelineStage
{
private:
    carla::SharedPtr<carla::client::ActorList> _actor_list;
public:
    ActorReadState(carla::SharedPtr<carla::client::ActorList> _actor_list);
    ~ActorReadState();
    std::vector<carla::geom::Location> getLocation(carla::SharedPtr<carla::client::ActorList> _actor_list);
    void createPipelineCallables();
};

} //namespace traffic_manager