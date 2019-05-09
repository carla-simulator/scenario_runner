//Reading the State of an Actor
#include<vector>
#include "carla/Memory.h"
#include "carla/geom/Location.h"
#include "carla/client/ActorList.h"
#include "carla/client/Actor.h"

namespace traffic_manager {

    class ReadActorState
    {
    private:
        carla::SharedPtr<carla::client::ActorList> _actor_list;
        
    public:
        ReadActorState(carla::SharedPtr<carla::client::ActorList> _actor_list);
        ~ReadActorState();
        std::vector<carla::geom::Location> getLocation(carla::SharedPtr<carla::client::ActorList> _actor_list);
    };
    
    
    
}// namespace traffic_manager