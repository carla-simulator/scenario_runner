// Declaration of class to localise actor

#include <vector>
#include "carla/Memory.h"
//#include "carla/geom/Location.h"
#include "carla/client/ActorList.h"
#include "carla/client/Actor.h"
#include "InMemoryMap.hpp"
#include "ActorReadState.hpp"

namespace traffic_manager {

    class ActorLocalization
    {

    private:
        //carla::geom::Location actor_location;
        carla::SharedPtr<carla::client::ActorList> _actor_list;

    public:
        ActorLocalization(carla::SharedPtr<carla::client::ActorList> _actor_list);
        ~ActorLocalization();
        carla::geom::Location getLocation(carla::SharedPtr<carla::client::ActorList> _actor_list);
        
        std::vector<SimpleWaypoint*> getWaypointBuffer(carla::geom::Location current_location);

    

    };
}