// Declaration of class to localise actor

#include <vector>
#include "carla/Memory.h"
#include "carla/geom/Location.h"
#include "carla/client/ActorList.h"
#include "carla/client/Actor.h"
#include "InMemoryMap.hpp"
#include "ActorReadState.hpp"

namespace traffic_manager {

    class ActorLocalizatiion {
        
    public:
        ActorLocalizatiion();
       ~ActorLocalizatiion();
        std::vector<SimpleWaypoint*> getWaypointBuffer(carla::geom::Location actor_location );

    private:
        carla::geom::Location actor_location;

    };
}