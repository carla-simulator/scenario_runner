// A simple waypoint class

#include "carla/Memory.h"
#include "carla/geom/Vector3D.h"
#include "carla/geom/Location.h"
#include "carla/client/Waypoint.h"

namespace traffic_manager {

class SimpleWaypoint
{
private:
    carla::SharedPtr<carla::client::Waypoint> waypoint;
    SimpleWaypoint* next_waypoint;
public:
    SimpleWaypoint(carla::SharedPtr<carla::client::Waypoint> waypoint);
    ~SimpleWaypoint();
    SimpleWaypoint* getNextWaypoint();
    void setNextWaypoint(SimpleWaypoint* next_waypoint);
};

} // namespace traffic_manager
