// A simple waypoint class
#pragma once
#include "carla/Memory.h"
#include "carla/geom/Vector3D.h"
#include "carla/geom/Location.h"
#include "carla/client/Waypoint.h"

namespace traffic_manager {

class SimpleWaypoint
{
private:
    carla::SharedPtr<carla::client::Waypoint> waypoint;
    std::vector<SimpleWaypoint*> next_waypoints;
public:
    SimpleWaypoint(carla::SharedPtr<carla::client::Waypoint> waypoint);
    ~SimpleWaypoint();
    std::vector<SimpleWaypoint*> getNextWaypoint();
    void setNextWaypoint(std::vector<SimpleWaypoint*> next_waypoints);
    float distance(carla::geom::Location location);
};

} // namespace traffic_manager
