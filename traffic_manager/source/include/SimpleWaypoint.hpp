// A simple waypoint class
#pragma once

#include <memory.h>
#include "carla/Memory.h"
#include "carla/geom/Vector3D.h"
#include "carla/geom/Location.h"
#include "carla/client/Waypoint.h"

namespace traffic_manager {

class SimpleWaypoint
{
private:
    carla::SharedPtr<carla::client::Waypoint> waypoint;
    std::vector<std::shared_ptr<SimpleWaypoint>> next_waypoints;
public:
    SimpleWaypoint(carla::SharedPtr<carla::client::Waypoint> waypoint);
    ~SimpleWaypoint();
    std::vector<std::shared_ptr<SimpleWaypoint>> getNextWaypoint();
    int setNextWaypoint(std::vector<std::shared_ptr<SimpleWaypoint>> next_waypoints);
    float distance(carla::geom::Location location);
    carla::geom::Vector3D getVector();
    std::vector<float> getXYZ();
};

} // namespace traffic_manager
