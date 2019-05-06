// Implementation for SimpleWaypoint

#include "SimpleWaypoint.hpp"

namespace traffic_manager{

SimpleWaypoint::SimpleWaypoint(carla::SharedPtr<carla::client::Waypoint>) {
    this->waypoint = waypoint;
    this->next_waypoints = std::vector<SimpleWaypoint*>();
}
SimpleWaypoint::~SimpleWaypoint(){}

void SimpleWaypoint::setNextWaypoint(std::vector<SimpleWaypoint*> next_waypoints) {
    this->next_waypoints.insert(
        this->next_waypoints.end(),
        next_waypoints.begin(),
        next_waypoints.end());
}

std::vector<SimpleWaypoint*> SimpleWaypoint::getNextWaypoint() {
    return this->next_waypoints;
}

float SimpleWaypoint::distance(carla::geom::Location location) {
    return this->waypoint->GetTransform().location.Distance(location);
}

} // namespace traffic_manager