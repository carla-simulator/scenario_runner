// Implementation for SimpleWaypoint

#include "SimpleWaypoint.hpp"

namespace traffic_manager{

SimpleWaypoint::SimpleWaypoint(carla::SharedPtr<carla::client::Waypoint>) {
    this->waypoint = waypoint;
    this->next_waypoints = std::vector<SimpleWaypoint*>();
}
SimpleWaypoint::~SimpleWaypoint(){}

int SimpleWaypoint::setNextWaypoint(std::vector<SimpleWaypoint*> next_waypoints) {
    this->next_waypoints.insert(
        this->next_waypoints.end(),
        next_waypoints.begin(),
        next_waypoints.end());

    return 0;
}

std::vector<SimpleWaypoint*> SimpleWaypoint::getNextWaypoint() {
    return this->next_waypoints;
}

float SimpleWaypoint::distance(carla::geom::Location location) {
    return this->waypoint->GetTransform().location.Distance(location);
}

carla::geom::Vector3D SimpleWaypoint::getVector(){
    return waypoint->GetTransform().rotation.GetForwardVector();
}

std::vector<float> SimpleWaypoint::getXYZ(){
    float x = waypoint->GetTransform().location.x;
    float y = waypoint->GetTransform().location.y;
    float z = waypoint->GetTransform().location.z;
    std::vector<float> coordinates = {x,y,z};
    return coordinates;
}

} // namespace traffic_manager