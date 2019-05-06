// Implementation for SimpleWaypoint

#include "SimpleWaypoint.hpp"

namespace traffic_manager{

SimpleWaypoint::SimpleWaypoint(carla::SharedPtr<carla::client::Waypoint>) {
    this->waypoint = waypoint;
    this->next_waypoint = NULL;
    this->previous_waypoint = NULL;
}
SimpleWaypoint::~SimpleWaypoint(){}

void SimpleWaypoint::setNextWaypoint(SimpleWaypoint* next_waypoint) {
    this->next_waypoint = next_waypoint;
}

void SimpleWaypoint::setPreviousWaypoint(SimpleWaypoint* previous_waypoint) {
    this->previous_waypoint = previous_waypoint;
}

SimpleWaypoint* SimpleWaypoint::getNextWaypoint() {
    return this->next_waypoint;
}

SimpleWaypoint* SimpleWaypoint::getPreviousWaypoint() {
    return this->previous_waypoint;
}

} // namespace traffic_manager