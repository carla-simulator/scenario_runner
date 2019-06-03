// Implementation for an in memory descrete map representation

#include "InMemoryMap.hpp"
#include<typeinfo>

namespace traffic_manager {

InMemoryMap::InMemoryMap(traffic_manager::TopologyList topology) {
    this->topology = topology;
}
InMemoryMap::~InMemoryMap(){}

std::pair<int, int> InMemoryMap::make_node_key(carla::SharedPtr<carla::client::Waypoint> waypooint) {
    auto current_location = waypooint->GetTransform().location;
    int rounded_x = static_cast<int>(std::round(current_location.x * 10));
    int rounded_y = static_cast<int>(std::round(current_location.y * 10));
    return std::pair<int, int>(rounded_x, rounded_y);
}

void InMemoryMap::setUp(int sampling_resolution){
    for(auto const &pair : this->topology) {
        // Looping through every topology segment
        auto begin_waypoint = pair.first;
        auto end_waypoint = pair.second;
        auto current_waypoint = begin_waypoint;

        auto begin_node_key = this->make_node_key(begin_waypoint);
        auto end_node_key = this->make_node_key(end_waypoint);

        // Checking previous entry for begin_waypoint
        typedef std::pair<std::pair<int, int>, std::shared_ptr<SimpleWaypoint>> NodeEntry;
        if(this->entry_node_map.find(begin_node_key) == this->entry_node_map.end()) {
            this->dense_topology.push_back(std::make_shared<SimpleWaypoint>(begin_waypoint));
            this->entry_node_map.insert(NodeEntry(begin_node_key, this->dense_topology.back()));
        }
        auto entry_node_ptr = this->entry_node_map[begin_node_key];
        // Cross segment linking for previous occurance of begin_waypoint in exit_node_map
        if(this->exit_node_map.find(begin_node_key) == this->exit_node_map.end());
        else {
            auto exit_node_ptr = this->exit_node_map[begin_node_key];
            exit_node_ptr->setNextWaypoint({entry_node_ptr});
        }

        // Populating waypoints from begin_waypoint to end_waypoint
        this->dense_topology.push_back(std::make_shared<SimpleWaypoint>(current_waypoint));
        entry_node_ptr->setNextWaypoint({this->dense_topology.back()});
        while (
            current_waypoint->GetTransform().location.Distance(
                end_waypoint->GetTransform().location) > sampling_resolution) {
            current_waypoint = current_waypoint->GetNext(1.0)[0];
            auto previous_wp = this->dense_topology.back();
            this->dense_topology.push_back(std::make_shared<SimpleWaypoint>(current_waypoint));
            previous_wp->setNextWaypoint({this->dense_topology.back()});
        }

        // Checking previous entry for end_waypoint
        if(this->exit_node_map.find(end_node_key) == this->exit_node_map.end()){
            this->dense_topology.push_back(std::make_shared<SimpleWaypoint>(end_waypoint));
            this->exit_node_map.insert(NodeEntry(end_node_key, this->dense_topology.back()));
        }
        // Cross segment linking for previous occurance of end_waypoint in entry_node_map
        auto exit_node_ptr = this->exit_node_map[end_node_key];
        if(this->entry_node_map.find(end_node_key) == this->entry_node_map.end());
        else {
            auto entry_node_ptr = this->entry_node_map[end_node_key];
            exit_node_ptr->setNextWaypoint({entry_node_ptr});
        }
    }
}

std::shared_ptr<SimpleWaypoint> InMemoryMap::getWaypoint(carla::geom::Location location) {
    /* Dumb first draft implementation. Need more efficient code for the functionality */
    std::shared_ptr<SimpleWaypoint> closest_waypoint;
    float min_distance = std::numeric_limits<float>::max();
    for(auto simple_waypoint : this->dense_topology){
        float current_distance = simple_waypoint->distance(location);
        if (current_distance < min_distance) {
            min_distance = current_distance;
            closest_waypoint = simple_waypoint;
        }
    }
    std::cout << "Closest waypoint distance : " << min_distance << std::endl;

    return closest_waypoint;
}

}
