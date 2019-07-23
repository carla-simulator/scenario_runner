// Implementation for an in memory descrete map representation

#include "InMemoryMap.hpp"
#include<typeinfo>

namespace traffic_manager {

InMemoryMap::InMemoryMap(traffic_manager::TopologyList topology) {
    this->topology = topology;
}
InMemoryMap::~InMemoryMap(){}

void InMemoryMap::setUp(int sampling_resolution){
    // Creating dense topology
    auto ZERO = 0.0001; // Very important that this is less than 10^-4
    for(auto &pair : this->topology) {

        // Looping through every topology segment
        auto begin_waypoint = pair.first;
        auto end_waypoint = pair.second;

        if (begin_waypoint->GetTransform().location.Distance(
                end_waypoint->GetTransform().location) > ZERO) {

            // Adding entry waypoint
            auto current_waypoint = begin_waypoint;
            this->dense_topology.push_back(std::make_shared<SimpleWaypoint>(current_waypoint));
            this->entry_node_list.push_back(this->dense_topology.back());

            // Populating waypoints from begin_waypoint to end_waypoint
            while (current_waypoint->GetTransform().location.Distance(
                    end_waypoint->GetTransform().location) > sampling_resolution) {

                current_waypoint = current_waypoint->GetNext(sampling_resolution)[0];
                auto previous_wp = this->dense_topology.back();
                this->dense_topology.push_back(std::make_shared<SimpleWaypoint>(current_waypoint));
                previous_wp->setNextWaypoint({this->dense_topology.back()});
            }

            // Adding exit waypoint
            auto previous_wp = this->dense_topology.back();
            this->dense_topology.push_back(std::make_shared<SimpleWaypoint>(end_waypoint));
            previous_wp->setNextWaypoint({this->dense_topology.back()});
            this->exit_node_list.push_back(this->dense_topology.back());
        }
    }

    // Linking segments
    int i = 0, j = 0;
    for (auto end_point : this->exit_node_list) {
        for (auto begin_point : this->entry_node_list) {
            if (end_point->distance(begin_point->getLocation()) < ZERO and i != j) { // Make this a constant
                end_point->setNextWaypoint({begin_point});
            }
            j++;
        }
        i++;
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
    return closest_waypoint;
}

std::vector<std::shared_ptr<SimpleWaypoint>> InMemoryMap::get_dense_topology() {
    return this->dense_topology;
}

}
