// Implementation for an in memory descrete map representation

#include "InMemoryMap.hpp"
#include<typeinfo>

namespace traffic_manager {

InMemoryMap::InMemoryMap(traffic_manager::TopologyList topology) {
    this->topology = topology;
    this->dense_topology = std::vector<SimpleWaypoint>();
    this->entry_node_map = NodeMap();
    this->exit_node_map = NodeMap();
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

        auto begin_waypoint = pair.first;
        auto end_waypoint = pair.second;
        auto current_waypoint = begin_waypoint;

        auto begin_node_key = this->make_node_key(begin_waypoint);
        auto end_node_key = this->make_node_key(end_waypoint);

        typedef std::pair<std::pair<int, int>, SimpleWaypoint*> NodeEntry;

        this->dense_topology.push_back(SimpleWaypoint(begin_waypoint));
        if(this->entry_node_map.find(begin_node_key) == this->entry_node_map.end())
            this->entry_node_map.insert(NodeEntry(begin_node_key, &(this->dense_topology.back())));
        if(this->exit_node_map.find(begin_node_key) == this->exit_node_map.end());
        else {
            auto exit_node_ptr = this->exit_node_map[begin_node_key];
            exit_node_ptr->setNextWaypoint({&(this->dense_topology.back())});
        }

        while (true) {
            current_waypoint = current_waypoint->GetNext(1.0)[0];
            this->dense_topology.push_back(SimpleWaypoint(current_waypoint));
            auto end_ptr = this->dense_topology.end();
            (end_ptr-1)->setNextWaypoint({&(this->dense_topology.back())});
        }

        this->dense_topology.push_back(SimpleWaypoint(end_waypoint));
        if(this->exit_node_map.find(end_node_key) == this->exit_node_map.end())
            this->exit_node_map.insert(NodeEntry(end_node_key, &(this->dense_topology.back())));
        if(this->entry_node_map.find(end_node_key) == this->entry_node_map.end());
        else {
            this->dense_topology.back().setNextWaypoint({this->entry_node_map[end_node_key]});
        }

    }
}

SimpleWaypoint* InMemoryMap::getWaypoint(carla::geom::Location location) {
    /* Dumb first draft implementation. Need more efficient code for the functionality */
    SimpleWaypoint* closest_waypoint;
    float min_distance = std::numeric_limits<float>::max();
    for(auto &simple_waypoint : this->dense_topology){
        std::cout<<typeid(simple_waypoint).name()<<std::endl;
        float current_distance = simple_waypoint.distance(location);
        if (current_distance < min_distance) {
            min_distance = current_distance;
            closest_waypoint = &simple_waypoint;
        }
    }

    return closest_waypoint;
}

std::vector<SimpleWaypoint*> InMemoryMap::listofAllWaypoint(){
    std::vector<SimpleWaypoint*> all_waypoints;
    SimpleWaypoint* _waypoint;
    for(auto &simple_waypoint : this->dense_topology){
        _waypoint = &simple_waypoint;
        all_waypoints.push_back(_waypoint);
    }
    return all_waypoints;
}


}
