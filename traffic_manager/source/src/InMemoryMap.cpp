// Implementation for an in memory descrete map representation

#include "InMemoryMap.hpp"

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
        auto begin_waypoint = pair.first;
        auto end_waypoint = pair.second;
        auto current_waypoint = begin_waypoint;
        auto begin_node_key = this->make_node_key(begin_waypoint);
        auto end_node_key = this->make_node_key(end_waypoint);
        typedef std::pair<std::pair<int, int>, carla::SharedPtr<carla::client::Waypoint>> NodeKey;
        if(this->entry_node_map.find(begin_node_key) == this->entry_node_map.end())
            this->entry_node_map.insert(NodeKey(begin_node_key, begin_waypoint));
        if(this->exit_node_map.find(end_node_key) == this->exit_node_map.end())
            this->exit_node_map.insert(NodeKey(end_node_key, begin_waypoint));
        
    }
}

}
