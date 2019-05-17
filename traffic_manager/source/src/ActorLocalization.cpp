// Defination of ActorLocalization class members

#include "ActorLocalization.hpp"

namespace traffic_manager {

    ActorLocalization::ActorLocalization(carla::SharedPtr<carla::client::ActorList> _actor_list){

        this->_actor_list = _actor_list;

    }
    ActorLocalization::~ActorLocalization(){}

    carla::geom::Location ActorLocalization::getLocation(carla::SharedPtr<carla::client::ActorList> _actor_list){
        for(auto  it = _actor_list->begin() ; it != _actor_list->end(); it++ ) {
            auto current_location = (*it)->GetTransform().location;
            return current_location;  
        }
    }
    
    std::vector<SimpleWaypoint*> ActorLocalization::getWaypointBuffer(carla::geom::Location current_location){
        std::vector<SimpleWaypoint*> waypointBuffer;
        InMemoryMap* topology;
        SimpleWaypoint*  simple_waypoint;
        auto closest_waypoint = topology->getWaypoint(current_location);
        waypointBuffer.push_back(closest_waypoint);
        // for(int i = 0 ; i < 10; i++){
        //    std::cout << &(simple_waypoint->getNextWaypoint());
        // }

        // std::vector<int>::iterator it;
        // it = std::find (InMemoryMap::dense_topology.begin(), vec.end(), ser); 
        // if (it != vec.end()) { 
        //     std::cout << "Element " << ser <<" found at position : " ; 
        //     std:: cout << it - vec.begin() + 1 << "\n" ; 
        // } 
        // else{std::cout << "Element not found.\n\n";}

        for(auto it = waypointBuffer.begin(); it != waypointBuffer.end(); it++){
            std::cout << *it << std::endl;
        }
    }
}