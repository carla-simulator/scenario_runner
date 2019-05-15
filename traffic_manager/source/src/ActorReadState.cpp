
#include "ActorReadState.hpp"

namespace traffic_manager {

    ActorReadState::ActorReadState(){}
    ActorReadState::~ActorReadState(){}

    // _actor_list = getActorList();

    // void createPipelineCallables()
    // {
        
    // }

    

    
    std::vector<carla::geom::Location> ActorReadState::getLocation(carla::SharedPtr<carla::client::ActorList> _actor_list){
        
        std::pair<int, int> location_coordinates;
        std::vector<carla::geom::Location> actor_current_location_list;
       
        for(auto  it = _actor_list->begin() ; it != _actor_list->end(); it++ ) {
            
            auto current_location = (*it)->GetTransform().location;
            actor_current_location_list.push_back(current_location);
            // int rounded_x = static_cast<int>(std::round(current_location.x));
            // int rounded_y = static_cast<int>(std::round(current_location.y));
            // location_coordinates.first = rounded_x;
            // location_coordinates.second = rounded_y;
            // std::cout<< location_coordinates.first <<"\t" << location_coordinates.second <<"\n";   
        }
        return actor_current_location_list;
    
        
    }
}
