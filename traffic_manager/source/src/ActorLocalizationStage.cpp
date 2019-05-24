// // Defination of ActorLocalizationStage class members

// #include "ActorLocalizationStage.hpp"

// namespace traffic_manager {

//     ActorLocalizationStage::ActorLocalizationStage(
//         carla::geom::Transform actor_transform):actor_transform(actor_transform){}

//     ActorLocalizationStage::~ActorLocalizationStage(){}

//     carla::geom::Location ActorLocalizationStage::getLocation(carla::geom::Transform actor_transform)
//     {
//         carla::geom::Location actor_location = actor_transform.location;
//         return  actor_location;
//     }
    
//     std::vector<SimpleWaypoint*> ActorLocalizationStage::getWaypointBuffer(carla::geom::Location actor_location)
//     {
//         std::vector<SimpleWaypoint*> waypointBuffer;
//         InMemoryMap* topology;
//         SimpleWaypoint*  closest_waypoint;
//         closest_waypoint = topology->getWaypoint(actor_location);
//         waypointBuffer.push_back(closest_waypoint);
       
//         std::vector<SimpleWaypoint*> map_waypoint = topology->listofAllWaypoint();
//         std::vector<SimpleWaypoint*>::iterator it;
//         it = std::find (map_waypoint.begin(), map_waypoint.end(), closest_waypoint); 
//         if (it != map_waypoint.end())
//         { 
//             std::cout << "Element " << closest_waypoint <<" found at position : " ; 
//             std:: cout << it - map_waypoint.begin() + 1 << "\n" ; 
//         }

//         for(auto iter = it+1; iter != map_waypoint.end(); iter++)
//         {
//             waypointBuffer.push_back(*it);
//         }
//     }
// }