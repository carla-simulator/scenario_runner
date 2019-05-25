// // Declaration of class to localise actor
// #pragma once

// #include "PipelineStage.hpp"
// #include <vector>
// #include "carla/Memory.h"
// #include "carla/geom/Transform.h"
// #include "carla/client/ActorList.h"
// #include "InMemoryMap.hpp"

// namespace traffic_manager {

//     class ActorLocalizationStage//: public PipelineStage
//     {

//     private:
//         carla::geom::Transform actor_transform;
//         carla::SharedPtr<carla::client::ActorList> _actor_list;

//     public:
//         ActorLocalizationStage(carla::geom::Transform actor_transform);
//         ~ActorLocalizationStage();
//         carla::geom::Location getLocation(carla::geom::Transform actor_transform);
//         std::vector<SimpleWaypoint*> getWaypointBuffer(carla::geom::Location current_location);
//     };
// } //namespace traffic_manager