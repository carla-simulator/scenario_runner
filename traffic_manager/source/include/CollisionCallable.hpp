 //Declaration of CollisionCallable class member

#pragma once

#include <cmath>
#include <deque>
#include <string>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/foreach.hpp>
#include "system/boost/pointer_cast.hpp"
#include "carla/client/Vehicle.h"
#include "Rectangle.hpp"
#include "PipelineCallable.hpp"

namespace traffic_manager
{

    class CollisionCallable : public PipelineCallable
    {
        private:
            bool checkCollisionByDistance (carla::SharedPtr<carla::client::Actor> vehicle , carla::SharedPtr<carla::client::Actor> ego_vehicle);
            bool check_rect_intersection(carla::SharedPtr<carla::client::Actor> vehicle , carla::SharedPtr<carla::client::Actor> ego_vehicle);
        public:
            CollisionCallable(
                SyncQueue<PipelineMessage>* input_queue,
                SyncQueue<PipelineMessage>* output_queue,
                SharedData* shared_data);
            ~CollisionCallable();

            PipelineMessage action (PipelineMessage message);
    };

}