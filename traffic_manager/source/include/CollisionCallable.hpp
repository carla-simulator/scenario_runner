 //Declaration of CollisionCallable class member

#pragma once

#include <cmath>
#include <deque>
#include <string>
#include <vector>
#include <algorithm>

#include "boost/geometry.hpp"
#include "boost/geometry/geometries/point_xy.hpp"
#include "boost/geometry/geometries/polygon.hpp"
#include "boost/foreach.hpp"
#include "boost/pointer_cast.hpp"
#include "carla/client/Vehicle.h"
#include "carla/geom/Location.h"
#include "carla/geom/Vector3D.h"

#include "Rectangle.hpp"
#include "PipelineCallable.hpp"

namespace traffic_manager
{
    typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > polygon;

    class CollisionCallable : public PipelineCallable
    {
        private:
            void drawBoundary(std::vector<carla::geom::Location>);
            bool checkCollisionByDistance (carla::SharedPtr<carla::client::Actor> vehicle , carla::SharedPtr<carla::client::Actor> ego_vehicle);
            bool checkGeodesicCollision(carla::SharedPtr<carla::client::Actor> vehicle , carla::SharedPtr<carla::client::Actor> ego_vehicle);
            std::vector<carla::geom::Location> getBoundary (carla::SharedPtr<carla::client::Actor> actor);
            std::vector<carla::geom::Location> getGeodesicBoundary (
                carla::SharedPtr<carla::client::Actor> actor, std::vector<carla::geom::Location> bbox);
            polygon getPolygon(std::vector<carla::geom::Location> boundary);
        
        public:

            CollisionCallable(
                SyncQueue<PipelineMessage>* input_queue,
                SyncQueue<PipelineMessage>* output_queue,
                SharedData* shared_data);
            ~CollisionCallable();

            PipelineMessage action (PipelineMessage message);
    };

}