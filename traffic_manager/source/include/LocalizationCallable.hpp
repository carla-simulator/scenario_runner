//Declaration of class members
#pragma once

#include <memory>
#include <map>
#include "carla/geom/Vector3D.h"
#include "carla/geom/Location.h"
#include "carla/geom/Transform.h"
#include "PipelineCallable.hpp"
#include "SimpleWaypoint.hpp"

namespace traffic_manager{

    class LocalizationCallable: public PipelineCallable
    {
        private:
            float nearestDotProduct(carla::SharedPtr<carla::client::Actor>);
            float nearestCrossProduct(carla::SharedPtr<carla::client::Actor>);
            float nearestDistance(carla::SharedPtr<carla::client::Actor>);
        public:
        LocalizationCallable(
            SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue,
            SharedData* shared_data);
        ~LocalizationCallable();

        PipelineMessage action (PipelineMessage message);
    };
}