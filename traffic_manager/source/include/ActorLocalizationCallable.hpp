//Declaration of ActorLocalizationCallable class members
#pragma once

#include <memory>
#include <map>
#include "carla/geom/Vector3D.h"
#include "carla/geom/Location.h"
#include "PipelineCallable.hpp"
#include "SimpleWaypoint.hpp"

namespace traffic_manager{

    class ActorLocalizationCallable: public PipelineCallable
    {
        private:
            float nearestDotProduct(SharedData*, PipelineMessage*);
            float nearestCrossProduct(SharedData*, PipelineMessage*);
        public:
        ActorLocalizationCallable(
            SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue,
            SharedData* shared_data);
        ~ActorLocalizationCallable();

        PipelineMessage action (PipelineMessage message);
    };
}