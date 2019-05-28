//Declaration of ActorLocalizationCallable class members
#pragma once

#include <map>
#include "carla/geom/Location.h"
#include "PipelineCallable.hpp"
#include "SimpleWaypoint.hpp"

namespace traffic_manager{

    class ActorLocalizationCallable: public PipelineCallable
    {
        public:
        ActorLocalizationCallable(
            SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue,
            SharedData* shared_data);
        ~ActorLocalizationCallable();

        PipelineMessage action (PipelineMessage message);
    };
}