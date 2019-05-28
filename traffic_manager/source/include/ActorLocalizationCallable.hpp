//Declaration of ActorLocalizationCallable class members
#pragma once

#include <map>
#include "carla/geom/Location.h"
#include "PipelineCallable.hpp"
#include "LocalMapMessage.hpp"
#include "SimpleWaypoint.hpp"

namespace traffic_manager{

    class ActorLocalizationCallable: public PipelineCallable
    {
        private:
            std::map<int , SyncQueue<SimpleWaypoint*>> buffer_map;
            LocalMapMessage* shared_data;
        public:
        ActorLocalizationCallable(
            SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue,
            LocalMapMessage* shared_data);
        ~ActorLocalizationCallable();

        PipelineMessage action (PipelineMessage message);
    };
}