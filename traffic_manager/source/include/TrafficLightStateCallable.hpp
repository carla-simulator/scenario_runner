//Declaration of TrafficLightStateCallable class members
#pragma once

#include "PipelineCallable.hpp"
#include "carla/client/Vehicle.h"
#include "carla/rpc/TrafficLightState.h"

namespace traffic_manager{

class TrafficLightStateCallable: public PipelineCallable
{

public:
    TrafficLightStateCallable(
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue,
        SharedData* shared_data);
    ~TrafficLightStateCallable();

    PipelineMessage action(PipelineMessage message);
};

}