// Declaration of class memebers

#pragma once
#include <chrono>
#include "carla/rpc/Command.h"
#include "carla/rpc/VehicleControl.h"
#include "PipelineCallable.hpp"
#include "SharedData.hpp"

namespace traffic_manager{
    class BatchControlCallable: public PipelineCallable
    {
    private:
        SyncQueue<PipelineMessage>* input_queue;
    public:
        BatchControlCallable(SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue, SharedData* shared_data);
        ~BatchControlCallable();
        PipelineMessage action(PipelineMessage message);       
    };
}