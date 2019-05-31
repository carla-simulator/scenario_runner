// Declaration of class memebers

#pragma once
#include "carla/rpc/Command.h"
#include "carla/client/Client.h"
#include "carla/rpc/VehicleControl.h"
#include "PipelineCallable.hpp"

namespace traffic_manager{
    class BatchControlCallable: public PipelineCallable
    {
    private:
        int batch_size;
        SyncQueue<PipelineMessage>* input_queue;
    public:
        BatchControlCallable(int batch_size, SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue,
            SharedData* shared_data);
        ~BatchControlCallable();
        PipelineMessage action(PipelineMessage message);       
    };
}