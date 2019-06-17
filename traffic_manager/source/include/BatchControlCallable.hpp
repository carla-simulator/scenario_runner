// // Declaration of class memebers

// #pragma once
// #include "carla/rpc/Command.h"
// #include "carla/client/Client.h"
// #include "carla/rpc/VehicleControl.h"
// #include "PipelineCallable.hpp"
// #include "carla/client/Vehicle.h"
// #include "carla/client/detail/Client.h"

// namespace traffic_manager{
//     class BatchControlCallable: public PipelineCallable
//     {
//     private:
//         int batch_size;
//         SyncQueue<PipelineMessage>* input_queue;
//     public:
//         BatchControlCallable(int batch_size, SyncQueue<PipelineMessage>* input_queue,
//             SyncQueue<PipelineMessage>* output_queue);
//         ~BatchControlCallable();
//         PipelineMessage action(PipelineMessage message);       
//     };
// }