// //Declaration of ActorStateCallable class members
// #pragma once

// #include "carla/client/Actor.h"
// #include "carla/client/ActorList.h"
// #include "ActorStateStage.hpp"
// #include "PipelineCallable.hpp"
// #include "ActorStateMessage.hpp"

// namespace traffic_manager{

// class ActorStateCallable: public PipelineCallable
// {

// public:
//     ActorStateCallable(
//         std::queue<PipelineMessage>* input_queue,
//         std::queue<PipelineMessage>* output_queue,
//         std::mutex& read_mutex,
//         std::mutex& write_mutex,
//         int output_buffer_size);
//     ~ActorStateCallable();

//     PipelineMessage action(PipelineMessage message);
// };

// }