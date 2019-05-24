//Declaration of FeederCallable class members
#pragma once

#include "PipelineCallable.hpp"
#include "RegisteredActorMessage.hpp"
#include <mutex>
#include "carla/client/Actor.h"

namespace traffic_manager{

    class Feedercallable: public PipelineCallable
    {
        private:
        RegisteredActorMessage* reg_actor;  
        public:
        std::mutex inmutex;
        std::mutex outmutex;
        PipelineMessage action (PipelineMessage message);
        Feedercallable(RegisteredActorMessage* reg_actor,
            std::queue<PipelineMessage>* input_queue,
            std::queue<PipelineMessage>* output_queue,
            std::mutex& read_mutex,
            std::mutex& write_mutex,
            int output_buffer_size);
        ~Feedercallable();
    };
}