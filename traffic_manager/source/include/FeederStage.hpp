//Declaration of FeederStage class members 
#pragma once
#include "PipelineStage.hpp"
#include "FeederCallable.hpp"

namespace traffic_manager
{
    class FeederStage: public PipelineStage
    {
    private:
        RegisteredActorMessage* reg_actor;
    public:
        FeederStage(RegisteredActorMessage* reg_actor,
            int output_buffer_size,
            std::queue<PipelineMessage>* input_queue,
            std::queue<PipelineMessage>* output_queue);
        ~FeederStage();
        void createPipelineCallables();
    };
}

