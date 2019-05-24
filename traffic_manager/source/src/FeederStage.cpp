#include "FeederStage.hpp"

namespace traffic_manager
{
    FeederStage::FeederStage(
        RegisteredActorMessage* reg_actor,
            int output_buffer_size,
            std::queue<PipelineMessage>* input_queue,
            std::queue<PipelineMessage>* output_queue):
            reg_actor(reg_actor),PipelineStage(1,
            output_buffer_size,input_queue,output_queue,reg_actor){}

    FeederStage::~FeederStage(){}

    void FeederStage::createPipelineCallables()
    {
        std::cout << "Calling createPipelineCallables" << std::endl;
        Feedercallable* feeder_callable = new Feedercallable(reg_actor,input_queue,output_queue,read_mutex,write_mutex,output_buffer_size);
        threadCallables.push_back(feeder_callable);
    } 
}