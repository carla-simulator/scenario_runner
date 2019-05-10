// Definition file for members of class PipelineStage

#include "PipelineStage.hpp"

namespace traffic_manager {

PipelineStage::PipelineStage(
    int pool_size, int output_buffer_size,
    std::queue<PipelineMessage>* input_queue,
    std::queue<PipelineMessage>* output_queue):
    input_queue(input_queue), output_queue(output_queue),
    pool_size(pool_size), output_buffer_size(output_buffer_size) {
    }
PipelineStage::~PipelineStage(){}

void PipelineStage::runThreads(){
    for (auto threadCallable: threadCallables)
        threads.push_back(
            std::thread(
                &PipelineCallable::run,
                threadCallable));
}

void PipelineStage::start() {
    this->createPipelineCallables();
    this->runThreads();
}

}