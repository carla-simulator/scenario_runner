// Definition file for members of class PipelineStage

#include "PipelineStage.hpp"

namespace traffic_manager {

PipelineStage::PipelineStage(
    int pool_size,
    std::queue<PipelineMessage>* const input_queue,
    std::queue<PipelineMessage>* const output_queue):
    input_queue(input_queue), output_queue(output_queue),
    pool_size(pool_size) {
    }
PipelineStage::~PipelineStage(){}

void PipelineStage::runThread(){
    for (auto threadCallable: threadCallables)
        threads.push_back(std::thread(threadCallable));
}

}