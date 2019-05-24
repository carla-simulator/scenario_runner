// Declaration for common base class of all pipeline stages
#pragma once
#include <thread> 
#include <mutex>
#include <queue>
#include "PipelineCallable.hpp"

namespace traffic_manager {

class PipelineStage
{
private:
    std::vector<std::thread> threads;
    void runThreads();

protected:
    std::queue<PipelineMessage>* const input_queue;
    std::queue<PipelineMessage>* const output_queue;
    std::mutex read_mutex;
    std::mutex write_mutex;
    const int pool_size;
    const int output_buffer_size;
    PipelineMessage* shared_data;
    std::vector<PipelineCallable*> threadCallables;
    virtual void createPipelineCallables()=0;

public:
    PipelineStage(
        int pool_size, int output_buffer_size,
        std::queue<PipelineMessage>* input_queue,
        std::queue<PipelineMessage>* output_queue,
        PipelineMessage* shared_data);
    virtual ~PipelineStage();
    void start();
};

}