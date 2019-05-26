// Declaration for common base class of all pipeline stages
#pragma once
#include <thread>
#include "PipelineCallable.hpp"

namespace traffic_manager {

class PipelineStage
{
private:
    std::vector<std::thread> threads;
    void runThreads();

protected:
    const int pool_size;
    PipelineCallable& thread_callable;

public:
    PipelineStage(
        int pool_size,
        PipelineCallable& thread_callable);
    virtual ~PipelineStage();
    void start();
};

}