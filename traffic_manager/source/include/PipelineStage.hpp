// Declaration for common base class of all pipeline stages

#include <thread>
#include <mutex>
#include <queue>
#include "PipelineMessage.hpp"
#include "PipelineCallable.hpp"

namespace traffic_manager {

class PipelineStage
{
private:
    std::queue<PipelineMessage>* const input_queue;
    std::queue<PipelineMessage>* const output_queue;
    std::vector<std::thread> threads;
    std::mutex read_mutex;
    std::mutex write_mutex;
    void runThreads();

protected:
    const int pool_size;
    const int output_buffer_size;
    std::vector<PipelineCallable> threadCallables;
    virtual void createPipelineCallables()=0;

public:
    PipelineStage(
        int pool_size, int output_buffer_size,
        std::queue<PipelineMessage>* input_queue,
        std::queue<PipelineMessage>* output_queue);
    ~PipelineStage();
    void start();
};

}