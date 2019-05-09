// Declaration for common base class of all pipeline stages

#include <thread>
#include <mutex>
#include <queue>
#include "PipelineMessage.hpp"
#include "PipelineThread.hpp"

class PipelineStage
{
private:
    const std::queue<PipelineMessage>* input_queue;
    const std::queue<PipelineMessage>* output_queue;
    std::vector<std::thread> threads;
    const std::mutex& mutex;
    void runThread();

protected:
    const int pool_size;
    std::vector<PipelineCallable> threadCallables;
    virtual void createPipelineCallables()=0;

public:
    PipelineStage(
        int pool_size,
        const std::queue<PipelineMessage>* input_queue,
        const std::queue<PipelineMessage>* output_queue);
    ~PipelineStage();
};
