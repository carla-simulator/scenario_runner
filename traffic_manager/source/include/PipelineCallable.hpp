// Declaration for base class of all pipeline threads

#include <queue>
#include <mutex>
#include "PipelineMessage.hpp"

class PipelineCallable
{
private:
    const std::queue<PipelineMessage>* input_queue;
    const std::queue<PipelineMessage>* output_queue;
    const std::mutex& mutex;
    PipelineMessage readQueue();

protected:
    virtual PipelineMessage action(PipelineMessage message);

public:
    PipelineCallable(
        const std::queue<PipelineMessage>* input_queue,
        const std::queue<PipelineMessage>* output_queue,
        const std::mutex& mutex
        );
    ~PipelineCallable();
    void operator()();
};
