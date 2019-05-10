// Declaration for base class of all pipeline threads

#include <queue>
#include <mutex>
#include "PipelineMessage.hpp"

namespace traffic_manager {

class PipelineCallable
{
private:
    std::queue<PipelineMessage>* const input_queue;
    std::queue<PipelineMessage>* const output_queue;
    std::mutex& read_mutex;
    std::mutex& write_mutex;
    const int output_buffer_size;
    
    PipelineMessage readQueue();
    void writeQueue(PipelineMessage);

protected:
    virtual PipelineMessage action(PipelineMessage message)=0;

public:
    PipelineCallable(
        std::queue<PipelineMessage>* input_queue,
        std::queue<PipelineMessage>* output_queue,
        std::mutex& read_mutex,
        std::mutex& write_mutex,
        int output_buffer_size
        );
    ~PipelineCallable();
    void operator()();
};

}