// Member definitions for class PipelineCallable

#include "PipelineCallable.hpp"

namespace traffic_manager {

PipelineCallable::PipelineCallable(
    std::queue<PipelineMessage>* input_queue,
    std::queue<PipelineMessage>* output_queue,
    std::mutex& read_mutex,
    std::mutex& write_mutex,
    int output_buffer_size):
    read_mutex(read_mutex), write_mutex(write_mutex),
    input_queue(input_queue), output_queue(output_queue),
    output_buffer_size(output_buffer_size){
}
PipelineCallable::~PipelineCallable(){}

PipelineMessage PipelineCallable::readQueue() {
    std::lock_guard<std::mutex> lock(read_mutex);
    while(input_queue->empty());
    PipelineMessage message = input_queue->front();
    input_queue->pop();
    return message;
}

void PipelineCallable::writeQueue(PipelineMessage message) {
    std::lock_guard<std::mutex> lock(write_mutex);
    while(input_queue->size() > output_buffer_size);
    output_queue->push(message);
}

void PipelineCallable::run() {
    PipelineMessage in_message;
    if (input_queue != NULL)
        in_message = readQueue();
    auto out_message = action(in_message);
    if (output_queue != NULL)
        writeQueue(out_message);
}

}