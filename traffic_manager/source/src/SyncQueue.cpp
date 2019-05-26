#include "SyncQueue.hpp"

namespace traffic_manager {

template <typename T>
SyncQueue<T>::SyncQueue(int buffer_size):buffer_size(buffer_size){}

template <typename T>
void SyncQueue<T>::push(T value) {
    std::unique_lock<std::mutex> lock(this->q_mutex);
    this->full_condition.wait(lock, [=]{ return !this->queue.size()>=buffer_size; });
    queue.push(value);
    this->empty_condition.notify_one();
}

template <typename T>
T SyncQueue<T>::pop() {
    std::unique_lock<std::mutex> lock(this->q_mutex);
    this->empty_condition.wait(lock, [=]{ return !this->queue.empty(); });
    T rc(std::move(this->queue.front()));
    this->queue.pop();
    this->full_condition.notify_one();
    return rc;
}

}
