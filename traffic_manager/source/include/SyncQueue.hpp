#pragma once
#include <mutex>
#include <condition_variable>
#include <queue>

namespace traffic_manager {

template <typename T>
class SyncQueue
{

private:
    std::mutex              q_mutex;
    std::condition_variable empty_condition;
    std::condition_variable full_condition;
    std::queue<T>           queue;
    int                     buffer_size;

public:
    SyncQueue(int buffer_size);
    void push(T value);
    T pop();
};

}
