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
    SyncQueue(int buffer_size = 20):buffer_size(buffer_size){}
    
    void push(T value){
        std::unique_lock<std::mutex> lock(this->q_mutex);
        this->full_condition.wait(lock, [=]{ return !(this->queue.size()>=buffer_size); });
        queue.push(value);
        this->empty_condition.notify_one();
    }
    
    T pop(){
        std::unique_lock<std::mutex> lock(this->q_mutex);
        this->empty_condition.wait(lock, [=]{ return !this->queue.empty(); });
        T rc(std::move(this->queue.front()));
        this->queue.pop();
        this->full_condition.notify_one();
        return rc;
    }

    T front(){
        T rc(std::move(this->queue.front()));
        return rc;
    }

    T back(){
        T rc(std::move(this->queue.back()));
        return rc;
    }

    int size() {
        return queue.size();
    }

    bool empty() {
        return queue.empty();
    }

    bool full() {
        return queue.size() >= buffer_size;
    }
};

}
