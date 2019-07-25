#pragma once
#include <mutex>
#include <condition_variable>
#include <deque>
#include <algorithm>
#include <vector>

namespace traffic_manager {

template <typename T>
class SyncQueue
{

private:
    std::mutex              q_mutex;
    std::condition_variable empty_condition;
    std::condition_variable full_condition;
    std::condition_variable half_condition;
    std::deque<T>           queue;
    int                     buffer_size;

public:
    SyncQueue(int buffer_size = 20):buffer_size(buffer_size){}
    
    void push(T value){
        std::unique_lock<std::mutex> lock(this->q_mutex);
        this->full_condition.wait(lock, [=]{ return !(this->queue.size()>=buffer_size); });
        queue.push_back(value);
        this->empty_condition.notify_one();
    }
    
    T pop(){
        std::unique_lock<std::mutex> lock(this->q_mutex);
        this->empty_condition.wait(lock, [=]{ return !this->queue.empty(); });
        T rc(std::move(this->queue.front()));
        this->queue.pop_front();
        this->full_condition.notify_one();
        return rc;
    }

    T front(){
        return this->queue.front();
    }

    T back(){
        return this->queue.back();
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

    std::vector<T> getContent(int number_of_elements) {
        std::unique_lock<std::mutex> lock(q_mutex);
        if (queue.size() >= number_of_elements)
            return std::vector<T>(queue.begin(), queue.begin()+number_of_elements);
        else
            return std::vector<T>(queue.begin(), queue.end());
    }

    T get(int index) {
        std::unique_lock<std::mutex> lock(q_mutex);
        auto queue_size = this->size();
        index = index >= queue_size ? queue_size: index;
        return queue.at(index);
    }
};

}
