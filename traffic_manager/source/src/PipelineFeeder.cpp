// Implementation of PipelineFeeder class

#include "PipelineFeeder.hpp"
#include <boost/interprocess/ipc/message_queue.hpp>

using namespace boost::interprocess;

namespace traffic_manager
{

class PipelineFeeder
{
 public:
  PipelineFeeder();
  ~PipelineFeeder();

 private:
  message_queue* queue_;
  std::size_t get_max_msg() const;
  std::size_t get_max_msg_size() const;
};

PipelineFeeder::PipelineFeeder() : queue_(nullptr) 
{
  queue_ = new message_queue(open_or_create, "PipelineFeeder",  get_max_msg(), get_max_msg_size());
}

PipelineFeeder::~PipelineFeeder() {}

}