
#include "ActorStateStage.hpp"

namespace traffic_manager {

    ActorStateStage::ActorStateStage(ActorStateMessage* actorstate_msg, int output_buffer_size,
        std::queue<PipelineMessage>* input_queue,
        std::queue<PipelineMessage>* output_queue):actorstate_msg(actorstate_msg),
        PipelineStage(1, output_buffer_size,
        input_queue,output_queue,actorstate_msg){}
    
    ActorStateStage::~ActorStateStage(){}
    
    // std::vector<carla::geom::Transform> ActorStateStage::getTransform(
    //     carla::SharedPtr<carla::client::ActorList> _actor_list)
    // {
    //     std::vector<carla::geom::Transform> all_actor_transform;   
    //     for(auto  it = _actor_list->begin() ; it != _actor_list->end(); it++ )
    //     {
    //         auto actor_transform = (*it)->GetTransform();
    //         all_actor_transform.push_back(actor_transform);
    //     }
    //     return all_actor_transform;   
    // }
   
    void ActorStateStage::createPipelineCallables()
    {
        ActorStateCallable* actorstate_callable = new ActorStateCallable(actorstate_msg, 
        input_queue, output_queue, 
        read_mutex, write_mutex, 
        output_buffer_size);
       threadCallables.push_back(actorstate_callable);
    }
}
