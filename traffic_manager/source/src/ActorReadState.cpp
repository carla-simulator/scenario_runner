
#include "ActorReadState.hpp"

namespace traffic_manager {

    ActorReadState::ActorReadState(carla::SharedPtr<carla::client::ActorList> _actor_list)
    {
        this->_actor_list = _actor_list;
    }
    
    ActorReadState::~ActorReadState(){}
    
    std::vector<carla::geom::Transform> ActorReadState::getTransform(carla::SharedPtr<carla::client::ActorList> _actor_list)
    {
        std::vector<carla::geom::Transform> all_actor_transform;   
        for(auto  it = _actor_list->begin() ; it != _actor_list->end(); it++ )
        {
            auto actor_transform = (*it)->GetTransform();
            all_actor_transform.push_back(actor_transform);
        }
        return all_actor_transform;   
    }
   
    // void createPipelineCallables()
    // {
    //     return ActorReadState();
    // }
    // PipelineMessage action(PipelineMessage message)
    // {
        
    // }
}
