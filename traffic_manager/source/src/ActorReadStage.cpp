
#include "ActorReadStage.hpp"

namespace traffic_manager {

    ActorReadStage::ActorReadStage(carla::SharedPtr<carla::client::ActorList> _actor_list):PipelineStage(0,0, NULL, NULL, NULL)
    {
        this->_actor_list = _actor_list;
    }
    
    ActorReadStage::~ActorReadStage(){}
    
    std::vector<carla::geom::Transform> ActorReadStage::getTransform(carla::SharedPtr<carla::client::ActorList> _actor_list)
    {
        std::vector<carla::geom::Transform> all_actor_transform;   
        for(auto  it = _actor_list->begin() ; it != _actor_list->end(); it++ )
        {
            auto actor_transform = (*it)->GetTransform();
            all_actor_transform.push_back(actor_transform);
        }
        return all_actor_transform;   
    }
   
    void ActorReadStage::createPipelineCallables()
    {
       ActorReadStage* actor_state;
       threadCallables.push_back((PipelineCallable*)actor_state);
    }
}
