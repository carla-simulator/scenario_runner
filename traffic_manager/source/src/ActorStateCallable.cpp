// Member Defination for Class ActorStateCallable

#include "ActorStateCallable.hpp"

namespace traffic_manager {

    ActorStateCallable::ActorStateCallable(carla::SharedPtr<carla::client::ActorList> _actor_list, int _actor_id)
    {
        this->_actor_list = _actor_list;
        this->_actor_id = _actor_id;
    }

    ActorStateCallable::~ActorStateCallable(){}

    carla::geom::Transform ActorStateCallable::getActorTransform(){
        // ActorReadState* all_actors_state;
        // std::vector<carla::geom::Transform> all_actor_transform = all_actors_state->getTransform(_actor_list);
        // std::vector<carla::geom::Transform>::iterator it;
        // it = std::find (all_actor_transform.begin(), all_actor_transform.end(), actor_id);
        // if(it != all_actor_transform.end())
        // {
        //     this->actor_transform = *it;
        return _actor_transform;
        // }
        //_actor_list->Find(actor_id)->GetTransform();
    }

    void ActorStateCallable::setActorTransform(carla::geom::Transform _actor_transform){
        this->_actor_transform = _actor_transform;
    }
    
    PipelineMessage ActorStateCallable::action(PipelineMessage message)
    {
        ActorReadStage* all_actors_state;
        std::vector<carla::geom::Transform> all_actor_transform = all_actors_state->getTransform(_actor_list);
        std::vector<carla::geom::Transform>::iterator it;
        it = std::find (all_actor_transform.begin(), all_actor_transform.end(), _actor_id);
        if(it != all_actor_transform.end())
        {
            this->_actor_transform = *it;
            return (PipelineMessage*)_actor_transform;
        }

 
    }
}