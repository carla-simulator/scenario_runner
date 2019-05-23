//Declaration of ActorStateMessage class members

#include "PipelineMessage.hpp"


namespace traffic_manager{

    class ActorStateMessage: public PipelineMessage
    {
    private:
        carla::geom::Transform _actor_transform;
        
    public:
        ActorStateMessage(carla::geom::Transform _actor_transform);
        ~ActorStateMessage();
        //PipelineMessage getStageMessage(PipelineMessage in_message);    
    };
}