//Declaration of ActorStateMessage class members

#include "PipelineMessage.hpp"


namespace traffic_manager{

    class ActorStateMessage: public PipelineMessage
    {
    private:
        PipelineMessage in_message;
        PipelineMessage out_message;
    public:
        ActorStateMessage(PipelineMessage in_message, PipelineMessage out_message);
        ~ActorStateMessage();
        //PipelineMessage getStageMessage(PipelineMessage in_message);    
    };
}