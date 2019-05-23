//Declaration of FeederStage class members 
#include "PipelineStage.hpp"
#include "FeederCallable.hpp"

namespace traffic_manager
{
    class FeederStage: public PipelineStage
    {
    private:
        RegisteredActorMessage* reg_actor;
    public:
        FeederStage(RegisteredActorMessage* reg_actor);
        ~FeederStage();
        void createPipelineCallables();
    };
}

