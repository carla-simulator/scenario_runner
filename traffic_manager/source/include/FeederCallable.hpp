//Declaration of FeederCallable class members

#include "PipelineCallable.hpp"
#include "RegisteredActorMessage.hpp"
#include <mutex>
#include "carla/client/Actor.h"

namespace traffic_manager{

    class Feedercallable: public PipelineCallable
    {
        private:
        RegisteredActorMessage* reg_actor;  
        public:
        std::mutex inmutex;
        std::mutex outmutex;
        PipelineMessage action (PipelineMessage message);
        Feedercallable(RegisteredActorMessage* reg_actor);
        ~Feedercallable();
    };
}