//Declaration of FeederCallable class members

#include "PipelineCallable.hpp"
#include "RegisteredActorMessage.hpp"
#include <mutex>
#include "carla/client/Actor.h"

namespace traffic_manager{

    class Feedercallable: public PipelineCallable
    {   
        public:
        std::mutex inmutex;
        std::mutex outmutex;
        PipelineMessage action (PipelineMessage message);
        Feedercallable();
        ~Feedercallable();

    };
}