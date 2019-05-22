// Declaration of RegisteredActorMessage class

#include "PipelineMessage.hpp"

namespace traffic_manager
{
    class RegisteredActorMessage: public PipelineMessage
    {
        public:
        std::vector<int> shared_actor_list;
        RegisteredActorMessage();
        ~RegisteredActorMessage();
        void addActorID(int actor_id);
        void removeActorID(int actor_id);
    };
}