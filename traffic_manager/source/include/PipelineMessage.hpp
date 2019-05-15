// Declaration for a common base class to all messages between pipeline stages
#include "carla/client/Actor.h"

namespace traffic_manager {

class PipelineMessage
{
private:
   int actor_id;
public:
    PipelineMessage();
    virtual ~PipelineMessage();
    int getActorID();
    void setActorID( int actor_id);
};

}