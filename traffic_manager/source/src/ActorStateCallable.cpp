// Member Defination for Class ActorStateCallable

#include "ActorStateCallable.hpp"

namespace traffic_manager {

    ActorStateCallable::ActorStateCallable(
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue):
        PipelineCallable(input_queue, output_queue, NULL){}

    ActorStateCallable::~ActorStateCallable(){}

    PipelineMessage ActorStateCallable::action(PipelineMessage message)
    {
        PipelineMessage out_message;
        auto actor = message.getActor();
        out_message.setActor(actor);
        auto transform = actor->GetTransform();
        out_message.setAttribute("x", transform.location.x);
        out_message.setAttribute("y", transform.location.y);
        out_message.setAttribute("z", transform.location.z);
        out_message.setAttribute("yaw", transform.rotation.yaw);
        out_message.setAttribute("velocity", actor->GetVelocity().Length());
        return out_message;
    }
}