// Declaration for a common base class to all messages between pipeline stages
#pragma once
#include <string>
#include <map>
#include "carla/client/Actor.h"

namespace traffic_manager {

class PipelineMessage
{
private:
    int actor_id;
    carla::SharedPtr<carla::client::Actor> actor;
    std::map<std::string, float> attributes;

public:
    PipelineMessage();
    virtual ~PipelineMessage();

    void setActor(carla::SharedPtr<carla::client::Actor> actor);
    carla::SharedPtr<carla::client::Actor> getActor();
    int getActorID();

    void setAttribute(std::string, float);
    float getAttribute(std::string);
    bool hasAttribute(std::string);
    void removeAttribute(std::string);
};

}