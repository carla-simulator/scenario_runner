#include "FeederStage.hpp"

namespace traffic_manager
{
    FeederStage::FeederStage(RegisteredActorMessage* reg_actor):PipelineStage(1,20,NULL,NULL,nullptr){
        this->reg_actor = reg_actor;
    }

    FeederStage::~FeederStage(){}

    void FeederStage::createPipelineCallables()
    {
        Feedercallable* feeder_callable = new Feedercallable(reg_actor);
        threadCallables.push_back(feeder_callable);
    } 
}