//Definition of class members

#include "ActorPIDCallable.hpp"

namespace traffic_manager{

    ActorPIDCallable::ActorPIDCallable(float k_v, float k_s, float target_velocity,
            SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue): k_v(k_v), k_s(k_s),
            target_velocity(target_velocity),
            PipelineCallable(input_queue, output_queue, NULL){}

    ActorPIDCallable::~ActorPIDCallable(){}

    PipelineMessage ActorPIDCallable::action(PipelineMessage message){
        PipelineMessage out_message;
        float current_velocity = message.getAttribute("velocity");
        float deviation = message.getAttribute("deviation");

        float max_throttle = 1.0;
        float expr_v = k_v*((target_velocity - current_velocity) / target_velocity);
        
        float throttle;
        float brake;
        float steer;

        if(expr_v > 0.0){
            throttle = std::max(expr_v, max_throttle);
            brake = 0.0;
        }
        else{
            throttle = 0.0;
            brake = std::max(std::abs(expr_v), max_throttle);
        }
        steer = k_s*deviation;
        
        out_message.setActor(message.getActor());
        out_message.setAttribute("throttle", throttle);
        out_message.setAttribute("brake", brake);
        out_message.setAttribute("steer", steer);
        
        return out_message;
    }
}   