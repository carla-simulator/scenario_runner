//Definition of class members

#include "ActorPIDCallable.hpp"

namespace traffic_manager{

    ActorPIDCallable::ActorPIDCallable(float k_v, float k_s, float target_velocity,
            SyncQueue<PipelineMessage>* input_queue,
            SyncQueue<PipelineMessage>* output_queue): k_v(k_v), k_s(k_s),
            target_velocity(target_velocity),
            PipelineCallable(input_queue, output_queue,NULL){}

    ActorPIDCallable::~ActorPIDCallable(){}

    PipelineMessage ActorPIDCallable::action(PipelineMessage message){
        PipelineMessage out_message;
        float current_velocity = message.getAttribute("velocity");
        float deviation = message.getAttribute("deviation");

        float max_throttle = 1.0;
        float expr_v = k_v*((target_velocity - current_velocity) / target_velocity);
        carla::rpc::VehicleControl actor_control;        
        if(expr_v > 0.0){
            actor_control.throttle = std::max(expr_v, max_throttle);
            actor_control.brake = 0.0;
        }
        else{
            actor_control.throttle = 0.0;
            actor_control.brake = std::max(std::abs(expr_v), max_throttle);
        }
        actor_control.steer = k_s*deviation;
        out_message.setActor(message.getActor());
        out_message.setAttribute("throttle", actor_control.throttle);
        out_message.setAttribute("brake", actor_control.brake);
        out_message.setAttribute("steer", actor_control.steer);
        
        return out_message;
    }
}   