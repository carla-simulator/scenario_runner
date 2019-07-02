// Member Defination for Class TrafficLightStateCallable

#include "TrafficLightStateCallable.hpp"

namespace traffic_manager {

    TrafficLightStateCallable::TrafficLightStateCallable(
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue):
        PipelineCallable(input_queue, output_queue, NULL){}

    TrafficLightStateCallable::~TrafficLightStateCallable(){}

    PipelineMessage TrafficLightStateCallable::action(PipelineMessage in_message)
    {
        PipelineMessage out_message;

        float throttle = in_message.getAttribute("throttle");
        float brake = in_message.getAttribute("brake");
        float steer = in_message.getAttribute("steer");

        auto vehicle = boost::static_pointer_cast<carla::client::Vehicle>(in_message.getActor());
        auto traffic_light_state = vehicle->GetTrafficLightState();

        if(traffic_light_state == carla::rpc::TrafficLightState::Red)
        {
            throttle = 0.0;
            brake = 1.0;          
        }
        else if(traffic_light_state == carla::rpc::TrafficLightState::Yellow)
        {
            throttle = throttle / 2;
        }

        out_message.setActor(in_message.getActor());
        out_message.setAttribute("throttle", throttle);
        out_message.setAttribute("brake", brake);
        out_message.setAttribute("steer", steer);

        return out_message;
    }
}