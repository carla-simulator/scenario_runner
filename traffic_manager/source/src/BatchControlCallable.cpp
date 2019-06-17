// Definition of class memebers

#include "BatchControlCallable.hpp"

namespace traffic_manager{
    BatchControlCallable::BatchControlCallable(
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue,
        SharedData* shared_data):
        input_queue(input_queue),
        PipelineCallable(input_queue, output_queue, shared_data){}

    BatchControlCallable::~BatchControlCallable(){}

    PipelineMessage BatchControlCallable::action(PipelineMessage message)
    {
        while(true)
        {
            std::vector<carla::rpc::Command> commands;

            for (int i=0; i < shared_data->registered_actors.size(); i++) {
                carla::rpc::VehicleControl vehicle_control;

                auto element = input_queue->pop();
                auto actor = element.getActor();
                vehicle_control.throttle = element.getAttribute("throttle");
                vehicle_control.brake = element.getAttribute("brake");
                vehicle_control.steer = element.getAttribute("steer");
                carla::rpc::Command::ApplyVehicleControl control_command(element.getActorID(), vehicle_control);
                commands.push_back(control_command);
            }

            shared_data->client->ApplyBatch(commands);
        }
        PipelineMessage empty_message;
        return empty_message;
    }

}