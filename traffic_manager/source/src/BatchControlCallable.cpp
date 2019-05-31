// Definition of class memebers

#include "BatchControlCallable.hpp"

namespace traffic_manager{
    BatchControlCallable::BatchControlCallable(
        int batch_size,
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue,
        SharedData* shared_data): batch_size(batch_size), input_queue(input_queue),
        PipelineCallable(input_queue, output_queue, shared_data){}

    BatchControlCallable::~BatchControlCallable(){}

    PipelineMessage BatchControlCallable::action(PipelineMessage message)
    {
        while(true)
        {
            carla::rpc::VehicleControl vehicle_control;            
            std::vector<carla::rpc::Command> commands;

            while(!input_queue->empty()){
                auto element = input_queue->pop();
                int actor_id = element.getActorID();       
                vehicle_control.throttle = element.getAttribute("throttle");
                vehicle_control.brake = element.getAttribute("brake");
                vehicle_control.steer = element.getAttribute("steer");
                carla::rpc::Command::ApplyVehicleControl control_command(actor_id, vehicle_control);
                commands.push_back(control_command);                
            }
            if(commands.size() >= batch_size){
                carla::client::Client* client_obj;
                client_obj->ApplyBatch(commands);
                commands.empty();
            }
        }
        PipelineMessage empty_message;
        return empty_message;
    }

}