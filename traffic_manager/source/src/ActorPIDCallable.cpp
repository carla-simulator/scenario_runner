//Definition of class members

#include "ActorPIDCallable.hpp"

namespace traffic_manager{

    ActorPIDCallable::ActorPIDCallable (
        float target_velocity,
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue,
        SharedData* shared_data,
        std::vector<float> vpid = {0.1f, 0.15f, 0.01f}, // This is a good tune for most cars
        std::vector<float> spid = {5.0f, 0.1f, 1.0f} // Pretty stable, still needs improvement
    ): vpid(vpid), spid(spid), shared_data(shared_data),
    target_velocity(target_velocity),
    PipelineCallable(input_queue, output_queue, NULL){}

    ActorPIDCallable::~ActorPIDCallable(){}

    PipelineMessage ActorPIDCallable::action(PipelineMessage message){
        PipelineMessage out_message;

        // Initializing present state
        float current_velocity = message.getAttribute("velocity");
        float current_deviation = message.getAttribute("deviation");
        auto current_time = std::chrono::system_clock::now();
        auto actor_id = message.getActorID();
        traffic_manager::StateEntry current_state = {
            current_deviation,
            (current_velocity - target_velocity) / target_velocity,
            current_time,
            0,
            0
        };

        // Retreiving previous state
        traffic_manager::StateEntry previous_state;
        if (shared_data->state_map.find(actor_id) != shared_data->state_map.end()) {
            previous_state = shared_data->state_map[actor_id];
        } else {
            previous_state = current_state;
        }

        // Calculating dt for 'D' and 'I' controller components
        std::chrono::duration<double> duration = current_state.time_instance - previous_state.time_instance;
        auto dt = duration.count();

        // Calculating integrals
        current_state.deviation_integral =
            current_deviation*dt
            + previous_state.deviation_integral;
        current_state.velocity_integral = 
            dt * current_state.velocity
            + previous_state.velocity_integral;

        // Longitudinal PID calculation
        float max_throttle = 0.8;
        float max_brake = 0.2;
        float expr_v =
            vpid[0] * current_state.velocity
            + vpid[1] * current_state.velocity_integral
            + vpid[2] * (current_state.velocity - previous_state.velocity) / dt;

        float throttle;
        float brake;

        if(expr_v < 0.0){
            throttle = std::min(std::abs(expr_v), max_throttle);
            brake = 0.0;
        }
        else{
            throttle = 0.0;
            brake = max_brake;
        }

        // Lateral PID calculation
        float steer;
        steer =
            spid[0] * current_deviation
            + spid[1] * current_state.deviation_integral
            + spid[2] * (current_state.deviation - previous_state.deviation) / dt;
        steer = std::max(-1.0f, std::min(steer, 1.0f));

        // In case of collision or traffic light
        if (
            message.getAttribute("collision") > 0
            or message.getAttribute("traffic_light") > 0
        ) {
            current_state.deviation_integral = 0;
            current_state.velocity_integral = 0;
            throttle = 0;
            brake = 1.0;
        }

        // Updating state
        shared_data->state_map[actor_id] = current_state;

        // Constructing actuation signal
        out_message.setActor(message.getActor());
        out_message.setAttribute("throttle", throttle);
        out_message.setAttribute("brake", brake);
        out_message.setAttribute("steer", steer);
        
        return out_message;
    }
}
