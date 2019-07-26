//Definition of class members

#include "MotionPlannerCallable.hpp"

namespace traffic_manager{

    MotionPlannerCallable::MotionPlannerCallable (
        float target_velocity,
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue,
        SharedData* shared_data,
        std::vector<float> longitudinal_parameters = {0.1f, 0.15f, 0.01f}, // This is a good tune for most cars
        std::vector<float> lateral_parameters = {10.0f, 0.0f, 0.1f} // Pretty stable, still needs improvement
    ): longitudinal_parameters(longitudinal_parameters),
    lateral_parameters(lateral_parameters),
    shared_data(shared_data),
    target_velocity(target_velocity),
    PipelineCallable(input_queue, output_queue, NULL) {}

    MotionPlannerCallable::~MotionPlannerCallable(){}

    PipelineMessage MotionPlannerCallable::action(PipelineMessage message){
        PipelineMessage out_message;
        float current_velocity = message.getAttribute("velocity");
        float current_deviation = message.getAttribute("deviation");
        auto current_time = std::chrono::system_clock::now();
        auto actor_id = message.getActorID();

        // Retreiving previous state
        traffic_manager::StateEntry previous_state;
        if (shared_data->state_map.find(actor_id) != shared_data->state_map.end()) {
            previous_state = shared_data->state_map[actor_id];
        } else {
            previous_state.time_instance = current_time;
        }

        // Slow down upon approaching a junction
        bool approaching_junction = false;
        auto dynamic_target_velocity = target_velocity;
        int junction_index = std::max(std::floor(std::sqrt(target_velocity*current_velocity)), 5.0f);
        if (
            shared_data->buffer_map[actor_id]->get(junction_index)->checkJunction()
            and !(shared_data->buffer_map[actor_id]->get(1)->checkJunction())
        ) {
            dynamic_target_velocity = 3.0f; // 10.8 kmph, Account for constant
            approaching_junction = true;
        }

        // State update for vehicle
        auto current_state = controller.stateUpdate(
            previous_state,
            current_velocity,
            dynamic_target_velocity,
            current_deviation,
            current_time
        );

        // Controller actuation
        auto actuation_signal = controller.runStep(
            current_state,
            previous_state,
            longitudinal_parameters,
            lateral_parameters
        );

        // In case of collision or traffic light or approaching a junction
        if (
            message.getAttribute("collision") > 0
            or message.getAttribute("traffic_light") > 0
            or (approaching_junction and current_velocity > 3.0) // Account for constant
        ) {
            current_state.deviation_integral = 0;
            current_state.velocity_integral = 0;
            actuation_signal.throttle = 0;
            actuation_signal.brake = 1.0;
        }

        // Updating state
        shared_data->state_map[actor_id] = current_state;

        // Constructing actuation signal
        out_message.setActor(message.getActor());
        out_message.setAttribute("throttle", actuation_signal.throttle);
        out_message.setAttribute("brake", actuation_signal.brake);
        out_message.setAttribute("steer", actuation_signal.steer);
        
        return out_message;
    }
}
