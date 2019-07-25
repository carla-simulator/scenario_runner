#include "LocalizationCallable.hpp"

namespace traffic_manager
{
    LocalizationCallable::LocalizationCallable(
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue,
        SharedData* shared_data):PipelineCallable(input_queue, output_queue, shared_data){}

    LocalizationCallable::~LocalizationCallable(){}

    PipelineMessage LocalizationCallable::action(PipelineMessage message)
    {   
        auto vehicle = message.getActor();
        auto actor_id = message.getActorID();
        auto vehicle_location = vehicle->GetLocation();
        auto vehicle_velocity = vehicle->GetVelocity().Length();
        
        if(
            shared_data->buffer_map.find(actor_id) != shared_data->buffer_map.end()
            and !shared_data->buffer_map[actor_id]->empty()
        ) {
            // Existing actor in buffer map

            float nearest_distance = 10;
            float dot_product = 1;
            nearest_distance = nearestDistance(vehicle);
            dot_product = nearestDotProduct(vehicle);

            // Purge past waypoints
            auto distance_threshold = std::max(vehicle_velocity * 0.5, 2.0);
            while ((dot_product <= 0 || nearest_distance <= distance_threshold)) {
                shared_data->buffer_map[actor_id]->pop();
                if (!shared_data->buffer_map[actor_id]->empty()) {
                    dot_product = nearestDotProduct(vehicle);
                    nearest_distance = nearestDistance(vehicle);
                } else {break;}
            }

            // Re-initialize buffer if empty
            if (shared_data->buffer_map[actor_id]->empty()) {
                auto closest_waypoint = shared_data->local_map->getWaypoint(vehicle_location);
                shared_data->buffer_map[actor_id]->push(closest_waypoint);
            }

            // Re-populate buffer
            while (
                shared_data->buffer_map[actor_id]->back()->distance(
                    shared_data->buffer_map[actor_id]->front()->getLocation()
                ) <= 20.0 // Make this a constant
            ) {
                auto next_waypoints = shared_data->buffer_map[actor_id]->back()->getNextWaypoint();
                auto selection_index = next_waypoints.size() > 1 ? rand()%next_waypoints.size() : 0;
                auto feed_waypoint = next_waypoints[selection_index];
                shared_data->buffer_map[actor_id]->push(feed_waypoint);
            }
        }
        else
        {
            // New actor to buffer map

            // Make size of queue a derived or constant
            shared_data->buffer_map[actor_id] = std::make_shared<SyncQueue<std::shared_ptr<SimpleWaypoint>>>(200);
            auto closest_waypoint = shared_data->local_map->getWaypoint(vehicle_location);
            // Initialize buffer for actor
            shared_data->buffer_map[actor_id]->push(closest_waypoint);
            // Populate buffer
            while (
                shared_data->buffer_map[actor_id]->back()->distance(
                    shared_data->buffer_map[actor_id]->front()->getLocation()
                ) <= 20.0 // Make this a constant
            ) {
                auto next_waypoints = closest_waypoint->getNextWaypoint();
                auto selection_index = next_waypoints.size() > 1 ? rand()%next_waypoints.size() : 0;
                closest_waypoint = next_waypoints[selection_index];
                shared_data->buffer_map[actor_id]->push(closest_waypoint);
            }
        }

        // Generate output message
        PipelineMessage out_message;
        float dot_product = nearestDotProduct(vehicle);
        float cross_product = nearestCrossProduct(vehicle);
        dot_product = 1 - dot_product;
        if(cross_product < 0)
            dot_product *= -1;
        out_message.setActor(message.getActor());
        out_message.setAttribute("velocity", vehicle_velocity);
        out_message.setAttribute("deviation", dot_product);

        return out_message;
    }
    
    float LocalizationCallable::nearestDotProduct(carla::SharedPtr<carla::client::Actor> actor) {
        auto wp = shared_data->buffer_map[actor->GetId()]->front();
        auto next_location = wp->getLocation();
        auto heading_vector = actor->GetTransform().GetForwardVector();
        auto next_vector = next_location - actor->GetLocation(); 
        next_vector = next_vector.MakeUnitVector();
        auto dot_product = next_vector.x*heading_vector.x +
            next_vector.y*heading_vector.y + next_vector.z*heading_vector.z;
        return dot_product;
    }

    float LocalizationCallable::nearestCrossProduct(carla::SharedPtr<carla::client::Actor> actor){
        auto wp = shared_data->buffer_map[actor->GetId()]->front();
        auto next_location = wp->getLocation();
        auto heading_vector = actor->GetTransform().GetForwardVector();
        auto next_vector = next_location - actor->GetLocation(); 
        next_vector = next_vector.MakeUnitVector();
        float cross_z = heading_vector.x*next_vector.y - heading_vector.y*next_vector.x;
        return cross_z;
    }

    float LocalizationCallable::nearestDistance(carla::SharedPtr<carla::client::Actor> actor){
        auto wp = shared_data->buffer_map[actor->GetId()]->front();
        return wp->distance(actor->GetLocation());
    }

}