#include "ActorLocalizationCallable.hpp"

namespace traffic_manager
{
    ActorLocalizationCallable::ActorLocalizationCallable(
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue,
        SharedData* shared_data):PipelineCallable(input_queue, output_queue, shared_data){}

    ActorLocalizationCallable::~ActorLocalizationCallable(){}

    PipelineMessage ActorLocalizationCallable::action(PipelineMessage message)
    {   
        int actor_id = message.getActorID();
        if(shared_data->buffer_map.find(actor_id) != shared_data->buffer_map.end()
            and !shared_data->buffer_map[actor_id]->empty()){
            // Existing actor in buffer map

            float nearest_distance = 10;
            float dot_product = 1;
            nearest_distance = nearestDistance(shared_data, &message);
            dot_product = nearestDotProduct(shared_data, &message);

            // Purge past waypoints
            auto distance_threshold = std::max(message.getAttribute("velocity") * 0.5, 2.0);
            while ((dot_product <= 0 || nearest_distance <= distance_threshold)) {
                shared_data->buffer_map[actor_id]->pop();
                if (!shared_data->buffer_map[actor_id]->empty()) {
                    dot_product = nearestDotProduct(shared_data, &message);
                    nearest_distance = nearestDistance(shared_data, &message);
                } else {break;}
            }

            // Re-initialize buffer if empty
            if (shared_data->buffer_map[actor_id]->empty()) {
                auto actor_location = carla::geom::Location(
                    message.getAttribute("x"),
                    message.getAttribute("y"),
                    message.getAttribute("z"));
                auto closest_waypoint = shared_data->local_map->getWaypoint(actor_location);
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
            auto actor_location = carla::geom::Location(
                message.getAttribute("x"),
                message.getAttribute("y"),
                message.getAttribute("z"));
            auto closest_waypoint = shared_data->local_map->getWaypoint(actor_location);
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
        float dot_product = nearestDotProduct(shared_data, &message);
        float cross_product = nearestCrossProduct(shared_data, &message);
        dot_product = 1 - dot_product;
        if(cross_product < 0)
            dot_product *= -1;
        out_message.setActor(message.getActor());
        out_message.setAttribute("velocity", message.getAttribute("velocity"));
        out_message.setAttribute("deviation", dot_product);

        return out_message;
    }
    
    float ActorLocalizationCallable::nearestDotProduct(SharedData* data, PipelineMessage* message){
        auto wp = shared_data->buffer_map[message->getActorID()]->front();
        auto next_coordinate = wp->getXYZ();
        std::vector<float> actor_coordinate = {
            message->getAttribute("x"),
            message->getAttribute("y"),
            message->getAttribute("z")};
        std::vector<float> heading_vector = {
            message->getAttribute("heading_x"),
            message->getAttribute("heading_y"),
            message->getAttribute("heading_z")};
        float dot_product = 0;
        for(int i = 0; i < 3; i++)
        {
            next_coordinate[i] = next_coordinate[i] - actor_coordinate[i];
        }
        carla::geom::Vector3D next_vector(next_coordinate[0], next_coordinate[1], next_coordinate[2]);
        next_vector = next_vector.MakeUnitVector();
        dot_product = next_vector.x*heading_vector[0] +
            next_vector.y*heading_vector[1] + next_vector.z*heading_vector[2];
        return dot_product;
    }

    float ActorLocalizationCallable::nearestCrossProduct(SharedData* data, PipelineMessage* message){
        auto wp = shared_data->buffer_map[message->getActorID()]->front();
        auto next_coordinate = wp->getXYZ();
        std::vector<float> actor_coordinate = {
            message->getAttribute("x"),
            message->getAttribute("y"),
            message->getAttribute("z")};
        std::vector<float> heading_vector = {
            message->getAttribute("heading_x"),
            message->getAttribute("heading_y"),
            message->getAttribute("heading_z")};
        float dot_product = 0;
        for(int i = 0; i < 3; i++)
        {
            next_coordinate[i] = next_coordinate[i] - actor_coordinate[i];
        }
        carla::geom::Vector3D next_vector(next_coordinate[0], next_coordinate[1], next_coordinate[2]);
        next_vector = next_vector.MakeUnitVector();
        std::vector<float> target_vector = {next_vector.x, next_vector.y, next_vector.z};
        float cross_z = heading_vector[0]*target_vector[1] - heading_vector[1]*target_vector[0];
        return cross_z;
    }

    float ActorLocalizationCallable::nearestDistance(SharedData* data, PipelineMessage* message){
        auto wp = shared_data->buffer_map[message->getActorID()]->front();
        auto next_coordinate = wp->getXYZ();
        std::vector<float> xyz = {
            message->getAttribute("x"),
            message->getAttribute("y"),
            message->getAttribute("z")};
        auto distance = wp->distance(carla::geom::Location(xyz[0],xyz[1], xyz[2]));
        return distance;
    }

}