//definition of Ac class members

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
        std::cout << "Inside action !" << std::endl;
        if(shared_data->buffer_map.find(actor_id) != shared_data->buffer_map.end()){
            std::cout << "Registered actor found ! " << std::endl;
            float dot_product = nearestDotProduct(shared_data, &message);
            while(dot_product <= 0){
                shared_data->buffer_map[actor_id].pop();
                dot_product = nearestDotProduct(shared_data, &message);
            }
            while(!shared_data->buffer_map[actor_id].full()) {
                auto feed_waypoint = shared_data->buffer_map[actor_id].back()->getNextWaypoint()[0];
                shared_data->buffer_map[actor_id].push(feed_waypoint);
            }
        }
        else
        {
            std::cout << "New actor found ! " << std::endl;
            auto actor_location = carla::geom::Location(
                message.getAttribute("x"), 
                message.getAttribute("y"), 
                message.getAttribute("z"));
            auto closest_waypoint = shared_data->local_map->getWaypoint(actor_location);
            while(!shared_data->buffer_map[actor_id].full())
            {   
                shared_data->buffer_map[actor_id].push(closest_waypoint);
                closest_waypoint = closest_waypoint->getNextWaypoint()[0];
                shared_data->buffer_map[actor_id].front()->getXYZ();
                std::cout <<"Able to getXYZ in while" << std::endl;
            }
        }

        PipelineMessage out_message;
        float dot_product = nearestDotProduct(shared_data, &message);
        float cross_product = nearestCrossProduct(shared_data, &message);
        if(cross_product < 0)
            dot_product *= -1;
        out_message.setAttribute("velocity", message.getAttribute("velocity"));
        out_message.setAttribute("deviation", dot_product);

        return out_message;
    }
    
    float ActorLocalizationCallable::nearestDotProduct(SharedData* data, PipelineMessage* message){
        auto wp = shared_data->buffer_map[message->getActorID()].front();
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
        auto wp = shared_data->buffer_map[message->getActorID()].front();
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

}