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
        if(shared_data->buffer_map.find(actor_id) != shared_data->buffer_map.end()){
            float dot_product = -1;
            while(dot_product <= 0){
                auto wp = shared_data->buffer_map[actor_id].front();
                auto heading_direction = message.getActor()->GetTransform().rotation.GetForwardVector();
                auto next_coordinate = wp->getXYZ();
                float x = message.getAttribute("x");
                float y = message.getAttribute("y");
                float z = message.getAttribute("z");
                std::vector<float> actor_coordinate = {x, y, z};
                std::vector<float> heading_coordinate = {heading_direction.x, heading_direction.y, heading_direction.z};
                for(int i = 0; i < 3; i++)
                {
                    next_coordinate[i] = next_coordinate[i] - actor_coordinate[i];
                    dot_product += next_coordinate[i]*heading_coordinate[i];
                }
                if(dot_product <= 0)
                {
                    shared_data->buffer_map[actor_id].pop();
                    auto feed_waypoint =shared_data->buffer_map[actor_id].back()->getNextWaypoint()[0];
                    while(!shared_data->buffer_map[actor_id].full())
                    {
                        shared_data->buffer_map[actor_id].push(feed_waypoint);
                    }
                }
            }
        }
        else
        {
            auto actor_location = carla::geom::Location(
                message.getAttribute("x"), 
                message.getAttribute("y"), 
                message.getAttribute("z"));
            auto closest_waypoint = shared_data->memory_map->getWaypoint(actor_location);
            while(!shared_data->buffer_map[actor_id].full())
            {
                shared_data->buffer_map[actor_id].push(closest_waypoint);
                closest_waypoint = closest_waypoint->getNextWaypoint()[0];                
            }
        }

        PipelineMessage empty_message;
        return empty_message;
    }

}