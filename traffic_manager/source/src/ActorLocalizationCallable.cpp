//definition of Ac class members

#include "ActorLocalizationCallable.hpp"

namespace traffic_manager
{
    ActorLocalizationCallable::ActorLocalizationCallable(
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue,
        LocalMapMessage* shared_data): shared_data(shared_data),
        PipelineCallable(input_queue,output_queue, shared_data){}

    ActorLocalizationCallable::~ActorLocalizationCallable(){}

    PipelineMessage ActorLocalizationCallable::action(PipelineMessage message)
    {   
        int actor_id = message.getActorID();
        if(buffer_map.find(actor_id) != buffer_map.end()){
            auto wp = buffer_map[actor_id].peek();
            wp->getVector();
        }
        else{
            auto actor_location = carla::geom::Location(
                message.getAttribute("x"), 
                message.getAttribute("y"), 
                message.getAttribute("z"));
            auto closest_waypoint = shared_data->memory_map->getWaypoint(actor_location);
            
            while(!buffer_map[actor_id].full()){
                buffer_map[actor_id].push(closest_waypoint);
                closest_waypoint = closest_waypoint->getNextWaypoint()[0];
            }
        }
        PipelineMessage empty_message;
        
        return empty_message;
    }

}