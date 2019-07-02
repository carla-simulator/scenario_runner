// Defination of CollisionCallable calss members

#include "CollisionCallable.hpp"

namespace traffic_manager
{

    CollisionCallable::CollisionCallable(
        SyncQueue<PipelineMessage>* input_queue,
        SyncQueue<PipelineMessage>* output_queue,
        SharedData* shared_data):
        PipelineCallable(input_queue, output_queue, shared_data){}
    CollisionCallable::~CollisionCallable(){}

    PipelineMessage CollisionCallable::action(PipelineMessage message)
    {
        auto throttle = message.getAttribute("throttle");
        auto brake = message.getAttribute("brake");
        auto actor_list = shared_data->registered_actors;

        for(auto vehicle : actor_list){
            
            if(vehicle->GetId()!= message.getActorID())
            {   
                auto ego_actor = message.getActor();
                auto ego_actor_location = ego_actor.get()->GetLocation();
                float actor_distance = vehicle->GetLocation().Distance(ego_actor_location);
                if(actor_distance <= 20.0)
                {       
                    if(check_rect_intersection(vehicle , ego_actor) == true)
                    {
                        brake = 1.0;
                        throttle = 0.0;
                        break;
                    }
                }

            }
        }
        PipelineMessage out_message;
        out_message.setActor(message.getActor());
        out_message.setAttribute("throttle", throttle);
        out_message.setAttribute("brake", brake);
        out_message.setAttribute("steer", message.getAttribute("steer"));
        return out_message;
    }

    bool CollisionCallable::checkCollisionByDistance(carla::SharedPtr<carla::client::Actor> vehicle , carla::SharedPtr<carla::client::Actor> ego_vehicle)
    {
        auto ego_actor_location = ego_vehicle->GetLocation();
        float actor_distance = vehicle->GetLocation().Distance(ego_actor_location);
        if ( actor_distance < 10.0 )
        {   
            auto ego_forward_vector = ego_vehicle->GetTransform().GetForwardVector();
            auto magnitude_ego_forward_vector = sqrt(std::pow(ego_forward_vector.x, 2) + std::pow(ego_forward_vector.y , 2));
            auto actor_forward_vector = vehicle->GetTransform().location - ego_actor_location;
            auto magnitude_actor_forward_vector = sqrt(std::pow(actor_forward_vector.x, 2) + std::pow(actor_forward_vector.y , 2));
            auto dot_prod = ((ego_forward_vector.x * actor_forward_vector.x) + (ego_forward_vector.y * actor_forward_vector.y));
            dot_prod = ((dot_prod/magnitude_ego_forward_vector)/magnitude_actor_forward_vector);
            if(dot_prod > 0.9800)
            {   
                return true;
            }
            
        }
        return false;

    }

    bool CollisionCallable::check_rect_intersection(carla::SharedPtr<carla::client::Actor> vehicle ,
            carla::SharedPtr<carla::client::Actor> ego_vehicle){
            
        Rectangle boundry_box;
        auto ego_actor_location = ego_vehicle->GetLocation();
        auto ego_heading_vector = ego_vehicle->GetTransform().GetForwardVector();
        std::vector <std::vector <float> > ego_b_coor;
       
        auto ego_vehicle_bbox = boost::static_pointer_cast<carla::client::Vehicle> (ego_vehicle);
        auto ego_bounding_box = ego_vehicle_bbox->GetBoundingBox();
        auto ego_extent = ego_bounding_box.extent;
        float length = ego_extent.x;
        float width = ego_extent.y;
        
        ego_b_coor  = boundry_box.find_rectangle_coordinates(ego_heading_vector, ego_actor_location, length ,width);

        std::vector <std::vector <float> > actor_b_coor;
        auto actor_location = vehicle.get()->GetLocation();
        auto actor_heading_vector = vehicle.get()->GetTransform().GetForwardVector();
        
        auto actor_vehicle = boost::static_pointer_cast<carla::client::Vehicle> (vehicle);
        auto actor_bounding_box = actor_vehicle->GetBoundingBox();
        auto actor_extent = actor_bounding_box.extent;
        float actor_length = actor_extent.x;
        float actor_width = actor_extent.y;


        actor_b_coor =  boundry_box.find_rectangle_coordinates(actor_heading_vector, actor_location, actor_length, actor_width);
        std::string ego_coor;
        std::string vehicle_coor;
        for(int iter = 0; iter < ego_b_coor.size(); iter++){
            ego_coor = ego_coor + std::to_string(ego_b_coor[iter][0]) + " " + std::to_string(ego_b_coor[iter][1]) + ",";
            vehicle_coor = vehicle_coor + std::to_string(actor_b_coor[iter][0]) + " " + std::to_string(actor_b_coor[iter][1]) + ",";
        }
        ego_coor += std::to_string(ego_b_coor[0][0]) + " " + std::to_string(ego_b_coor[0][1]);
        vehicle_coor += std::to_string(actor_b_coor[0][0]) + " " + std::to_string(actor_b_coor[0][1]); 
        typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > polygon;
        polygon green, blue;
        boost::geometry::read_wkt("POLYGON(("+ego_coor+"))", green);
        boost::geometry::read_wkt("POLYGON(("+vehicle_coor+"))", blue);

        std::deque<polygon> output;
        boost::geometry::intersection(green, blue, output);

        auto ego_forward_vector = ego_vehicle->GetTransform().GetForwardVector();
        auto magnitude_ego_forward_vector = sqrt(std::pow(ego_forward_vector.x, 2) + std::pow(ego_forward_vector.y , 2));
        auto actor_forward_vector = vehicle->GetTransform().location - ego_actor_location;
        auto magnitude_actor_forward_vector = sqrt(std::pow(actor_forward_vector.x, 2) + std::pow(actor_forward_vector.y , 2));
        auto dot_prod = ((ego_forward_vector.x * actor_forward_vector.x) + (ego_forward_vector.y * actor_forward_vector.y));
        dot_prod = ((dot_prod/magnitude_ego_forward_vector)/magnitude_actor_forward_vector);
       
        BOOST_FOREACH(polygon const& p, output)
        {
            if(boost::geometry::area(p) > 0.0001 && dot_prod > 0.200){
                return true;
            }
        }
        return false;
    }
}