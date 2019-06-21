// Defination of CollisionCallable calss members

#include "CollisionCallable.hpp"
#include <cmath>

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
        std::vector<float> ego_actor_location = {message.getActor()->GetLocation().x,
            message.getActor()->GetLocation().y,
            message.getActor()->GetLocation().z};
        
        auto ego_first_coor = carla::geom::Location(ego_actor_location[0]+1.2, ego_actor_location[1]+2.5, ego_actor_location[2]);
        auto ego_second_coor = carla::geom::Location(ego_actor_location[0]-1.2, ego_actor_location[1]+2.5, ego_actor_location[2]);
        auto ego_third_coor = carla::geom::Location(ego_actor_location[0]-1.2, ego_actor_location[1]-2.5, ego_actor_location[2]);
        auto ego_fourth_coor = carla::geom::Location(ego_actor_location[0]+1.2, ego_actor_location[1]-2.5, ego_actor_location[2]);

        for(auto vehicle : actor_list){
            // auto vehicle = (carla::client::Vehicle*) &(*actor);
            // auto vehicle_bounding = vehicle-> 

            if(vehicle->GetId()!= message.getActorID() )
            {   
                std::vector<float> vehicle_location = {vehicle->GetLocation().x,
                vehicle->GetLocation().y, vehicle->GetLocation().z};
                auto rect_first_coor = carla::geom::Location(vehicle_location[0]+1.2, vehicle_location[1]+2.5, vehicle_location[2]);
                auto rect_second_coor = carla::geom::Location(vehicle_location[0]-1.2, vehicle_location[1]+2.5, vehicle_location[2]);
                auto rect_third_coor = carla::geom::Location(vehicle_location[0]-1.2, vehicle_location[1]-2.5, vehicle_location[2]);
                auto rect_fourth_coor = carla::geom::Location(vehicle_location[0]+1.2, vehicle_location[1]-2.5, vehicle_location[2]);

                if(check_rect_inter(ego_first_coor,rect_first_coor,rect_second_coor,rect_third_coor,rect_fourth_coor)||
                    check_rect_inter(ego_second_coor,rect_first_coor,rect_second_coor,rect_third_coor,rect_fourth_coor)||
                    check_rect_inter(ego_third_coor,rect_first_coor,rect_second_coor,rect_third_coor,rect_fourth_coor)||
                    check_rect_inter(ego_fourth_coor,rect_first_coor,rect_second_coor,rect_third_coor,rect_fourth_coor))
                {
                    throttle = 0.5 * throttle;
                    std::cout << "changing throttle"<< std::endl;
                }

            }


            

            // shared_data->debug->DrawLine(rect_first_coo, rect_sec_coo, 0.2, carla::client::DebugHelper::Color{255U, 0U, 0U}, 10.0);
            // shared_data->debug->DrawLine(rect_sec_coo, rect_thi_coo, 0.2, carla::client::DebugHelper::Color{0U, 255U, 0U}, 10.0);
            // shared_data->debug->DrawLine(rect_thi_coo, rect_fou_coo, 0.2, carla::client::DebugHelper::Color{0U, 0U, 255U}, 10.0);
            // shared_data->debug->DrawLine(rect_first_coo, rect_fou_coo, 0.2, carla::client::DebugHelper::Color{255U, 0U, 255U}, 10.0);
        }
        PipelineMessage out_message;
        out_message.setActor(message.getActor());
        out_message.setAttribute("throttle", throttle);
        out_message.setAttribute("brake", brake);
        out_message.setAttribute("steer", message.getAttribute("steer"));
        return out_message;
    }
    bool CollisionCallable::check_rect_inter(carla::geom::Location point, carla::geom::Location coor1, carla::geom::Location coor2,
        carla::geom::Location coor3, carla::geom::Location coor4)
    {
        float total_area = calculate_area(coor1, coor2, coor3) + calculate_area( coor3, coor4, coor1);
        float A1 = calculate_area(coor1, coor2, point);
        float A2 = calculate_area(coor2, coor3, point);
        float A3 = calculate_area(coor3, coor4, point);
        float A4 = calculate_area(coor4, coor1, point);
        if( A1 + A2 + A3 + A4 <= total_area)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    float CollisionCallable::calculate_area (carla::geom::Location coor1, carla::geom::Location coor2,
        carla::geom::Location coor3)
    {   
        float area = ((coor1.x*(coor2.y -coor3.y)) + (coor2.x*(coor3.y -coor1.y)) + (coor3.x*(coor1.y-coor2.y)))/2.0;
        return area;
    }
}