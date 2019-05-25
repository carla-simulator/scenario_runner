#include <algorithm>
#include <iostream>
#include <typeinfo>
#include "carla/client/Client.h"
#include "CarlaDataAccessLayer.hpp"
#include "InMemoryMap.hpp"
#include "carla/client/Waypoint.h"
//#include "ActorReadStage.hpp"
#include "carla/client/ActorList.h"
#include "carla/client/Actor.h"
#include "RegisteredActorMessage.hpp"
#include "FeederStage.hpp"
#include "PipelineMessage.hpp"
#include <queue>
#include "ActorStateMessage.hpp"
#include "ActorStateStage.hpp"

void test(traffic_manager::RegisteredActorMessage registerActorObj,
    int pull, int buffer_size,
    std::queue <traffic_manager::PipelineMessage> input_queue,
    std::queue <traffic_manager::PipelineMessage> out_queue);

void test2(traffic_manager::ActorStateMessage actor_state_obj,
    int pull, int buffer_size,
    std::queue <traffic_manager::PipelineMessage> input_queue,
    std::queue <traffic_manager::PipelineMessage> out_queue);

int main()
{   
    auto client_conn = carla::client::Client("localhost", 2000);
    std::cout<<"Connected with client object : "<<client_conn.GetClientVersion()<<std::endl;
    auto world = client_conn.GetWorld();
    auto world_map = world.GetMap();
    auto actorList = world.GetActors();
    auto vehicle_list = actorList->Filter("vehicle.*");
    traffic_manager::ActorStateMessage actor_state_obj;
    traffic_manager::RegisteredActorMessage registerActorObj;
    //auto registerActorObj.shared_actor_list;
    for(auto it = vehicle_list->begin(); it != vehicle_list->end(); it++)
    {
        registerActorObj.shared_actor_list.push_back(*it);
    }


    std::queue <traffic_manager::PipelineMessage> input_queue;
    std::queue <traffic_manager::PipelineMessage> out_queue;

    //test(registerActorObj, 1, 20, input_queue, out_queue);
    test2(actor_state_obj, 1, 20,input_queue,out_queue);
    //traffic_manager::FeederStage feeder_stage(&registerActorObj, 20, &input_queue, &out_queue);
    
}

void test(traffic_manager::RegisteredActorMessage registerActorObj,
    int pull, int buffer_size,
    std::queue <traffic_manager::PipelineMessage> input_queue,
    std::queue <traffic_manager::PipelineMessage> out_queue)
{
    traffic_manager::FeederStage feeder_stage(&registerActorObj, buffer_size, &input_queue, &out_queue);
    feeder_stage.start();
    sleep(1);
    //std::cout <<"out_queue size : " <<out_queue.size()<< std::endl;
    int count = 20;
    while(!out_queue.empty() && count > 0)
    {
        auto out = out_queue.front();
        out_queue.pop();

        std::cout << "Actor_Type_id " << out.getActor()->GetTypeId() << std::endl;
        std::cout << "Actor_id " << out.getActor()->GetId() << std::endl;
        count--;
    }
    while(true);
}

void test2(traffic_manager::ActorStateMessage actor_state_obj,
    int pull, int buffer_size,
    std::queue <traffic_manager::PipelineMessage> input_queue,
    std::queue <traffic_manager::PipelineMessage> out_queue)
{
    traffic_manager::ActorStateStage actorstage_obj(&actor_state_obj, buffer_size, &input_queue, &out_queue);
    actorstage_obj.start();
    sleep(1);

     int count = 20;
    while(!out_queue.empty() && count > 0)
    {
        auto out = out_queue.front();
        out_queue.pop();

        std::cout << "Actor_Type_id " << out_queue.size()<< std::endl;
        //std::cout << "Actor_id " << out.getActor()->GetId() << std::endl;
        count--;
    }
    while(true);

}






// //void test_closest_waypoint(carla::SharedPtr<carla::client::ActorList> vehicle_list, carla::SharedPtr<carla::client::Map> world_map);
// std::vector<int> getActors(carla::SharedPtr<carla::client::ActorList> _actor_list);
// int main(){
//     auto client_conn = carla::client::Client("localhost", 2000);
//     std::cout<<"Connected with client object : "<<client_conn.GetClientVersion()<<std::endl;
//     auto world = client_conn.GetWorld();
//     auto world_map = world.GetMap();
//     auto actorList = world.GetActors();
//     auto vehicle_list = actorList->Filter("vehicle.*");
//     auto topology = world_map->GetTopology();
//     auto newtopo = traffic_manager::InMemoryMap(topology);
//     auto all_actors = getActors(vehicle_list);
//     traffic_manager::RegisteredActorMessage actorobj;
//     int actorId = actorobj.getActorID();
//     std::cout << actorId << std::endl;
//     // auto location = getLocation(vehicle_list);
//     // auto allwayp = newtopo.getWaypoint(location);
//     return 0;
// }
