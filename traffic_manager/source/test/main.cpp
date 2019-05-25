#include <algorithm>
#include <iostream>
#include <typeinfo>
#include "carla/client/Client.h"
#include "CarlaDataAccessLayer.hpp"
#include "InMemoryMap.hpp"
#include "carla/client/Waypoint.h"
#include "carla/client/ActorList.h"
#include "carla/client/Actor.h"
#include "RegisteredActorMessage.hpp"
#include "FeederStage.hpp"
#include "PipelineMessage.hpp"
#include <queue>
#include "ActorStateMessage.hpp"
#include "ActorStateStage.hpp"

void test_get_topology(carla::SharedPtr<carla::client::Map> world_map);
void test_feeder_stage(carla::SharedPtr<carla::client::ActorList> actor_list);

void test_actor_state_stage(carla::SharedPtr<carla::client::ActorList> actor_list);

int main()
{   
    auto client_conn = carla::client::Client("localhost", 2000);
    std::cout<<"Connected with client object : "<<client_conn.GetClientVersion()<<std::endl;
    auto world = client_conn.GetWorld();
    auto world_map = world.GetMap();
    auto actorList = world.GetActors();
    auto vehicle_list = actorList->Filter("vehicle.*");

    test_get_topology(world_map);
    // test_feeder_stage(vehicle_list);
    test_actor_state_stage(vehicle_list);

    return 0;
}

void test_get_topology(carla::SharedPtr<carla::client::Map> world_map) {

    auto dao = traffic_manager::CarlaDataAccessLayer(world_map);
    auto topology = dao.getTopology();

    typedef std::vector<
        std::pair<
            carla::SharedPtr<carla::client::Waypoint>,
            carla::SharedPtr<carla::client::Waypoint>
        >
    > toplist;
    for(toplist::iterator it = topology.begin(); it != topology.end(); it++) {
        auto wp1 = it->first;
        auto wp2 = it->second;
        std::cout << "Segment end road IDs : " << wp1->GetRoadId() << " -- " << wp2->GetRoadId() << std::endl;
    }
}

void test_feeder_stage(carla::SharedPtr<carla::client::ActorList> actor_list)
{
    std::queue <traffic_manager::PipelineMessage> input_queue;
    std::queue <traffic_manager::PipelineMessage> out_queue;

    traffic_manager::RegisteredActorMessage registerActorObj;
    for(auto it = actor_list->begin(); it != actor_list->end(); it++)
    {
        registerActorObj.shared_actor_list.push_back(*it);
    }

    traffic_manager::FeederStage feeder_stage(&registerActorObj, 20, &input_queue, &out_queue);
    feeder_stage.start();
    sleep(1);
    int count = 20;
    while(!out_queue.empty() && count > 0)
    {
        auto out = out_queue.front();
        out_queue.pop();

        std::cout << "Actor_Type_id " << out.getActor()->GetTypeId() << std::endl;
        std::cout << "Actor_id " << out.getActor()->GetId() << std::endl;
        count--;
    }
    while(true) {
        sleep(5);
    };
}

void test_actor_state_stage(carla::SharedPtr<carla::client::ActorList> actor_list)
{
    std::queue <traffic_manager::PipelineMessage> input_queue;
    std::queue <traffic_manager::PipelineMessage> out_queue;
    std::queue <traffic_manager::PipelineMessage> in_queue_to_actorstage;
    std::queue <traffic_manager::PipelineMessage> out_queue_to_actorstage;

    traffic_manager::RegisteredActorMessage registerActorObj;
    for(auto it = actor_list->begin(); it != actor_list->end(); it++)
    {
        registerActorObj.shared_actor_list.push_back(*it);
    }

    traffic_manager::FeederStage feeder_stage(&registerActorObj, 20, &input_queue, &out_queue);
    feeder_stage.start();
    sleep(1);

    //call_feeder_stage(&registerActorObj, &input_queue, &out_queue);
    // int count = 20;
    // while(!in_queue_to_actorstage.empty() && count > 0)
    // {
    //     auto out = in_queue_to_actorstage.front();
    //     in_queue_to_actorstage.pop();

    //     std::cout << "Actor_id " << out.getActor()->GetId() << std::endl;
    //     count--;
    // }

    traffic_manager::ActorStateStage actorstage_obj(20, &out_queue, &out_queue_to_actorstage);
    actorstage_obj.start();
    sleep(1);

    // int count1 = 20;
    // while(!out_queue_to_actorstage.empty() && count1 > 0)
    // {
    //     auto out = out_queue_to_actorstage.front();
    //     out_queue_to_actorstage.pop();

    //     std::cout << "actor state actor_id " << out.getActor()->GetId() << std::endl;
    //     count1--;
    // }
    while(true);
}