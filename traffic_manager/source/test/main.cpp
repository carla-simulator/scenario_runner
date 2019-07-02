#include <algorithm>
#include <iostream>
#include <typeinfo>
#include <queue>
#include <chrono>
#include "carla/client/Client.h"
#include "CarlaDataAccessLayer.hpp"
#include "carla/client/Waypoint.h"
#include "carla/client/ActorList.h"
#include "carla/client/Actor.h"
#include "InMemoryMap.hpp"
#include "PipelineMessage.hpp"
#include "PipelineStage.hpp"
#include "FeederCallable.hpp"
#include "ActorStateCallable.hpp"
#include "SyncQueue.hpp"
#include "ActorLocalizationCallable.hpp"
#include "ActorPIDCallable.hpp"
#include "BatchControlCallable.hpp"
#include "CollisionCallable.hpp"
#include "carla/rpc/Command.h"
#include "carla/rpc/VehicleControl.h"
#include "carla/client/Vehicle.h"
#include "carla/client/detail/Client.h"

void test_get_topology(carla::SharedPtr<carla::client::Map> world_map);
void test_feeder_stage(carla::SharedPtr<carla::client::ActorList> actor_list);
void test_actor_state_stage(carla::SharedPtr<carla::client::ActorList> actor_list);
void test_actor_state_stress(carla::SharedPtr<carla::client::ActorList> actor_list);
void test_actor_localization_stage(
    carla::SharedPtr<carla::client::ActorList> actor_list,
    carla::SharedPtr<carla::client::Map> world_map);
void test_in_memory_map(carla::SharedPtr<carla::client::Map>);
void test_actor_PID_stage(
    carla::SharedPtr<carla::client::ActorList> actor_list,
    carla::SharedPtr<carla::client::Map> world_map, carla::client::Client& client_conn);
void test_batch_control_stage(
    carla::SharedPtr<carla::client::ActorList> actor_list,
    carla::SharedPtr<carla::client::Map> world_map, carla::client::Client& client_conn);
void test_rectangle_class();
void test_collision_stage (
    carla::SharedPtr<carla::client::ActorList> actor_list,
    carla::SharedPtr<carla::client::Map> world_map,
    carla::client::Client& client_conn);

int main()
{   
    auto client_conn = carla::client::Client("localhost", 2000);
    std::cout<<"Connected with client object : "<<client_conn.GetClientVersion()<<std::endl;
    auto world = client_conn.GetWorld();
    auto world_map = world.GetMap();
    auto actorList = world.GetActors();
    auto vehicle_list = actorList->Filter("vehicle.*");

    // GenerateWaypoints();
    // test_get_topology(world_map);
    // test_feeder_stage(vehicle_list);
    // test_actor_state_stage(vehicle_list);
    // test_actor_state_stress(vehicle_list);
    // test_in_memory_map(world_map);
    // test_actor_localization_stage(vehicle_list, world_map);
    // test_actor_PID_stage(vehicle_list, world_map, client_conn);
    // test_batch_control_stage(vehicle_list, world_map, client_conn);
    test_collision_stage(vehicle_list, world_map, client_conn);
    // test_rectangle_class();

    return 0;
}

void test_collision_stage (
    carla::SharedPtr<carla::client::ActorList> actor_list,
    carla::SharedPtr<carla::client::Map> world_map,
    carla::client::Client& client_conn)
{

    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> feeder_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> actor_state_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> localization_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> pid_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> collision_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> batch_control_queue(20);

    traffic_manager::SharedData shared_data;
    for(auto it = actor_list->begin(); it != actor_list->end(); it++)
    {
        shared_data.registered_actors.push_back(*it);
    }

    auto dao = traffic_manager::CarlaDataAccessLayer(world_map);
    auto topology = dao.getTopology();
    auto local_map = std::make_shared<traffic_manager::InMemoryMap>(topology);
    local_map->setUp(1.0);
    shared_data.local_map = local_map;
    shared_data.client = &client_conn;
    auto debug_helper = client_conn.GetWorld().MakeDebugHelper();
    shared_data.debug = &debug_helper;

    traffic_manager::Feedercallable feeder_callable(NULL, &feeder_queue, &shared_data);
    traffic_manager::PipelineStage feeder_stage(1, feeder_callable);
    feeder_stage.start();
    
    traffic_manager::ActorStateCallable actor_state_callable(&feeder_queue, &actor_state_queue);
    traffic_manager::PipelineStage actor_state_stage(8, actor_state_callable);
    actor_state_stage.start();

    traffic_manager::ActorLocalizationCallable actor_localization_callable(&actor_state_queue, &localization_queue, &shared_data);
    traffic_manager::PipelineStage actor_localization_stage(8, actor_localization_callable);
    actor_localization_stage.start();

    float k_v = 1.0;
    float k_s = 3.0;
    float target_velocity = 10.0;
    traffic_manager::ActorPIDCallable actor_pid_callable(k_v, k_s, target_velocity, &localization_queue, &pid_queue);
    traffic_manager::PipelineStage actor_pid_stage(8, actor_pid_callable);
    actor_pid_stage.start();

    traffic_manager::CollisionCallable collision_callable(&pid_queue, &collision_queue, &shared_data);
    traffic_manager::PipelineStage collision_stage(1, collision_callable);
    collision_stage.start();

    traffic_manager::BatchControlCallable batch_control_callable(&collision_queue, &batch_control_queue, &shared_data);
    traffic_manager::PipelineStage batch_control_stage(1, batch_control_callable);
    batch_control_stage.start();

    std::cout << "All stage pipeline started !" <<std::endl;
    
    while(true)
    {
        sleep(1);
    }
}


void test_batch_control_stage (
    carla::SharedPtr<carla::client::ActorList> actor_list,
    carla::SharedPtr<carla::client::Map> world_map,
    carla::client::Client& client_conn)
{

    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> feeder_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> actor_state_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> localization_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> pid_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> batch_control_queue(20);

    traffic_manager::SharedData shared_data;
    for(auto it = actor_list->begin(); it != actor_list->end(); it++)
    {
        shared_data.registered_actors.push_back(*it);
    }

    auto dao = traffic_manager::CarlaDataAccessLayer(world_map);
    auto topology = dao.getTopology();
    auto local_map = std::make_shared<traffic_manager::InMemoryMap>(topology);
    local_map->setUp(1.0);
    shared_data.local_map = local_map;
    shared_data.client = &client_conn;
    auto debug_helper = client_conn.GetWorld().MakeDebugHelper();
    shared_data.debug = &debug_helper;

    traffic_manager::Feedercallable feeder_callable(NULL, &feeder_queue, &shared_data);
    traffic_manager::PipelineStage feeder_stage(1, feeder_callable);
    feeder_stage.start();
    
    traffic_manager::ActorStateCallable actor_state_callable(&feeder_queue, &actor_state_queue);
    traffic_manager::PipelineStage actor_state_stage(8, actor_state_callable);
    actor_state_stage.start();

    traffic_manager::ActorLocalizationCallable actor_localization_callable(&actor_state_queue, &localization_queue, &shared_data);
    traffic_manager::PipelineStage actor_localization_stage(8, actor_localization_callable);
    actor_localization_stage.start();

    float k_v = 1.0;
    float k_s = 3.0;
    float target_velocity = 10.0;
    traffic_manager::ActorPIDCallable actor_pid_callable(k_v, k_s, target_velocity, &localization_queue, &pid_queue);
    traffic_manager::PipelineStage actor_pid_stage(8, actor_pid_callable);
    actor_pid_stage.start();


    traffic_manager::BatchControlCallable batch_control_callable(&pid_queue, &batch_control_queue, &shared_data);
    traffic_manager::PipelineStage batch_control_stage(1, batch_control_callable);
    batch_control_stage.start();

    std::cout << "All stage pipeline started !" <<std::endl;
    
    while(true)
    {
        sleep(1);
    }
}


void test_in_memory_map(carla::SharedPtr<carla::client::Map> world_map) {
    auto dao = traffic_manager::CarlaDataAccessLayer(world_map);
    auto topology = dao.getTopology();
    traffic_manager::InMemoryMap local_map(topology);

    std::cout << "setup starting" << std::endl;
    local_map.setUp(1.0);
    std::cout << "setup complete" << std::endl;
    int loose_ends_count = 0;
    auto dense_topology = local_map.get_dense_topology();
    for (auto& swp : dense_topology) {
        if (swp->getNextWaypoint().size() < 1 || swp->getNextWaypoint()[0] == 0) {
            loose_ends_count += 1;
        }
    }
    std::cout << "Number of loose ends : " << loose_ends_count << std::endl;
}


void test_actor_PID_stage(
    carla::SharedPtr<carla::client::ActorList> actor_list,
    carla::SharedPtr<carla::client::Map> world_map,
    carla::client::Client& client_conn)
{

    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> feeder_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> actor_state_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> localization_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> pid_queue(20);

    traffic_manager::SharedData shared_data;
    for(auto it = actor_list->begin(); it != actor_list->end(); it++)
    {
        shared_data.registered_actors.push_back(*it);
    }

    auto dao = traffic_manager::CarlaDataAccessLayer(world_map);
    auto topology = dao.getTopology();
    auto local_map = std::make_shared<traffic_manager::InMemoryMap>(topology);
    local_map->setUp(1.0);
    shared_data.local_map = local_map;
    shared_data.client = &client_conn;
    auto debug_helper = client_conn.GetWorld().MakeDebugHelper();
    shared_data.debug = &debug_helper;

    traffic_manager::Feedercallable feeder_callable(NULL, &feeder_queue, &shared_data);
    traffic_manager::PipelineStage feeder_stage(1, feeder_callable);
    feeder_stage.start();
    
    traffic_manager::ActorStateCallable actor_state_callable(&feeder_queue, &actor_state_queue);
    traffic_manager::PipelineStage actor_state_stage(8, actor_state_callable);
    actor_state_stage.start();

    traffic_manager::ActorLocalizationCallable actor_localization_callable(&actor_state_queue, &localization_queue, &shared_data);
    traffic_manager::PipelineStage actor_localization_stage(1, actor_localization_callable);
    actor_localization_stage.start();

    float k_v = 1.0;
    float k_s = 3.0;
    float target_velocity = 10.0;
    traffic_manager::ActorPIDCallable actor_pid_callable(k_v, k_s, target_velocity, &localization_queue, &pid_queue);
    traffic_manager::PipelineStage actor_pid_stage(4, actor_pid_callable);
    actor_pid_stage.start();

    std::cout << "All stage pipeline started !" <<std::endl;
    
    carla::rpc::VehicleControl vehicle_control;            
    std::vector<carla::rpc::Command> commands;

    long count = 0;
    auto last_time = std::chrono::system_clock::now();
    while(true)
    {
        
        auto out = pid_queue.pop();
        auto actor = out.getActor();
        vehicle_control.throttle = out.getAttribute("throttle");
        vehicle_control.brake = out.getAttribute("brake");
        vehicle_control.steer = out.getAttribute("steer");

        auto vehicle = (carla::client::Vehicle*) &(*actor); 
        vehicle->ApplyControl(vehicle_control);

        count++;
        auto current_time = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = current_time - last_time;
    
        if(diff.count() > 1.0)    
        {    
            last_time = current_time;    
            std::cout << "Updates proces    sed per second " << count << std::endl;
            count = 0;    
        }    
    }    
}    
    
void test_actor_localization_stage(carla::SharedPtr<carla::client::ActorList> actor_list, carla::SharedPtr<carla::client::Map> world_map)
{

    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> feeder_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> actor_state_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> localization_queue(20);
    

    traffic_manager::SharedData shared_data;
    for(auto it = actor_list->begin(); it != actor_list->end(); it++)
    {
        shared_data.registered_actors.push_back(*it);
    }
   
    auto dao = traffic_manager::CarlaDataAccessLayer(world_map);
    auto topology = dao.getTopology();
    auto local_map = std::make_shared<traffic_manager::InMemoryMap>(topology);
    local_map->setUp(1.0);
    shared_data.local_map = local_map;

    traffic_manager::Feedercallable feeder_callable(NULL, &feeder_queue, &shared_data);
    traffic_manager::PipelineStage feeder_stage(1, feeder_callable);
    feeder_stage.start();
    
    traffic_manager::ActorStateCallable actor_state_callable(&feeder_queue, &actor_state_queue);
    traffic_manager::PipelineStage actor_state_stage(8, actor_state_callable);
    actor_state_stage.start();

    traffic_manager::ActorLocalizationCallable actor_localization_callable(&actor_state_queue, &localization_queue, &shared_data);
    traffic_manager::PipelineStage actor_localization_stage(1, actor_localization_callable);
    actor_localization_stage.start();

    std::cout << "All stage pipeline started !" <<std::endl;
    
    while(true)
    {
        auto out = localization_queue.pop();
        std::cout << "Velocity : " << out.getAttribute("velocity")
            << "\t Deviation : " << out.getAttribute("deviation") 
            << "\t Queue size : " << localization_queue.size() << std::endl;
    }
}

void test_actor_state_stress(carla::SharedPtr<carla::client::ActorList> actor_list)
{

    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> feeder_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> actor_state_queue(20);

    traffic_manager::SharedData shared_data;
    for(auto it = actor_list->begin(); it != actor_list->end(); it++)
    {
        shared_data.registered_actors.push_back(*it);
    }

    traffic_manager::Feedercallable feeder_callable(NULL, &feeder_queue, &shared_data);
    traffic_manager::PipelineStage feeder_stage(1, feeder_callable);
    feeder_stage.start();
    
    traffic_manager::ActorStateCallable actor_state_callable(&feeder_queue, &actor_state_queue);
    traffic_manager::PipelineStage actor_state_stage(8, actor_state_callable);
    actor_state_stage.start();
    
    long count = 0;
    auto last_time = std::chrono::system_clock::now();
    while(true)
    {
        auto out = actor_state_queue.pop();

        count++;
        auto current_time = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = current_time - last_time;

        if(diff.count() > 1.0)
        {
            last_time = current_time;
            std::cout << "Vehicles processed per second " << count << std::endl;
            count = 0;
        }
    }
}

void test_actor_state_stage(carla::SharedPtr<carla::client::ActorList> actor_list)
{

    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> feeder_queue(20);
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> actor_state_queue(20);

    traffic_manager::SharedData shared_data;
    for(auto it = actor_list->begin(); it != actor_list->end(); it++)
    {
        shared_data.registered_actors.push_back(*it);
    }

    traffic_manager::Feedercallable feeder_callable(NULL, &feeder_queue, &shared_data);
    traffic_manager::PipelineStage feeder_stage(1, feeder_callable);
    feeder_stage.start();
    sleep(1);

    std::cout << "Size of feeder queue : " << feeder_queue.size() << std::endl;
    traffic_manager::ActorStateCallable actor_state_callable(&feeder_queue, &actor_state_queue);
    traffic_manager::PipelineStage actor_state_stage(4, actor_state_callable);
    actor_state_stage.start();
    sleep(1);

    int count = 10;
    while(!actor_state_queue.empty() && count > 0)
    {
        auto out = actor_state_queue.pop();

        std::cout << "Actor type id " << out.getActor()->GetTypeId() << std::endl;
        std::cout << "Actor id " << out.getActorID() << std::endl;
        std::cout << "Actor velocity " << out.getAttribute("velocity") << std::endl;
        std::cout << "Actor x " << out.getAttribute("x") << std::endl;
        count--;
    }
    while(true)
        sleep(1);
}
void test_feeder_stage(carla::SharedPtr<carla::client::ActorList> actor_list)
{
    traffic_manager::SyncQueue<traffic_manager::PipelineMessage> out_queue(20);

    traffic_manager::SharedData shared_data;
    for(auto it = actor_list->begin(); it != actor_list->end(); it++)
    {
        shared_data.registered_actors.push_back(*it);
    }

    traffic_manager::Feedercallable feeder_callable(NULL, &out_queue, &shared_data);
    traffic_manager::PipelineStage feeder_stage(1, feeder_callable);
    feeder_stage.start();
    sleep(1);

    int count = 10;
    while(count > 0)
    {
        auto out = out_queue.pop();

        std::cout << "Actor_Type_id " << out.getActor()->GetTypeId() << std::endl;
        std::cout << "Actor_id " << out.getActor()->GetId() << std::endl;
        count--;
    }
    while(true)
        sleep(1);
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
