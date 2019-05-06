#include <algorithm>

#include "carla/client/Client.h"
#include "CarlaDataAccessLayer.hpp"
#include "carla/client/Waypoint.h"

void test_get_topology(carla::SharedPtr<carla::client::Map> world_map);

int main(){
    auto client_conn = carla::client::Client("localhost", 2000);
    std::cout<<"Connected with client object : "<<client_conn.GetClientVersion()<<std::endl;
    auto world = client_conn.GetWorld();
    auto world_map = world.GetMap();

    test_get_topology(world_map);

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