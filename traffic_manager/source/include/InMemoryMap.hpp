// Class for in memory representation of descrete map
#pragma once

#include <map>
#include <cmath>
#include <memory>
#include <string>
#include "carla/Memory.h"
#include "carla/client/Waypoint.h"
#include "carla/geom/Location.h"
#include "SimpleWaypoint.hpp"

namespace traffic_manager {

typedef std::vector<
    std::pair<
        carla::SharedPtr<carla::client::Waypoint>,
        carla::SharedPtr<carla::client::Waypoint>
    >
> TopologyList;
typedef std::vector<std::shared_ptr<SimpleWaypoint>> NodeList;

class InMemoryMap
{
private:
    TopologyList topology;
    std::vector<std::shared_ptr<SimpleWaypoint>> dense_topology;
    NodeList entry_node_list;
    NodeList exit_node_list;
public:
    InMemoryMap(TopologyList topology);
    ~InMemoryMap();
    void setUp(int sampling_resolution);
    std::shared_ptr<SimpleWaypoint> getWaypoint(carla::geom::Location location);
    std::vector<std::shared_ptr<SimpleWaypoint>> get_dense_topology();
};

}
