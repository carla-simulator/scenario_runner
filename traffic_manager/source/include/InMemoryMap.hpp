// Class for in memory representation of descrete map

#include <map>
#include <cmath>
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

class InMemoryMap
{
private:
    TopologyList topology;
    std::vector<SimpleWaypoint> dense_topology;
    typedef std::map<std::pair<int, int>, SimpleWaypoint*> NodeMap;
    NodeMap entry_node_map;
    NodeMap exit_node_map;
    std::pair<int, int> make_node_key(carla::SharedPtr<carla::client::Waypoint> waypooint);
public:
    InMemoryMap(TopologyList topology);
    ~InMemoryMap();
    void setUp(int sampling_resolution);
    traffic_manager::SimpleWaypoint* getWaypoint(carla::geom::Location location);
    //void listofAllWaypoint(std::vector<SimpleWaypoint> dense_topology);
};

}
