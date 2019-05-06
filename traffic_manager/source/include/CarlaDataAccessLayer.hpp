// Data access layer to carla server

#include <vector>
#include <array>

#include "carla/client/Map.h"
#include "carla/client/Waypoint.h"
#include "carla/Memory.h"

namespace traffic_manager {

class CarlaDataAccessLayer
{
private:
    carla::SharedPtr<carla::client::Map> world_map;
public:
    CarlaDataAccessLayer(carla::SharedPtr<carla::client::Map> world_map);
    ~CarlaDataAccessLayer();
    std::vector<
        std::pair<
            carla::SharedPtr<carla::client::Waypoint>,
            carla::SharedPtr<carla::client::Waypoint
            >
        >
    > getTopology();

};

} // namespace traffic_manager