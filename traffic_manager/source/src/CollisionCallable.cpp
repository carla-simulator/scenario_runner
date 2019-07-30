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
        auto actor_list = shared_data->registered_actors;

        float collision_hazard = -1;
        for(auto actor : actor_list){
            
            if (
                actor->GetId() != message.getActorID() 
                and shared_data->buffer_map.find(actor->GetId()) != shared_data->buffer_map.end()
            ) {   
                auto ego_actor = message.getActor();
                auto ego_actor_location = ego_actor->GetLocation();
                float actor_distance = actor->GetLocation().Distance(ego_actor_location);
                if(actor_distance <= 20.0) // Account for this constant
                {       
                    if(negotiateCollision(ego_actor, actor))
                    {
                        collision_hazard = 1;
                        break;
                    }
                }
            }
        }

        PipelineMessage out_message;
        out_message.setActor(message.getActor());
        out_message.setAttribute("collision", collision_hazard);
        out_message.setAttribute("velocity", message.getAttribute("velocity"));
        out_message.setAttribute("deviation", message.getAttribute("deviation"));

        return out_message;
    }

    void CollisionCallable::drawBoundary(std::vector<carla::geom::Location> boundary) {
        for (int i=0; i<boundary.size(); i++) {
            shared_data->debug->DrawLine(
                boundary[i] + carla::geom::Location(0, 0, 1),
                boundary[(i+1)%boundary.size()] + carla::geom::Location(0, 0, 1),
                0.1f, {255U, 0U, 0U} , 0.1f);
        }
    }

    bool CollisionCallable::negotiateCollision(
        carla::SharedPtr<carla::client::Actor> ego_vehicle,
        carla::SharedPtr<carla::client::Actor> other_vehicle
    ) {
        auto overlap = checkGeodesicCollision(ego_vehicle, other_vehicle);

        auto reference_heading_vector = ego_vehicle->GetTransform().GetForwardVector();
        reference_heading_vector.z = 0;
        reference_heading_vector = reference_heading_vector.MakeUnitVector();
        auto relative_other_vector = other_vehicle->GetLocation() - ego_vehicle->GetLocation();
        relative_other_vector.z = 0;
        relative_other_vector = relative_other_vector.MakeUnitVector();
        float reference_relative_dot = reference_heading_vector.x*relative_other_vector.x +
        reference_heading_vector.y*relative_other_vector.y;

        auto relative_reference_vector = ego_vehicle->GetLocation() - other_vehicle->GetLocation();
        relative_reference_vector.z = 0;
        relative_reference_vector = relative_reference_vector.MakeUnitVector();
        auto other_heading_vector = other_vehicle->GetTransform().GetForwardVector();
        other_heading_vector.z = 0;
        other_heading_vector = other_heading_vector.MakeUnitVector();
        float other_relative_dot = other_heading_vector.x*relative_reference_vector.x + 
            other_heading_vector.y*relative_reference_vector.y;

        if (
            overlap > 0
            and
            reference_relative_dot > other_relative_dot
        ) {
            return true;
        }

        return false;
    }

    bool CollisionCallable::checkGeodesicCollision(
        carla::SharedPtr<carla::client::Actor> reference_vehicle,
        carla::SharedPtr<carla::client::Actor> other_vehicle
    ) {
        auto reference_height = reference_vehicle->GetLocation().z;
        auto other_height = other_vehicle->GetLocation().z;
        if (abs(reference_height-other_height) < 1.0) { // Constant again
            auto reference_bbox = getBoundary(reference_vehicle);
            auto other_bbox = getBoundary(other_vehicle);
            auto reference_geodesic_boundary = getGeodesicBoundary(
                reference_vehicle, reference_bbox);
            auto other_geodesic_boundary = getGeodesicBoundary(
                other_vehicle, other_bbox);
            auto reference_polygon = getPolygon(reference_geodesic_boundary);
            auto other_polygon = getPolygon(other_geodesic_boundary);

            std::deque<polygon> output;
            boost::geometry::intersection(reference_polygon, other_polygon, output);

            BOOST_FOREACH(polygon const& p, output)
            {
                if(
                    boost::geometry::area(p) > 0.0001
                ){ // Make thresholds constants
                    // drawBoundary(reference_geodesic_boundary);
                    return true;
                }
            }
        }

        return false;
    }

    traffic_manager::polygon CollisionCallable::getPolygon(std::vector<carla::geom::Location> boundary) {
        std::string wkt_string;
        for(auto location: boundary){
            wkt_string += std::to_string(location.x) + " " + std::to_string(location.y) + ",";
        }
        wkt_string += std::to_string(boundary[0].x) + " " + std::to_string(boundary[0].y);

        traffic_manager::polygon boundary_polygon;
        boost::geometry::read_wkt("POLYGON(("+wkt_string+"))", boundary_polygon);

        return boundary_polygon;
    }

    std::vector<carla::geom::Location> CollisionCallable::getGeodesicBoundary (
        carla::SharedPtr<carla::client::Actor> actor,
        std::vector<carla::geom::Location> bbox
    ) {
        auto velocity = actor->GetVelocity().Length();
        int bbox_extension = static_cast<int>(
            std::max(std::sqrt(7*velocity), 2.0f)
            + std::max(velocity*0.5, 2.0) + 1.0
        ); // Account for these constants
        bbox_extension = velocity > 50/3.6 ? 5*velocity : bbox_extension;
        auto simple_waypoints = this->shared_data->buffer_map[actor->GetId()]->getContent(bbox_extension);
        std::vector<carla::geom::Location> left_boundary;
        std::vector<carla::geom::Location> right_boundary;
        auto vehicle = boost::static_pointer_cast<carla::client::Vehicle>(actor);
        float width = vehicle->GetBoundingBox().extent.y;
        
        for (auto swp: simple_waypoints) {
            auto vector = swp->getVector();
            auto location = swp->getLocation();
            auto perpendicular_vector = carla::geom::Vector3D(-1* vector.y, vector.x, 0);
            perpendicular_vector = perpendicular_vector.MakeUnitVector();
            left_boundary.push_back(location + carla::geom::Location(perpendicular_vector*width));
            right_boundary.push_back(location - carla::geom::Location(perpendicular_vector*width));
        }

        std::vector<carla::geom::Location> geodesic_boundary;
        std::reverse(left_boundary.begin(), left_boundary.end());
        geodesic_boundary.insert(geodesic_boundary.end(), left_boundary.begin(), left_boundary.end());
        geodesic_boundary.insert(geodesic_boundary.end(), bbox.begin(), bbox.end());
        geodesic_boundary.insert(geodesic_boundary.end(), right_boundary.begin(), right_boundary.end());
        std::reverse(geodesic_boundary.begin(), geodesic_boundary.end());

        return geodesic_boundary;
    }

    std::vector<carla::geom::Location> CollisionCallable::getBoundary (carla::SharedPtr<carla::client::Actor> actor) {
        auto vehicle = boost::static_pointer_cast<carla::client::Vehicle>(actor);
        auto bbox = vehicle->GetBoundingBox();
        auto extent = bbox.extent;
        auto location = vehicle->GetLocation();
        auto heading_vector = vehicle->GetTransform().GetForwardVector();
        heading_vector.z = 0;
        heading_vector = heading_vector.MakeUnitVector();
        auto perpendicular_vector = carla::geom::Vector3D(-1* heading_vector.y, heading_vector.x, 0);

        return {
            location + carla::geom::Location(heading_vector*extent.x + perpendicular_vector*extent.y),
            location + carla::geom::Location(-1*heading_vector*extent.x + perpendicular_vector*extent.y),
            location + carla::geom::Location(-1*heading_vector*extent.x - perpendicular_vector*extent.y),
            location + carla::geom::Location(heading_vector*extent.x - perpendicular_vector*extent.y)
            };
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

}
