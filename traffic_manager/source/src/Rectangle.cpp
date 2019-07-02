#include "Rectangle.hpp"


Rectangle::Rectangle(){}
Rectangle::~Rectangle(){}

std::vector<std::vector<float>> Rectangle::find_rectangle_coordinates(carla::geom::Vector3D heading_vector, carla::geom::Location vehicle_coordinate, float length, float width)
    {   
        int k = 3;
        float rear_angle = atan(width/length);
        float forward_angle = atan(width/(k*length));
        std::vector <std::vector<float>> rectangle_coordinates;
        rectangle_coordinates.push_back(find_coordinate(heading_vector, vehicle_coordinate, k*length, width, forward_angle));
        rectangle_coordinates.push_back(find_coordinate(heading_vector, vehicle_coordinate, k*length, width, 2*(3.141592) - forward_angle));
        rectangle_coordinates.push_back(find_coordinate(heading_vector, vehicle_coordinate, length, width, 3.141592 + rear_angle));
        rectangle_coordinates.push_back(find_coordinate(heading_vector, vehicle_coordinate, length, width, 3.141592 - rear_angle));

        return rectangle_coordinates;
    }


std::vector <float> Rectangle::find_coordinate (carla::geom::Vector3D heading_vector, carla::geom::Location vehicle_coordinate, float length, float width, float angle)
    {
        mat rotation_matrix(2,2);
        rotation_matrix(0,0) = cos(angle);
        rotation_matrix(0,1) = -(sin(angle));
        rotation_matrix(1,0) = sin(angle);
        rotation_matrix(1,1) = cos(angle);

        float magnitude = sqrt((heading_vector.x)*(heading_vector.x) + (heading_vector.y)*(heading_vector.y));
        mat heading_matrix(2,1);
        heading_matrix(0,0) = heading_vector.x/magnitude;
        heading_matrix(1,0) = heading_vector.y/magnitude;
        float distance = sqrt((length*length) + (width*width));
        mat rotated_heading_matrix = (boost::numeric::ublas::prod(rotation_matrix, heading_matrix))*distance;
        std::vector<float> coordinates = {(rotated_heading_matrix(0,0)+ vehicle_coordinate.x), 
                                        (rotated_heading_matrix(1,0)+ vehicle_coordinate.y), vehicle_coordinate.z};

        // std::cout<< "forward vector " << heading_vector.x <<"\t"<< heading_vector.y<< std::endl;                              
        // std::cout<< "vehicle coordinates: " << vehicle_coordinate.x <<"\t"<< vehicle_coordinate.y<< std::endl;                                
        // std::cout<< "rectangle class coordinates: " << coordinates[0] <<"\t"<< coordinates[1]<< std::endl;
        return coordinates;        
    }

