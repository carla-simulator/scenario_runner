#include "Rectangle.hpp"


Rectangle::Rectangle(){}
Rectangle::~Rectangle(){}

std::vector<std::vector<float>> Rectangle::find_rectangle_coordinates(std::vector<float> heading_vector, std::vector<float> vehicle_coordinate, float length, float width)
    {   
        float angle = atan(width/length);
        std::vector <std::vector<float>> rectangle_coordinates;
        rectangle_coordinates.push_back(find_coordinate(heading_vector, vehicle_coordinate, length, width, angle));
        float angle_new = 3.141592 - angle;
        rectangle_coordinates.push_back(find_coordinate(heading_vector, vehicle_coordinate, length, width, angle_new));
        angle_new = 3.141592 + angle;
        rectangle_coordinates.push_back(find_coordinate(heading_vector, vehicle_coordinate, length, width, angle_new));
        angle_new = 2*(3.141592) - angle;
        rectangle_coordinates.push_back(find_coordinate(heading_vector, vehicle_coordinate, length, width, angle_new));

        return rectangle_coordinates;
    }


std::vector <float> Rectangle::find_coordinate (std::vector<float> heading_vector, std::vector<float> vehicle_coordinate, float length, float width, float angle)
    {
        mat rotation_matrix(2,2);
        rotation_matrix(0,0) = cos(angle);
        rotation_matrix(0,1) = -(sin(angle));
        rotation_matrix(1,0) = sin(angle);
        rotation_matrix(1,1) = cos(angle);

        mat heading_matrix(2,1);
        heading_matrix(0,0) = heading_vector[0];
        heading_matrix(1,0) = heading_vector[1];
        float distance = sqrt(((length)*(length)) + ((width)*(width)));

        mat rotated_heading_matrix = boost::numeric::ublas::prod(rotation_matrix, heading_matrix)*distance;
        std::vector<float> coordinates = {(rotated_heading_matrix(0,0)+ vehicle_coordinate[0]), 
                                        (rotated_heading_matrix(1,0)+ vehicle_coordinate[1])};
        return coordinates;        
    }

