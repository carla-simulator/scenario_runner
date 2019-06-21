#include<vector>
#include<cmath>
#include"system/boost/numeric/ublas/matrix.hpp"

class Rectangle
{
private:
    std::vector <float> heading_vector;
    std::vector <float> vehicle_coordinate;
    float length;
    float width;
    typedef boost::numeric::ublas::matrix <float> mat;
public:
    Rectangle();
    ~Rectangle();
    std::vector <float> find_coordinate (std::vector<float> heading_vector, std::vector<float> vehicle_coordinate, float length, float width, float angle);
    std::vector<std::vector <float>> find_rectangle_coordinates (std::vector<float> heading_vector, std::vector<float> vehicle_coordinate, float length, float width);
};
