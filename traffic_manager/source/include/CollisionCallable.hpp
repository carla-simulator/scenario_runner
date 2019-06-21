//Declaration of CollisionCallable class member

#pragma once
#include "PipelineCallable.hpp"
namespace traffic_manager
{

    class CollisionCallable : public PipelineCallable
    {
        private:
            //SyncQueue<PipelineMessage>* input_queue;
            
        public:
            CollisionCallable(
                SyncQueue<PipelineMessage>* input_queue,
                SyncQueue<PipelineMessage>* output_queue,
                SharedData* shared_data);
            ~CollisionCallable();

            PipelineMessage action (PipelineMessage message);
            bool check_rect_inter(carla::geom::Location point, carla::geom::Location coor1, carla::geom::Location coor2,
            carla::geom::Location coor3, carla::geom::Location coor4);
            float calculate_area (carla::geom::Location coor1, carla::geom::Location coor2,
            carla::geom::Location coor3);
    };

}