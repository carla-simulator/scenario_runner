//Declaration of class members
#pragma once

#include "PipelineMessage.hpp"
#include "InMemoryMap.hpp"

namespace traffic_manager {

    class LocalMapMessage: public PipelineMessage
    {
    public:
        LocalMapMessage(InMemoryMap* memory_map);
        ~LocalMapMessage();
        InMemoryMap* memory_map;
    };    
}