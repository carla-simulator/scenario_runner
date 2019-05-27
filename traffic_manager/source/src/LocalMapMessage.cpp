//Definition of class members

#include "LocalMapMessage.hpp"

namespace traffic_manager {

    LocalMapMessage::LocalMapMessage(InMemoryMap* memory_map): memory_map(memory_map){}

    LocalMapMessage::~LocalMapMessage(){}
    
}