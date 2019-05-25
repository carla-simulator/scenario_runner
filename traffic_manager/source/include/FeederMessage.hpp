//Declaration of FeederMessage class members
#pragma once

#include "PipelineMessage.hpp"

namespace traffic_manager{

    class FeederMessage: public PipelineMessage
    {
        FeederMessage();
        ~FeederMessage();
    };
}