project(traffic_manager CXX)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER /usr/bin/clang-6.0)
set(CMAKE_CXX_COMPILER /usr/bin/clang++-6.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -DNDEBUG -g -fsanitize=address" CACHE STRING "" FORCE)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -DNDEBUG" CACHE STRING "" FORCE)

#set(LIBCARLA_LOCATION /home/praveen/opt/libcarla/)

include_directories(include/)
include_directories(/home/praveen/opt/libcarla/include)
include_directories(/home/praveen/opt/libcarla/include/system/)

link_directories(/home/praveen/opt/libcarla/lib/)

file(GLOB SOURCES src/*.cpp)
file(GLOB SOURCES_EXE test/*.cpp)

add_library(pipeline ${SOURCES})
add_executable(traffic_manager ${SOURCES_EXE})

target_link_libraries(traffic_manager pipeline carla_client rpc pthread boost_system)
