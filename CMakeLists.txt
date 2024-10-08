cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0003 NEW)

option(RL_TOOLS_ENABLE_BOOST_BEAST OFF "Enable Boost.Beast for the UI connection to the UI server")

project(learning_to_fly_hpo)

set(CMAKE_CXX_STANDARD 17)

if(NOT BUILD_UI_SERVER_ONLY)
find_package(HDF5)
if(HDF5_FOUND)
    set(RL_TOOLS_ENABLE_HDF5 ON)
endif()
if(APPLE)
find_package(Protobuf CONFIG)
else()
find_package(Protobuf)
endif()
if(Protobuf_FOUND)
    set(RL_TOOLS_ENABLE_TENSORBOARD ON)
endif()
set(RL_TOOLS_ENABLE_CLI11 ON)

if(APPLE)
    set(RL_TOOLS_BACKEND_ENABLE_ACCELERATE ON)
endif()

find_package(MKL)
if(MKL_FOUND)
    set(RL_TOOLS_BACKEND_ENABLE_MKL ON)
elseif(NOT APPLE)
    message(WARNING "MKL not found, training will be very slow")
endif()

add_library(learning_to_fly INTERFACE)
target_include_directories(learning_to_fly INTERFACE include)
endif()
#set(RL_TOOLS_ENABLE_BOOST_BEAST ON)
set(RL_TOOLS_ENABLE_JSON ON)

#set(RL_TOOLS_ENABLE_TESTS ON)
set(RL_TOOLS_ENABLE_FAST_MATH ON)
add_subdirectory(external/rl_tools)
add_subdirectory(src)

option(LEARNING_TO_FLY_IN_SECONDS_ENABLE_TESTS "Enable tests" OFF)

if(LEARNING_TO_FLY_IN_SECONDS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()