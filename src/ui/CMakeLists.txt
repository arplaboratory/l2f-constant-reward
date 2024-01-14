

if(RL_TOOLS_ENABLE_JSON AND RL_TOOLS_ENABLE_BOOST_BEAST)
find_package(Boost REQUIRED)
message(STATUS "Boost version: ${Boost_VERSION}")
if(NOT BUILD_UI_SERVER_ONLY)
    add_executable(ui ui.cpp)
    target_link_libraries(
            ui
            PRIVATE
            rl_tools
            learning_to_fly
            Boost::boost
    )
endif()
endif()

