add_executable(training training.cpp)
target_link_libraries(
        training
        PRIVATE
        RLtools::Backend
        HighFive
        nlohmann_json
        learning_to_fly
        CLI11::CLI11
)
if(RL_TOOLS_ENABLE_TENSORBOARD)
    target_link_libraries(training PRIVATE tensorboard_logger)
endif()

if(RL_TOOLS_ENABLE_BOOST_BEAST)
add_executable(evaluation_ui evaluate_actor.cpp)
target_link_libraries(
        evaluation_ui
        PRIVATE
        rl_tools
        learning_to_fly
        CLI11::CLI11
)
endif()

add_executable(evaluation evaluate_actor_headless.cpp)
target_link_libraries(
        evaluation
        PRIVATE
        rl_tools
        learning_to_fly
        CLI11::CLI11
)
