add_executable(training_headless training.cpp)
target_link_libraries(
        training_headless
        PRIVATE
        RLtools::Backend
        HighFive
        nlohmann_json
        learning_to_fly
        CLI11::CLI11
)
if(RL_TOOLS_ENABLE_TENSORBOARD)
    target_link_libraries(training_headless tensorboard_logger)
endif()


add_executable(evaluate_actor_headless evaluate_actor_headless.cpp)
target_link_libraries(
        evaluate_actor_headless
        PRIVATE
        rl_tools
        learning_to_fly
        CLI11::CLI11
)
