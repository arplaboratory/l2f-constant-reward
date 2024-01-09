
if(NOT BUILD_UI_SERVER_ONLY)
add_executable(training_headless_x500 training.cpp)
target_link_libraries(
        training_headless_x500
        PRIVATE
        rl_tools
        learning_to_fly
)
target_compile_definitions(training_headless_x500 PRIVATE LEARNING_TO_FLY_SQUARED_REWARD_FUNCTION)
target_compile_definitions(training_headless_x500 PRIVATE LEARNING_TO_FLY_WEIGHT_DECAY)

add_executable(training_headless training.cpp)
target_link_libraries(
        training_headless
        PRIVATE
        rl_tools
        learning_to_fly
)

add_executable(training_benchmark training.cpp)
target_link_libraries(
        training_benchmark
        PRIVATE
        rl_tools
        learning_to_fly
)
target_compile_definitions(training_benchmark PRIVATE LEARNING_TO_FLY_IN_SECONDS_BENCHMARK)

add_executable(evaluate_actor evaluate_actor.cpp)
target_link_libraries(
        evaluate_actor
        PRIVATE
        rl_tools
        learning_to_fly
)

add_executable(hpo training.cpp)
target_link_libraries(
        hpo
        PRIVATE
        rl_tools
        learning_to_fly
)
target_compile_definitions(hpo PRIVATE LEARNING_TO_FLY_HYPERPARAMETER_OPTIMIZATION)
target_compile_definitions(hpo PRIVATE LEARNING_TO_FLY_IN_SECONDS_BENCHMARK)

add_executable(validation validation.cpp)
target_link_libraries(
        validation
        PRIVATE
        rl_tools
        learning_to_fly
)


if(RL_TOOLS_ENABLE_HDF5)
add_executable(ablation_study ablation_study.cpp)
target_link_libraries(
        ablation_study
        PRIVATE
        rl_tools
        learning_to_fly
)
endif()

if(LEARNING_TO_FLY_ENABLE_OLD_UI)
add_executable(evaluate_actor evaluate_actor.cpp)
target_link_libraries(
        evaluate_actor
        PRIVATE
        rl_tools
        learning_to_fly
)
endif()


if(RL_TOOLS_BACKEND_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
endif()

add_subdirectory(ui)
