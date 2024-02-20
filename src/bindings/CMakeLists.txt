add_executable(bindings_test l2f.cpp)
target_compile_definitions(bindings_test PRIVATE TEST)
target_link_libraries(
        bindings_test
        PRIVATE
        rl_tools
        learning_to_fly
)


add_executable(bindings_import_export import_export.cpp)
target_compile_definitions(bindings_import_export PRIVATE TEST)
target_link_libraries(
        bindings_import_export
        PRIVATE
        RLtools::Minimal
        CLI11::CLI11
)

add_executable(test_actor_simulation_optimization test_actor_simulation_optimization.cpp)
target_compile_definitions(test_actor_simulation_optimization PRIVATE TEST)
target_link_libraries(
        test_actor_simulation_optimization
        PRIVATE
        RLtools::Minimal
        CLI11::CLI11
)