set -e

g++ import_export.cpp -I../../external/rl_tools/include --std=c++17 -I../../external/rl_tools/external/highfive/include -I/usr/include/hdf5/serial/ -I../../external/rl_tools/external/cli11/include -lhdf5_serial -o import_export

./import_export --model actor_simulation_optimization.h5