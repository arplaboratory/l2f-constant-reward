# Learning to Fly Using a Constant Reward Function
## Dependencies
### Ubuntu
Tested in a vanilla Ubuntu 24.04 Docker container
```
apt update
apt install -y build-essential cmake libhdf5-dev libboost-dev libboost-system-dev libboost-serialization-dev
```
Please also install Intel MKL for fast training (https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)
```
apt install -y gpg-agent wget
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
apt update
apt install -y intel-oneapi-mkl-devel
export MKL_ROOT=/opt/intel/oneapi/mkl/latest
```

### macOS
```
brew install cmake hdf5 boost
```
Apple Accelerate (the MKL equivalent) is installed by default on macOS and should be detected by the build system automatically
## Build
```
mkdir build
cd build
```

```
cmake ..
cmake --build . --target training evaluation
```
```
cd ..
```
## Run
```
./build/src/training --parameter-file parameters/output/crazyflie.json --num-runs 1
./build/src/evaluation
```

## UI
```
cmake .. -DRL_TOOLS_ENABLE_BOOST_BEAST=ON
cmake --build . --target evaluation_ui
docker run -it --rm -p 8000:8000 rltools/ui-server:v0.1-l2f
./build/src/evaluation_ui
```