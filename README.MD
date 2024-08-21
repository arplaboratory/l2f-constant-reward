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
TBD
## Build
```
mkdir build
cd build
```

```
cmake ..
cmake --build . --target training_headless evaluate_actor_headless
```
```
cd ..
```
## Run
```
./build/src/training_headless --parameter-file parameters/output/crazyflie.json --num-runs 1
./build/src/evaluate_actor_headless
```