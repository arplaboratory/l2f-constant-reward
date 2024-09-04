 find checkpoints/multirotor_td3/ | sort | grep h5$ | xargs -I{} ./build/src/evaluation --checkpoint {}


 # ./tools/learning_curves.sh > results_error_integral.txt