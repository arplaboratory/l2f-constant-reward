namespace learning_to_fly {
    namespace steps {
        template <typename T_CONFIG>
        void logger(TrainingState<T_CONFIG>& ts){
            rlt::set_step(ts.device, ts.device.logger, ts.step);
//            if(ts.step % 100000 == 0){
//                std::cout << "Logged topics:" << std::endl;
//                for (const auto &[key, value] : ts.device.logger.topic_frequency_dict) {
//                    std::cout << "Key: " << key << ", Value: " << value << std::endl;
//                }
//            }
        }
    }
}
