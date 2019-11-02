#if(UNIT_TEST)
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(TEST_main)
{
    std::cout << "main\n";
}

#else



#include <torch/torch.h>
#include <string>
#include <iostream>

namespace
{
    const std::string DATA_DIR_PATH {"/home/ubuntu/data/mnist"};
    constexpr int64_t TRAIN_BATCH_SIZE {64};
    constexpr int64_t TEST_BATCH_SIZE {64};
}

int main(int argc, const char* argv[])
{
    auto train_dataset = torch::data::datasets::MNIST(
        DATA_DIR_PATH,
        torch::data::datasets::MNIST::Mode::kTrain)
        .map(torch::data::transforms::Stack<>());

    const size_t train_dataset_size = train_dataset.size().value();
    std::cout << train_dataset_size << std::endl;

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_dataset), 
        TRAIN_BATCH_SIZE);

    auto test_dataset = torch::data::datasets::MNIST(
        DATA_DIR_PATH, 
        torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Stack<>());

    const size_t test_dataset_size = test_dataset.size().value();
    std::cout << test_dataset_size << std::endl;
    
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), TEST_BATCH_SIZE);


    return 0;
}
#endif // UNIT_TEST
