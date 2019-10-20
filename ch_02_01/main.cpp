#if(UNIT_TEST)
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>

BOOST_AUTO_TEST_CASE(TEST_main)
{
    std::cout << "main\n";
}

#else

//https://discuss.pytorch.org/t/libtorch-does-not-link-together-with-boost/27956/9
#include "custom_dataset.h"
#include <boost/filesystem.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>

namespace fs = boost::filesystem;

namespace
{
    const fs::path DIR_PATH = fs::path("/home/ubuntu/data/cifar-10/cifar-10-batches-bin/");

    const std::vector<fs::path> TRAIN_PATHS 
    {
        DIR_PATH / "data_batch_1.bin",
        DIR_PATH / "data_batch_2.bin",
        DIR_PATH / "data_batch_3.bin",
        DIR_PATH / "data_batch_4.bin",
        DIR_PATH / "data_batch_5.bin",
    };

    const std::vector<fs::path> TEST_PATHS 
    {
        DIR_PATH / "test_batch.bin",
    };

    auto load_dataset(const std::vector<fs::path>& paths)
        -> CustomDataset
    {
        std::vector<std::ifstream> ifss{};
        boost::copy(
            paths | boost::adaptors::transformed(
                [](const auto& p)
                {
                    return std::ifstream{p.string(), std::ios::binary};
                }
            ), 
            std::back_inserter(ifss)
        );
        return {ifss};
    }

    constexpr int BATCH_SIZE = 32;
}


int main()
{
    //_/_/_/ Loading the Data

    auto train_dataset = load_dataset(TRAIN_PATHS).
            map(torch::data::transforms::Normalize<>(0, 255.0)).
            map(torch::data::transforms::Stack<>());

    auto test_dataset = load_dataset(TEST_PATHS).
            map(torch::data::transforms::Normalize<>(0, 255.0)).
            map(torch::data::transforms::Stack<>());

    auto train_data_loader = torch::data::make_data_loader(
            std::move(train_dataset),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));

    auto test_data_loader = torch::data::make_data_loader(
            std::move(test_dataset),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));
}
#endif // UNIT_TEST
