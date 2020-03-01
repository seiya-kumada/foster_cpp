#if(UNIT_TEST)
#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
#include "gan.h"
#include "custom_dataset.h"

namespace
{
    void test_0()
    {
        GANImpl::Params discriminator_params {
            1, // start_channels
            {128, 4, 4}, // flatten_shape
            {64, 64, 128, 128}, // conv_filters
            {5, 5, 5, 5}, // kernel_size
            {2, 2, 2, 1}, // strides             
            boost::none, // batch_norm_momentum
            "relu", // activation          
            0.4, // dropout_rate       
            0.0008 // learning_rate  
        };

        GANImpl::Params generator_params {
            1, // start_channels
            {64, 7, 7}, // flatten_shape 
            {128, 64, 64, 1}, // conv_filters
            {5, 5, 5, 5}, // kernel_size
            {1, 1, 1, 1}, // strides             
            0.9, // batch_norm_momentum
            "relu", // activation          
            boost::none, // dropout_rate       
            0.0004 // learning_rate  
        };

        torch::manual_seed(1);
        GAN gan {
            discriminator_params,
            generator_params,
            std::vector<int>{2, 2, 1, 1}, // generator_upsample,
            100 // z_dim,
        };

        const std::string PATH {"/home/ubuntu/projects/GDL_code/data/camel/full_numpy_bitmap_camel.npy"};
        auto dataset = CustomDataset {PATH}
            .map(torch::data::transforms::Normalize<>(127.5, 127.5))
            .map(torch::data::transforms::Stack<>());

        const size_t dataset_size = dataset.size().value();
        BOOST_CHECK_EQUAL(121399, dataset_size);
        int BATCH_SIZE {3};
        const auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));
        
        auto i = loader->begin();
        const auto& data = i->data;
        BOOST_CHECK_CLOSE(data[0][0][7][4].item<float>(), (38 - 127.5) / 127.5, 1.0e-4);
        BOOST_CHECK_CLOSE(data[0][0][14][8].item<float>(), (141 - 127.5) / 127.5, 1.0e-4);

        int epoch = 1;
        //torch::Device device {torch::kGPU};
        //train(gan, device, *loader, dataset_size);
    }
}

BOOST_AUTO_TEST_CASE(TEST_main)
{
    std::cout << "main\n";
    test_0();
}

#else
#include <iostream>

int main(int argc, const char* argv[])
{
    std::cout << "hello world\n";
    return 0;
}

#endif // UNIT_TEST
