#if(UNIT_TEST_TRAINER)
#include <boost/test/unit_test.hpp>
#include "trainer.h"
#include "gan.h"
#include "custom_dataset.h"

namespace
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

    const std::string PATH {"/home/ubuntu/projects/GDL_code/data/camel/full_numpy_bitmap_camel.npy"};
    
    template<typename T>
    struct Type;

    void test_0()
    {
        int upper_size = 121399;
        auto dataset = CustomDataset {PATH, upper_size}
            .map(torch::data::transforms::Normalize<>(127.5, 127.5))
            .map(torch::data::transforms::Stack<>());

        const size_t dataset_size = dataset.size().value();
        BOOST_CHECK_EQUAL(upper_size, dataset_size);
        int BATCH_SIZE {3};
        const auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));

        auto i = loader->begin();
        const auto& data = i->data;
        
        BOOST_CHECK_CLOSE(data[0][0][7][4].item<float>(), (38 - 127.5) / 127.5, 1.0e-4);
        BOOST_CHECK_CLOSE(data[0][0][14][8].item<float>(), (141 - 127.5) / 127.5, 1.0e-4);
    }

    void test_1()
    {
        int upper_size = 80000;
        auto dataset = CustomDataset {PATH, upper_size}
            .map(torch::data::transforms::Normalize<>(127.5, 127.5))
            .map(torch::data::transforms::Stack<>());

        const size_t dataset_size = dataset.size().value();
        BOOST_CHECK_EQUAL(upper_size, dataset_size);
    }

    void test_2()
    {
        int UPPER_SIZE = 80000;
        auto dataset = CustomDataset {PATH, UPPER_SIZE}
            .map(torch::data::transforms::Normalize<>(127.5, 127.5))
            .map(torch::data::transforms::Stack<>());

        const size_t dataset_size = dataset.size().value();
        
        BOOST_CHECK_EQUAL(UPPER_SIZE, dataset_size);

        const int Z_DIM = 100;
        GAN gan {
            discriminator_params,
            generator_params,
            std::vector<int>{2, 2, 1, 1}, // generator_upsample,
            Z_DIM,
        };   
        const int BATCH_SIZE {64};
        const int batches_per_epoch = std::ceil(dataset_size / static_cast<double>(BATCH_SIZE));

        const auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));

        torch::Device device {torch::kCUDA};
        const int EPOCHS = 1; // 6000
        const int LOG_INTERVAL = 100;
        const int SAVE_INTERVAL = 1000;

        Trainer<decltype(loader)> trainer {loader, gan, EPOCHS, device, LOG_INTERVAL, SAVE_INTERVAL, batches_per_epoch};
        trainer.train();
    }
}

BOOST_AUTO_TEST_CASE(TEST_TRAINER)
{
    std::cout << "TRAINER\n";
    test_0();
    test_1();
    test_2();
}

#endif // UNIT_TEST_TRAINER
