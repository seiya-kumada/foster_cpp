#if(UNIT_TEST_TRAINER)
#include <boost/test/unit_test.hpp>
#include "trainer.h"
#include "gan.h"
#include "custom_dataset.h"
#include <npy.hpp>

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
        torch::manual_seed(1);
        int UPPER_SIZE = 1000;
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
        const int EPOCHS = 10; // 6000
        const int LOG_INTERVAL = 1;
        const int SAVE_INTERVAL = 1000;

        Trainer<decltype(loader)> trainer {loader, gan, EPOCHS, device, LOG_INTERVAL, SAVE_INTERVAL, batches_per_epoch};
        trainer.train();
    }

    void save_as_numpy(torch::Tensor t, const std::string& name)
    {
        t = t.detach().permute({1, 2, 0}).to(torch::kFloat); // 28, 28, 1
        t = t.to(torch::kCPU);

        std::vector<float> dst(28 * 28 * 1);
        auto begin = static_cast<float*>(t.data_ptr());
        auto end = begin + (28 * 28 * 1);
        std::copy(begin, end, std::begin(dst));
        const uint64_t shape [] = {28, 28, 1};
        npy::SaveArrayAsNumpy(name, false, 3, shape, dst);
    }

    void test_3()
    {
        torch::manual_seed(1);
        int UPPER_SIZE = 800;
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
        gan->to(device);
        float d_real_loss {};
        float d_fake_loss {};
        int c = 0;
        for (const auto& batch : *loader)
        {
            const auto data = batch.data.to(device);
            const auto real_output = gan->get_discriminator()->forward(data);
            const auto real_labels = torch::ones({real_output.size(0), 1}).to(device);
            const auto real_loss = torch::binary_cross_entropy(real_output, real_labels);
     
            std::cout << real_loss.template item<float>()  << std::endl;
            if (c == 2)
            {
                break;
            }
            ++c;
        } 

    }
}

BOOST_AUTO_TEST_CASE(TEST_TRAINER)
{
    std::cout << "TRAINER\n";
    //test_0();
    //test_1();
    test_2();
    //test_3();

}

#endif // UNIT_TEST_TRAINER
