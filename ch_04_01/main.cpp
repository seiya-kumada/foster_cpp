#include <torch/torch.h>
#include <gan.h>
// https://github.com/pytorch/examples/blob/master/cpp/dcgan/dcgan.cpp
namespace
{
    std::unique_ptr<torch::optim::Optimizer> select_optimizer(const std::string& name, GAN& gan, double learning_rate)
    {
        if (name == "adam")
        {
            return std::make_unique<torch::optim::Adam>(gan->parameters(), torch::optim::AdamOptions(learning_rate).beta1(0.5));
        }
        else if (name == "rmsprop")
        {
            return std::make_unique<torch::optim::RMSprop>(gan->parameters(), torch::optim::RMSpropOptions(learning_rate));
        }
        else
        {
            return std::make_unique<torch::optim::Adam>(gan->parameters(), torch::optim::AdamOptions(learning_rate));
        }
    }

    void train_discriminator(
        const torch::Tensor& real_image, 
        GAN& gan,
        const torch::Device& device,
        std::unique_ptr<torch::optim::Optimizer>& optimizer_for_discriminator,
        const torch::Tensor& real_labels,
        const torch::Tensor& fake_labels)
    {
        gan->get_discriminator()->train();
        optimizer_for_discriminator->zero_grad();
        
        // consider real image
        auto real_output = gan->get_discriminator()->forward(real_image);
        auto real_loss = torch::binary_cross_entropy(real_output, real_labels); 
        real_loss.backward();

        // consider generated image
        auto noise = torch::randn({real_image.size(0), gan->get_z_dim()}, device);
        auto fake_image = gan->get_generator()->forward(noise);
        // http://aidiary.hatenablog.com/entry/20180304/1520172429: detachの意味が書いてある。
        auto fake_output = gan->get_discriminator()->forward(fake_image.detach());
        auto fake_loss = torch::binary_cross_entropy(fake_output, fake_labels); 
        fake_loss.backward();

        auto loss = real_loss + fake_loss;
        //loss.backward(); それともこう書くべきか？
        //個々に最小にするか、和を最小にするか？の違い。
        optimizer_for_discriminator->step();
    }

    template<typename Loader>
    void train(
        int epoch, 
        GAN& gan, 
        const torch::Device& device,
        const torch::Tensor& real_labels,
        const torch::Tensor& fake_labels,
        Loader& loader, 
        std::unique_ptr<torch::optim::Optimizer>& optimizer_for_dis,
        std::unique_ptr<torch::optim::Optimizer>& optimizer_for_gen,
        std::size_t dataset_size)
    {
        for (auto& batch : loader)
        {
            auto data = batch.data.to(device);
            train_discriminator(data, gan, device, optimizer_for_dis, real_labels, fake_labels);
            break;
        }
    }
}

#if(UNIT_TEST)
#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
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
    }

    void test_1()
    {
        auto dataset = CustomDataset {PATH}
            .map(torch::data::transforms::Normalize<>(127.5, 127.5))
            .map(torch::data::transforms::Stack<>());

        const size_t dataset_size = dataset.size().value();
        
        BOOST_CHECK_EQUAL(121399, dataset_size);

        GAN gan {
            discriminator_params,
            generator_params,
            std::vector<int>{2, 2, 1, 1}, // generator_upsample,
            100 // z_dim,
        };   

        int BATCH_SIZE {3};
        const auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));

        auto optimizer_for_discriminator = select_optimizer("rmsprop", gan, discriminator_params.learning_rate_);
        auto optimizer_for_generator = select_optimizer("rmsprop", gan, generator_params.learning_rate_);
        int epoch = 1;
        torch::Device device {torch::kCUDA};
        gan->to(device);
        auto real_labels = torch::ones({BATCH_SIZE, 1}).to(device);
        auto fake_labels = torch::zeros({BATCH_SIZE, 1}).to(device);
        train(epoch, gan, device, real_labels, fake_labels, *loader, optimizer_for_discriminator, optimizer_for_generator, dataset_size);
    }
}

BOOST_AUTO_TEST_CASE(TEST_main)
{
    std::cout << "main\n";
    test_0();
    test_1();
}

#else
#include <iostream>

int main(int argc, const char* argv[])
{
    std::cout << "hello world\n";
    return 0;
}

#endif // UNIT_TEST