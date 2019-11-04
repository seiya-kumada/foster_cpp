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
#include "auto_encoder.h"

namespace
{
    const std::string DATA_DIR_PATH {"/home/ubuntu/data/mnist"};
    constexpr int64_t TRAIN_BATCH_SIZE {32};
    constexpr int64_t TEST_BATCH_SIZE {32};
    constexpr double  LEARNING_RATE {0.0005};
    constexpr int64_t EPOCHS {10};
    constexpr int LOG_INTERVAL      {10};

    AutoEncoder make_model()
    {
        std::vector<int> encoder_conv_filters       {32, 64, 64,  64};
        std::vector<int> encoder_conv_kernel_sizes  { 3,  3,  3,  3};
        std::vector<int> encoder_conv_strides       { 1,  2,  2,  1};
        std::vector<int> decoder_conv_filters       {64, 64, 32,  1};
        std::vector<int> decoder_conv_kernel_sizes  { 3,  3,  3,  3};
        std::vector<int> decoder_conv_strides       { 1,  2,  2,  1};
        int z_dim {2};

        return AutoEncoder{  
            std::move(encoder_conv_filters),
            std::move(encoder_conv_kernel_sizes),
            std::move(encoder_conv_strides),
            std::move(decoder_conv_filters),
            std::move(decoder_conv_kernel_sizes),
            std::move(decoder_conv_strides),
            z_dim 
        };
    }

    template<typename DataLoader>
    void train(
        size_t                      epoch,
        AutoEncoder&                model,
        torch::Device               device,
        DataLoader&                 data_loader,
        torch::optim::Optimizer&    optimizer,
        size_t                      dataset_size)
    {
        model->train();
        size_t batch_idx {0};
        for (auto& batch : data_loader)
        {
            auto data = batch.data.to(device);
            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = torch::mse_loss(output, data);
            AT_ASSERT(!std::isnan(loss.template item<float>()));
            loss.backward();
            optimizer.step();

            if (batch_idx % LOG_INTERVAL == 0)
            {
                std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
                    epoch,
                    batch_idx * batch.data.size(0),
                    dataset_size,
                    loss.template item<float>());
            }
            ++batch_idx;
        }
        std::cout << std::endl;
    }

    template<typename DataLoader>
    void test(
        AutoEncoder&    model,
        torch::Device   device,
        DataLoader&     data_loader,
        size_t          dataset_size)
    {
        torch::NoGradGuard no_grad{};
        model->eval();
        double test_loss {0};
        for (const auto& batch : data_loader)
        {
            auto data = batch.data.to(device);
            auto output = model->forward(data);
            test_loss += torch::mse_loss(
                output,
                data 
            ).template item<float>();
        }

        test_loss /= dataset_size;
        std::printf(
            "\nTest set: Average loss: %.4f\n",
            test_loss);
    }
}

int main(int argc, const char* argv[])
{
    //_/_/_/ Select device

    torch::manual_seed(1);
    torch::DeviceType device_type{};
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device{device_type};

    //_/_/_/ Define a model
    
    auto model = make_model();
    model->to(device);

    //_/_/_/ Load the Data
    
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

    //_/_/_/ Configure a optimizer
    
    torch::optim::Adam optimizer {
        model->parameters(),
        torch::optim::AdamOptions(LEARNING_RATE)
    };

    //_/_/_/ Train the model

    for (auto epoch = 1; epoch <= EPOCHS; ++epoch)
    {
        train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    }
    //test(model, device, *test_loader, test_dataset_size);

    return 0;
}
#endif // UNIT_TEST
