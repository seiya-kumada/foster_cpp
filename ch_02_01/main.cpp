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
#include "architecture.h"
#include <torch/torch.h>
//#include <ATen/ATen.h>
//#include <ATen/NativeFunctions.h>
//#include <ATen/Dispatch.h>
//#include <ATen/CPUApplyUtils.h>

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

    constexpr int BATCH_SIZE = 24;
    constexpr int EPOCHS = 20;
    constexpr int LOG_INTERVAL = 10;

    template<typename DataLoader>
    void train(
        size_t                      epoch,
        Architecture&               model,
        torch::Device               device,
        DataLoader&                 data_loader,
        torch::optim::Optimizer&    optimizer,
        size_t                      dataset_size)
    {
        model.train();
        size_t batch_idx {0};
        for (auto& batch : data_loader)
        {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);
            optimizer.zero_grad();
            auto output = model.forward(data);

            auto batch_size = output.size(0);
            targets = targets.reshape({batch_size});
            auto loss = torch::nll_loss(output, targets);
            // ここはクラスエントロピーを使えるはず。-> c++にはcross_entropy_lossが実装されていない。
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
    }

    template<typename DataLoader>
    void test(
        Architecture&   model,
        torch::Device   device,
        DataLoader&     data_loader,
        size_t          dataset_size)
    {
        torch::NoGradGuard no_grad{};
        model.eval();
        double test_loss {0};
        int32_t correct {0};
        for (const auto& batch : data_loader)
        {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);
            auto output = model.forward(data);
            auto batch_size = output.size(0);
            targets = targets.reshape({batch_size});
            test_loss += torch::nll_loss(
                output,
                targets,
                {},
                Reduction::Reduction::Sum).template item<float>();
            auto pred = output.argmax(1);
            correct += pred.eq(targets).sum().template item<int64_t>();
        }

        test_loss /= dataset_size;
        std::printf(
            "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
            test_loss,
            static_cast<double>(correct) / dataset_size);
    }
}


auto main() -> int
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
    
    Architecture model{32 * 32 * 3, 10};
    model.to(device);

    //_/_/_/ Load the Data

    auto train_dataset = load_dataset(TRAIN_PATHS).
            map(torch::data::transforms::Normalize<>(0, 255.0)).
            map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();

    auto test_dataset = load_dataset(TEST_PATHS).
            map(torch::data::transforms::Normalize<>(0, 255.0)).
            map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    
    auto train_loader = torch::data::make_data_loader(
            std::move(train_dataset),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));

    auto test_loader = torch::data::make_data_loader(
            std::move(test_dataset),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));

    //std::cout <<  train_dataset_size << " " << test_dataset_size << std::endl;
    
    //_/_/_/ Configure a optimizer
    
    torch::optim::Adam optimizer {
        model.parameters(),
        torch::optim::AdamOptions(0.0005)
    };

    //_/_/_/ Train the model

    for (auto epoch = 1; epoch <= EPOCHS; ++epoch)
    {
        std::cout << "> epoch: " <<  epoch << std::endl;
        train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        test(model, device, *test_loader, test_dataset_size);
    }
}
#endif // UNIT_TEST
