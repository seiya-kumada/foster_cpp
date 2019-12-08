#if(UNIT_TEST)
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <chrono>
#include <string>

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
#include <boost/program_options.hpp>

namespace fs = boost::filesystem;
namespace po = boost::program_options;

namespace
{
    const fs::path DIR_PATH             {"/home/ubuntu/data/cifar-10/cifar-10-batches-bin/"};
    const std::string OUTPUT_DIR_PATH   {"/home/ubuntu/data/foster/ch02_01"};
    constexpr int LOG_INTERVAL      {10};
    constexpr int IMAGE_WIDTH       {32};
    constexpr int IMAGE_HEIGHT      {32};
    constexpr int IMAGE_CHANNELS    {3};
    constexpr int CLASSES           {10};
    constexpr double LEARNING_RATE  {0.0005};

    po::options_description parse_arguments()
    {
        po::options_description desc {"ch02_01"};
        desc.add_options()
            ("help", "produce help messsage")
            ("batch_size", po::value<int>(), "set batch size")
            ("epochs", po::value<int>(), "set epochs")
            ("resume", po::value<bool>()->default_value(false), "set true or false")
            ("model_path", po::value<std::string>(), "set a path to the model")
            ("opt_path", po::value<std::string>(), "set a path to the optimizer")
            ("trained_model_path", po::value<std::string>(), "set a path to the trained model")
            ("trained_opt_path", po::value<std::string>(), "set a path to the trained optimizer")
            ("verbose", po::value<bool>()->default_value(false), "set true or false")
        ;
        return desc;
    }

    template<typename T>
    inline const T extract_parameter(
        const std::string& name,
        const std::string& message,
        const po::variables_map& vm)
    {
        if (vm.count(name))
        {
            return vm[name].as<T>();
        }
        else
        {
            throw std::runtime_error(message);
        }
    }

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

    CustomDataset load_dataset(const std::vector<fs::path>& paths)
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

    template<typename DataLoader>
    void train(
        size_t                      epoch,
        Architecture&               model,
        torch::Device               device,
        DataLoader&                 data_loader,
        torch::optim::Optimizer&    optimizer,
        size_t                      dataset_size,
        bool                        verbose)
    {
        model->train();
        size_t batch_idx {0};
        for (auto& batch : data_loader)
        {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);
            optimizer.zero_grad();
            
            auto output = model->forward(data);
            auto pred = output.argmax(1);
            
            auto batch_size = output.size(0);
            targets = targets.reshape({batch_size});
            auto correct = pred.eq(targets).sum().template item<int64_t>();
            
            auto loss = torch::nll_loss(output, targets);
            // we should be able to use cross_entropy_loss -> unfortunately, cross_entropy_loss is not implemented in c++.
            AT_ASSERT(!std::isnan(loss.template item<float>()));
            loss.backward();
            optimizer.step();

            if (verbose && (batch_idx % LOG_INTERVAL == 0))
            {
                std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f | Accuracy: %.3f",
                    epoch,
                    batch_idx * batch.data.size(0),
                    dataset_size,
                    loss.template item<float>(),
                    static_cast<double>(correct) / batch_size);
            }

            ++batch_idx;
        }
    }

    template<typename DataLoader>
    void test(
        Architecture&   model,
        torch::Device   device,
        DataLoader&     data_loader,
        size_t          dataset_size,
        bool            verbose)
    {
        torch::NoGradGuard no_grad{};
        model->eval();
        double test_loss {0};
        int32_t correct {0};
        for (const auto& batch : data_loader)
        {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);
            auto output = model->forward(data);
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
        if (verbose)
        {
            std::printf(
                "\nTest set: Average loss: %.4f | Accuracy: %.3f\n\n",
                test_loss,
                static_cast<double>(correct) / dataset_size);
        }
    }

    void print_parameters(const Architecture& model)
    {
        int s {0};
        for (const auto& pair : model->named_parameters())
        {
            const auto& key = pair.key();
            const auto& value = pair.value();
            //<< ": " << pair.value().sizes() << std::endl;
            auto c = 1;
            for (const auto& v : value.sizes())
            {
                c *= v; 
            }
            std::cout << key << ": " << pair.value().sizes() << " -> " << c << std::endl;
            s += c;
        }
        std::cout << "total number of parameters: " << s << std::endl;
    }
}

int main(int argc, const char* argv[])
{
    //_/_/_/ Extract arguments
    
    auto desc = parse_arguments();
    po::variables_map vm {};
    po::store(parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.empty() || vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    const auto batch_size = extract_parameter<int>("batch_size", "invalid batch size", vm);
    const auto epochs = extract_parameter<int>("epochs", "invalid epochs", vm);
    const auto resumes = extract_parameter<bool>("resume", "invalid resume", vm);
    const auto model_path = extract_parameter<std::string>("model_path", "invalid path", vm);
    const auto opt_path = extract_parameter<std::string>("opt_path", "invalid path", vm);
    const auto verbose = extract_parameter<bool>("verbose", "invalid verbose", vm);
    
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

    Architecture model{IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS, CLASSES};
    model->to(device);
    print_parameters(model);

    //_/_/_/ Load the Data

    const auto train_dataset = load_dataset(TRAIN_PATHS).
            map(torch::data::transforms::Normalize<>(0, 255.0)).
            map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();

    const auto test_dataset = load_dataset(TEST_PATHS).
            map(torch::data::transforms::Normalize<>(0, 255.0)).
            map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
   
    // memo: where is the shuffle executed?
    const auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

    const auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
    
    //_/_/_/ Configure a optimizer
    
    torch::optim::Adam optimizer {
        model->parameters(),
        torch::optim::AdamOptions(LEARNING_RATE)
    };

    //_/_/_/ Train the model
    
    if (resumes)
    {
        std::cout << "resume training!" << std::endl;
        const auto trained_model_path = extract_parameter<std::string>("trained_model_path", "invalid path", vm);
        const auto trained_opt_path = extract_parameter<std::string>("trained_opt_path", "invalid path", vm);
        torch::load(model, trained_model_path);
        torch::load(optimizer, trained_opt_path);
    }

    const auto start = std::chrono::system_clock::now();
    for (auto epoch = 1; epoch <= epochs; ++epoch)
    {
        train(epoch, model, device, *train_loader, optimizer, train_dataset_size, verbose);
        test(model, device, *test_loader, test_dataset_size, verbose);
    }
    const auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " [sec]" << std::endl;

    //_/_/_/ Save the model
    
    torch::save(model, model_path);
    torch::save(optimizer, opt_path);
}
#endif // UNIT_TEST
