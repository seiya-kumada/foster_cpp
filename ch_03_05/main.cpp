#include <torch/torch.h>
#include "variational_auto_encoder.h"
#include "custom_dataset.h"
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

namespace
{
    const std::string DATA_DIR_PATH     {"/home/ubuntu/data/celeba/img_align_celeba"};
    constexpr int64_t TRAIN_BATCH_SIZE  {32};
    constexpr double  LEARNING_RATE     {0.0005};
    constexpr int64_t EPOCHS            {10};
    constexpr int     LOG_INTERVAL      {10};
    constexpr double  R_LOSS_FACTOR     {10000};
    const std::string OUTPUT_DIR_PATH   {"/home/ubuntu/data/foster/ch03_05/"};

    VariationalAutoEncoder make_model(const torch::Device& device)
    {
        int              encoder_start_channles     {3};
        std::vector<int64_t> before_flatten_size    {64, 8, 8};
        std::vector<int> encoder_conv_filters       {32, 64, 64,  64};
        std::vector<int> encoder_conv_kernel_sizes  { 3,  3,  3,  3};
        std::vector<int> encoder_conv_strides       { 2,  2,  2,  2};
        std::vector<int> decoder_conv_filters       {64, 64, 32,  3};
        std::vector<int> decoder_conv_kernel_sizes  { 3,  3,  3,  3};
        std::vector<int> decoder_conv_strides       { 2,  2,  2,  2};
        int              z_dim                      {200};

        return VariationalAutoEncoder{
            encoder_start_channles,
            before_flatten_size,
            std::move(encoder_conv_filters),
            std::move(encoder_conv_kernel_sizes),
            std::move(encoder_conv_strides),
            std::move(decoder_conv_filters),
            std::move(decoder_conv_kernel_sizes),
            std::move(decoder_conv_strides),
            z_dim,
            device,
            true,
            true
        };
    }

    auto calculate_kl_divergence(const torch::Tensor& mu, const torch::Tensor& log_var)
    {
        auto v = -0.5 * (1 + log_var - mu * mu - torch::exp(log_var)).sum({1});    
        return v.mean();
    }

    auto vae_loss(
        const torch::Tensor& output, 
        const torch::Tensor& mu,
        const torch::Tensor& log_var,
        const torch::Tensor& data)
        -> std::tuple<torch::Tensor, torch::Tensor>
    {
        auto r_loss = R_LOSS_FACTOR * torch::mse_loss(output, data);
        auto kl_loss = calculate_kl_divergence(mu, log_var);  
        return std::make_tuple(r_loss, kl_loss);
    }

    template<typename DataLoader>
    void train(
        size_t                      epoch,
        VariationalAutoEncoder&     model,
        torch::Device               device,
        DataLoader&                 data_loader,
        torch::optim::Optimizer&    optimizer,
        size_t                      dataset_size)
    {
        model->train();
        size_t batch_idx {0};
        torch::Tensor output {};
        torch::Tensor mu {};
        torch::Tensor log_var {};
        for (auto& batch : data_loader)
        {
            auto data = batch.data.to(device);
            optimizer.zero_grad();
            std::tie(output, mu, log_var) = model->forward(data);
            auto losses = vae_loss(output, mu, log_var, data);
            const auto& r_loss = std::get<0>(losses);
            const auto& kl_loss = std::get<1>(losses);
            auto loss = r_loss + kl_loss;

            AT_ASSERT(!std::isnan(loss.template item<float>()));
            loss.backward();
            optimizer.step();

            if (batch_idx % LOG_INTERVAL == 0)
            {
                std::printf("\rTrain Epoch: %ld [%5ld/%5ld] loss: %.4f r_loss: %.4f kl_loss: %.4f ",
                    epoch,
                    batch_idx * batch.data.size(0),
                    dataset_size,
                    loss.template item<float>(),
                    r_loss.template item<float>(),
                    kl_loss.template item<float>());
            }
            ++batch_idx;
        }
        std::cout << std::endl;
    }

    template<typename DataLoader>
    void test(
        VariationalAutoEncoder& model,
        torch::Device           device,
        DataLoader&             data_loader,
        size_t                  dataset_size,
        const std::string&      phase)
    {
        torch::NoGradGuard no_grad{};
        model->eval();
        double test_loss {0};
        torch::Tensor output {};
        torch::Tensor mu {};
        torch::Tensor log_var {};
        int s = 0; 
        for (const auto& batch : data_loader)
        {
            auto data = batch.data.to(device);
            std::tie(output, mu, log_var) = model->forward(data);
            const auto losses = vae_loss(output, mu, log_var, data);
            const auto& r_loss = std::get<0>(losses);
            const auto& kl_loss = std::get<1>(losses);
            auto loss = r_loss + kl_loss;
            test_loss += loss.template item<float>();
            s += 1;
        }

        test_loss /= s;
        std::printf(
            "%s set: Average loss: %.4f\n", phase.c_str(),
            test_loss);
    }
}

#if(UNIT_TEST)
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
namespace
{
    void print_parameters(const VariationalAutoEncoder& model)
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

    void test0()
    {
        torch::Device device{torch::kCPU};
        auto model = make_model(device);
        //print_parameters(model);
        auto x = torch::ones({3, 218, 178});
        //auto y = model->forward(x);
    }
}

BOOST_AUTO_TEST_CASE(TEST_main)
{
    std::cout << "main\n";
    test0();
}

#else

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
    
    auto model = make_model(device);
    model->to(device);

    //_/_/_/ Load the Data
    std::vector<int> input_size {128, 128};
    auto train_dataset = CustomDataset{DATA_DIR_PATH, input_size}
        .map(torch::data::transforms::Normalize<>(0, 255.0))
        .map(torch::data::transforms::Stack<>());

    const size_t train_dataset_size = train_dataset.size().value();

    const auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(TRAIN_BATCH_SIZE).workers(2));

    //_/_/_/ Configure a optimizer
    
    torch::optim::Adam optimizer {
        model->parameters(),
        torch::optim::AdamOptions(LEARNING_RATE)
    };

    //_/_/_/ Train the model

    const auto start = std::chrono::system_clock::now();
    for (auto epoch = 1; epoch <= EPOCHS; ++epoch)
    {
        train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        test(model, device, *train_loader, train_dataset_size, "Test");
    }
    const auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() 
              << " [sec]" << std::endl;

    const auto model_path = fs::path(OUTPUT_DIR_PATH) / "model.pt";
    const auto opt_path = fs::path(OUTPUT_DIR_PATH) / "optimizer.pt";
    torch::save(model, model_path.string());
    torch::save(optimizer, opt_path.string());
    
    return 0;
}

#endif // UNIT_TEST
