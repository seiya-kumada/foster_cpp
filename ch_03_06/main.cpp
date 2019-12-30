#include <torch/torch.h>
#include "custom_dataset.h"
#include "variational_auto_encoder.h"
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace fs = boost::filesystem;

namespace
{
    const std::string MODEL_DIR_PATH    {"/home/ubuntu/data/foster/ch03_05/"};
    const std::string OUTPUT_DIR_PATH   {"/home/ubuntu/data/foster/ch03_06/"};
    const std::string DATA_DIR_PATH     {"/home/ubuntu/data/celeba/img_align_celeba"};
    constexpr int64_t BATCH_SIZE        {10};
 
    VariationalAutoEncoder make_model(const torch::Device& device)
    {
        const int                     encoder_start_channles    {3};
        const std::vector<int64_t>    before_flatten_size       {64, 8, 8};
        const std::vector<int>        encoder_conv_filters      {32, 64, 64,  64};
        const std::vector<int>        encoder_conv_kernel_sizes { 3,  3,  3,  3};
        const std::vector<int>        encoder_conv_strides      { 2,  2,  2,  2};
        const std::vector<int>        decoder_conv_filters      {64, 64, 32,  3};
        const std::vector<int>        decoder_conv_kernel_sizes { 3,  3,  3,  3};
        const std::vector<int>        decoder_conv_strides      { 2,  2,  2,  2};
        const int                     z_dim                     {200};

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

    void save_images(const torch::Tensor& images, const std::string& subdir_name)
    {
        auto batch_size = images.size(0);
        auto dir_path = fs::path(OUTPUT_DIR_PATH) / subdir_name;
        for (auto i = 0; i < batch_size; ++i)
        {
            const auto& image = images[i];
            auto tensor = image.permute({1, 2, 0}).mul(255).clamp(0, 255).to(torch::kU8).to(torch::kCPU); // 128,128,3
            auto sizes = tensor.sizes();
            cv::Mat img {static_cast<int>(sizes[0]), static_cast<int>(sizes[1]), CV_8UC3, tensor.data<std::uint8_t>()};
            auto path = dir_path / (boost::format("%02d.jpg") % i).str();
            std::cout << path << std::endl;
            cv::imwrite(path.string(), img);
        }
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

    auto model = make_model(device);
    model->to(device);

    const auto model_path = fs::path(MODEL_DIR_PATH) / "model.pt";
    torch::load(model, model_path.string());
    model->eval();

    //_/_/_/ Load the Data

    std::vector<int> input_size {128, 128};
    auto dataset = CustomDataset{DATA_DIR_PATH, input_size}
        .map(torch::data::transforms::Normalize<>(0, 255.0))
        .map(torch::data::transforms::Stack<>());
    const size_t dataset_size = dataset.size().value();
    const auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));

    //_/_/_/ Reconstruct faces
    
    auto i = loader->begin();
    auto data = i->data.to(device);
    torch::Tensor z_points {};
    torch::Tensor mu {};
    torch::Tensor log_var {};
    std::tie(z_points, mu, log_var) = model->predict(data);
    std::cout << "z_points: " << z_points.sizes() << std::endl;
    auto reconst_images = model->get_decoder()->forward(z_points);
    std::cout << "reconst: " << reconst_images.sizes() << std::endl;
    save_images(data, "source_images");
    save_images(reconst_images, "reconst_images");
    return 0;
}
