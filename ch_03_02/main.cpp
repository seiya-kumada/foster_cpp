#include <torch/torch.h>
#include <boost/filesystem.hpp>
#include "auto_encoder.h"
#include <random>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>

namespace fs = boost::filesystem;

namespace
{
    const std::string DATA_DIR_PATH     {"/home/ubuntu/data/mnist"};
    const std::string MODEL_DIR_PATH    {"/home/ubuntu/data/foster/ch03_01/"};
    constexpr int IMAGE_WIDTH           {32};
    constexpr int IMAGE_HEIGHT          {32};
    constexpr int IMAGE_CHANNELS        {3};
    constexpr int CLASSES               {10};
    constexpr double  LEARNING_RATE     {0.0005};
    constexpr int   TEST_BATCH_SIZE     {10};
    const std::string OUTPUT_DIR_PATH   {"/home/ubuntu/data/foster/ch03_02"};

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

    std::vector<double> generate_random(int n, int max_value)
    {
        std::mt19937 engine{0};
        std::uniform_int_distribution<> dist(0, max_value);
        std::vector<double> values(n);
        std::generate(std::begin(values), std::end(values), [&engine, &dist]() { 
            return dist(engine); 
        });
        return values;
    }

    template<typename T>
    struct Type;

    cv::Mat convert_to_mat(torch::Tensor image)
    {
        
        const auto& img = image[0]; 
        const int rows = img.size(0);
        const int cols = img.size(1);
        cv::Mat mat (rows, cols, CV_8UC1); 
        for (auto j = 0; j < rows; ++j)
        {
            auto ptr = mat.ptr<std::uint8_t>(j);
            const auto& dst = img[j];
            for (auto i = 0; i < cols; ++i)
            {
                ptr[i] = cv::saturate_cast<std::uint8_t>(255 * dst[i].item<double>());
            }
        }
        return mat;
    }

    void save_images(const torch::Tensor& tensor, const std::string& pre)
    {
        const auto size = tensor.size(0);
        fs::path dir_path {OUTPUT_DIR_PATH};
        for (auto i = 0; i < size; ++i)
        {
            const auto& image = tensor[i];
            const auto mat = convert_to_mat(image);        
            const auto path = dir_path / (boost::format("%s_%02d.jpg") % pre % i).str();
            cv::imwrite(path.string(), mat);
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

    auto model = make_model();
    model->to(device);

    //_/_/_/ Configure a optimizer
    
    torch::optim::Adam optimizer {
        model->parameters(),
        torch::optim::AdamOptions(LEARNING_RATE)
    };

    const auto model_path = fs::path(MODEL_DIR_PATH) / "model.pt";
    const auto opt_path = fs::path(MODEL_DIR_PATH) / "optimizer.pt";
    torch::load(model, model_path.string());
    torch::load(optimizer, opt_path.string());

    //_/_/_/ Load the Data

    auto test_dataset = torch::data::datasets::MNIST(
        DATA_DIR_PATH, 
        torch::data::datasets::MNIST::Mode::kTest
    ).map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), TEST_BATCH_SIZE);
    auto i = test_loader->begin();
    const auto data = i->data.to(device);
    const auto target = i->target.to(device);

    //_/_/_/ Encode 

    auto z_points = model->get_encoder()->forward(data);
    std::cout << z_points.sizes() << std::endl;

    //_/_/_/ Decode 
    
    auto reconstructed_images = model->get_decoder()->forward(z_points);
    std::cout << reconstructed_images.sizes() << std::endl;

    //_/_/_/ Convert to images

    save_images(data, "ori");
    save_images(reconstructed_images, "rec");


    return 0;
}