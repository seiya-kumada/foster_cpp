#include <torch/torch.h>
#include <boost/filesystem.hpp>
#include "variational_auto_encoder.h"
#include <random>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>

namespace fs = boost::filesystem;

namespace
{
    const std::string DATA_DIR_PATH     {"/home/ubuntu/data/mnist"};
    const std::string MODEL_DIR_PATH    {"/home/ubuntu/data/foster/ch03_03/"};
    constexpr int IMAGE_WIDTH           {32};
    constexpr int IMAGE_HEIGHT          {32};
    constexpr int IMAGE_CHANNELS        {3};
    constexpr int CLASSES               {10};
    constexpr double  LEARNING_RATE     {0.0005};
    constexpr int   TEST_BATCH_SIZE     {5000};
    const std::string OUTPUT_DIR_PATH   {"/home/ubuntu/data/foster/ch03_04"};

    VariationalAutoEncoder make_model(const torch::Device& device)
    {
        std::vector<int> encoder_conv_filters       {32, 64, 64,  64};
        std::vector<int> encoder_conv_kernel_sizes  { 3,  3,  3,  3};
        std::vector<int> encoder_conv_strides       { 1,  2,  2,  1};
        std::vector<int> decoder_conv_filters       {64, 64, 32,  1};
        std::vector<int> decoder_conv_kernel_sizes  { 3,  3,  3,  3};
        std::vector<int> decoder_conv_strides       { 1,  2,  2,  1};
        int z_dim {2};

        return VariationalAutoEncoder{  
            std::move(encoder_conv_filters),
            std::move(encoder_conv_kernel_sizes),
            std::move(encoder_conv_strides),
            std::move(decoder_conv_filters),
            std::move(decoder_conv_kernel_sizes),
            std::move(decoder_conv_strides),
            z_dim,
            device
        };
    }

    std::vector<float> generate_random(int n, float min_value, float max_value)
    {
        std::random_device seed_gen;
        std::mt19937 engine{seed_gen()};
        std::uniform_real_distribution<> dist(min_value, max_value);
        std::vector<float> values(n);
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
            const auto path = dir_path / (boost::format("%s_%03d.jpg") % pre % i).str();
            cv::imwrite(path.string(), mat);
        }
    }

    template<typename T>
    void reconstruct_original_paintings(
        const torch::Device&    device, 
        const T&                test_dataset, 
        VariationalAutoEncoder&            model, 
        int                     batch_size)
    {
        auto test_loader = torch::data::make_data_loader(std::move(test_dataset), batch_size);
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
    }

    void save_z_points(const torch::Tensor& points, const torch::Tensor& target)
    {
        fs::path dir_path {OUTPUT_DIR_PATH};
        const auto path = dir_path / "z_points.txt";
        std::ofstream ofs {path.string()};
        auto size = points.size(0);
        for (auto i = 0; i < size; ++i)
        {
            const auto& pair = points[i];
            const auto x = pair[0].item<double>();
            const auto y = pair[1].item<double>();
            ofs << x << " " << y << std::endl;
        }
        
        const auto target_path = dir_path / "targets.txt";
        std::ofstream target_ofs {target_path.string()};
        for (auto i = 0; i < size; ++i)
        {
            const auto t = target[i].item<int>();
            target_ofs << t << std::endl;
        }
    }

    template<typename T>
    void make_distribution_in_latent_space(
        const torch::Device&    device, 
        const T&                test_dataset, 
        VariationalAutoEncoder& model, 
        int                     batch_size)
    {
        auto test_loader = torch::data::make_data_loader(std::move(test_dataset), batch_size);
        auto i = test_loader->begin();
        const auto data = i->data.to(device);
        const auto target = i->target.to(device);

        //_/_/_/ Encode 

        auto out = model->predict(data);

        //_/_/_/  
        
        save_z_points(std::get<0>(out), target);
    }

    void save_points(const std::vector<float>& xs, const std::vector<float>& ys, const std::string name)
    {
        fs::path dir_path {OUTPUT_DIR_PATH};
        const auto path = dir_path / name;
        std::ofstream ofs {path.string()};
        for (auto i = 0; i < xs.size(); ++i)
        {
            ofs << xs[i] << " " << ys[i] << std::endl;
        }
    }

    void save_grid_points(const torch::Tensor& vs, const std::string name)
    {
        fs::path dir_path {OUTPUT_DIR_PATH};
        const auto path = dir_path / name;
        std::ofstream ofs {path.string()};
        auto s = vs.size(0);
        for (auto i = 0; i < s; ++i)
        {
            const auto v = vs[i];
            const auto x = v[0].item<float>();
            const auto y = v[1].item<float>();
            ofs << x << " " << y << std::endl;
        }
    }


    void generate_new_arts(
        const torch::Device&    device, 
        VariationalAutoEncoder&            model)
    {
        //_/_/_/ Generate 2-dim coordinates 

        auto max_x = 4.6541;
        auto min_x = -5.7535;
        auto max_y = 5.05247;
        auto min_y = -4.38811;    

        const auto xs = generate_random(30, min_x, max_x);
        const auto ys = generate_random(30, min_y, max_y);

        save_points(xs, ys, "generated_points.txt");

        //_/_/_/ Decode
        
        auto z_points = torch::ones({30, 2}, torch::kFloat);
        for (int i = 0; i < 30; ++i)
        {
            auto row = z_points[i];
            row[0] = xs[i];
            row[1] = ys[i];
        }
        auto zps = z_points.to(device);
        auto generated_images = model->get_decoder()->forward(zps); 
        save_images(generated_images, "generated");
    }

    std::vector<float> generate_range(int n, double min_value, double max_value)
    {
        auto step = (max_value - min_value) / n;
        std::vector<float> vs(n);
        for (auto i = 0; i < n; ++i)
        {
            vs[i] = min_value + step * i;
        }
        return vs;
    }

    torch::Tensor make_grid(const std::vector<float>& xs, std::vector<float> ys)
    {
        std::sort(ys.rbegin(), ys.rend());
        auto z_points = torch::ones({400, 2}, torch::kFloat);
        int i = 0;
        for (const auto& y : ys)
        {
            for(const auto& x : xs)
            {
                auto dst = z_points[i];
                dst[0] = x;
                dst[1] = y;
                ++i;
            }
        }
        return z_points;
    }

    void generate_new_arts_on_grid(
        const torch::Device&    device, 
        VariationalAutoEncoder&            model)
    {
        //_/_/_/ Generate 2-dim coordinates 

        auto max_x = 4;
        auto min_x = -4;
        auto max_y = 4;
        auto min_y = -4;    

        const auto xs = generate_range(20, min_x, max_x);
        const auto ys = generate_range(20, min_y, max_y);

        const auto grid_points = make_grid(xs, ys);
        save_grid_points(grid_points, "grid/grid_points.txt");

        //_/_/_/ Decode

        auto gps = grid_points.to(device); 
        auto generated_images = model->get_decoder()->forward(gps); 
        save_images(generated_images, "grid/grid");
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

    auto test_dataset = torch::data::datasets::MNIST(
        DATA_DIR_PATH, 
        torch::data::datasets::MNIST::Mode::kTest
    ).map(torch::data::transforms::Stack<>());

    //_/_/_/ 

    generate_new_arts_on_grid(device, model);
    //make_distribution_in_latent_space(device, test_dataset, model, TEST_BATCH_SIZE);
 
    return 0;
}
