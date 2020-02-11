#include <torch/torch.h>
#include "custom_dataset.h"
#include "variational_auto_encoder.h"
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include "make_attribute_vectors.h"
#include "add_attribute_vectors.h"

namespace fs = boost::filesystem;

namespace
{
    const std::string MODEL_DIR_PATH    {"/home/ubuntu/data/foster/ch03_05/"};
    const std::string OUTPUT_DIR_PATH   {"/home/ubuntu/data/foster/ch03_06/"};
    const std::string DATA_DIR_PATH     {"/home/ubuntu/data/celeba/img_align_celeba"};
    constexpr int64_t BATCH_SIZE        {500};
    const int         Z_DIM             {200};

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
        const int                     z_dim                     {Z_DIM};

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
            // std::cout << path << std::endl;
            cv::imwrite(path.string(), img);
        }
    }

    template<typename T>
    struct Type;

    void save_z_points(const torch::Tensor& z_points, int c, const std::string& subdir_name, const torch::Device& device)
    {
        auto dir_path = fs::path(OUTPUT_DIR_PATH) / subdir_name;
        auto batch_size = static_cast<int>(z_points.size(0));
        auto dim = static_cast<int>(z_points.size(1));
        auto points = z_points.to("cpu");
        for (auto i = 0; i < batch_size; ++i)
        {
            const auto& point = points[i];
            const auto data = point.data<float>();
            auto path = dir_path / (boost::format("%02d_%02d.bin") % c % i).str();
            std::ofstream ofs {path.string(), std::ios::out | std::ios::binary};
            std::cout << path << std::endl;
            std::cout << data[0] << std::endl;
            ofs.write(reinterpret_cast<char*>(data), sizeof(data[0]) * dim);     
            ofs.close();
        }
    }

    template<typename Dataset>
    void reconstruct_faces(VariationalAutoEncoder& model, const Dataset& dataset, const torch::Device& device)
    {
        const auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));

        torch::Tensor z_points {};
        torch::Tensor mu {};
        torch::Tensor log_var {};
        for (auto& batch : *loader)
        {
            auto data = batch.data.to(device);
            std::tie(z_points, mu, log_var) = model->predict(data);
            std::cout << "z_points: " << z_points.sizes() << std::endl; // (10,200)
            auto reconst_images = model->get_decoder()->forward(z_points);
            // std::cout << "reconst: " << reconst_images.sizes() << std::endl;
            save_images(data, "source_images");
            save_images(reconst_images, "reconst_images");
            break;
        }
    }

    template<typename Dataset>
    void make_latent_space_distibution(VariationalAutoEncoder& model, const Dataset& dataset, const torch::Device& device)
    { 
        const auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2));
        
        torch::Tensor z_points {};
        torch::Tensor mu {};
        torch::Tensor log_var {};
        auto steps = 20;
        auto c = 0;
        for (const auto& batch : *loader)
        {
            std::tie(z_points, mu, log_var) = model->predict(batch.data.to(device));
            save_z_points(z_points, c, "z_points", device);
            // std::cout << "z_points: " << z_points.sizes() << std::endl;
            c += 1;
            if (c == steps)
            {
                break;
            }
        }
    }

    void make_newly_generated_faces(VariationalAutoEncoder& model, const torch::Device& device)
    {
        auto size = std::vector<std::int64_t>{10, Z_DIM};
        const auto znew_0 = torch::empty(size).normal_().to(device);
        auto reconst_0 = model->get_decoder()->forward(znew_0).to("cpu");
        
        const auto znew_1 = torch::empty(size).normal_().to(device);
        auto reconst_1 = model->get_decoder()->forward(znew_1).to("cpu");

        //const auto znew_2 = torch::empty(size).normal_().to(device);
        //auto reconst_2 = model->get_decoder()->forward(znew_2).to("cpu");
        //
        auto vs = torch::cat({reconst_0, reconst_1}, 0);
        std::cout << vs.sizes() << std::endl;
        save_images(vs, "newly_generated_images");
    }

    inline std::string tolowercase(const std::string& src)
    {
        std::string dst {src};
        std::transform(std::begin(src), std::end(src), std::begin(dst), tolower); 
        return dst;
    }

    void make_attribute_vectors(
        VariationalAutoEncoder& model, 
        int                     batch_size, 
        const torch::Device&    device,
        bool                    is_verbose, 
        const fs::path&         sub_dir_path,
        int                     iterations,
        const std::string&      name)
    {
        auto vec = make_attribute_vectors(
            model, batch_size, device, iterations, Z_DIM, name, is_verbose).to(torch::kCPU);
        auto path = sub_dir_path / (boost::format("%1%_vec.pt") % tolowercase(name)).str();
        torch::save(vec, path.string());
        std::cout << boost::format("%1% done\n") % name;
    }

    void make_attribute_vectors(
        VariationalAutoEncoder& model, 
        int                     batch_size, 
        const torch::Device&    device,
        bool                    is_verbose)
    {
        auto sub_dir_path = fs::path(OUTPUT_DIR_PATH) / "attribute_vecs";
        int iterations = 10000;
    
        make_attribute_vectors(model, batch_size, device, is_verbose, sub_dir_path, iterations, "Attractive");
        make_attribute_vectors(model, batch_size, device, is_verbose, sub_dir_path, iterations, "Mouth_Slightly_Open");
        make_attribute_vectors(model, batch_size, device, is_verbose, sub_dir_path, iterations, "Smiling");
        make_attribute_vectors(model, batch_size, device, is_verbose, sub_dir_path, iterations, "Wearing_Lipstick");
        make_attribute_vectors(model, batch_size, device, is_verbose, sub_dir_path, iterations, "High_Cheekbones");
        make_attribute_vectors(model, batch_size, device, is_verbose, sub_dir_path, iterations, "Male");
        make_attribute_vectors(model, batch_size, device, is_verbose, sub_dir_path, iterations, "Eyeglasses");
        make_attribute_vectors(model, batch_size, device, is_verbose, sub_dir_path, iterations, "Blond_Hair");
    }

    template<typename Dataset>
    void add_attribute_vectors(
        VariationalAutoEncoder& model, 
        const Dataset&          dataset,
        int                     batch_size, 
        const torch::Device&    device, 
        const fs::path&         sub_dir_path,
        const std::string&      name)
    {
        auto path = sub_dir_path / (boost::format("%1%_vec.pt") % tolowercase(name)).str();
        torch::Tensor vec {};
        torch::load(vec, path.string());
        auto images = add_attribute_vectors(model, dataset, batch_size, device, vec);
    }

    template<typename Dataset>
    void add_attribute_vectors(
        VariationalAutoEncoder& model, 
        const Dataset&          dataset,
        int                     batch_size, 
        const torch::Device&    device)
    {
        auto sub_dir_path = fs::path(OUTPUT_DIR_PATH) / "attribute_vecs";

        add_attribute_vectors(model, dataset, batch_size, device, sub_dir_path, "Attractive");
        //add_attribute_vectors(model, batch_size, device, sub_dir_path, "Mouth_Slightly_Open");
        //add_attribute_vectors(model, batch_size, device, sub_dir_path, "Smiling");
        //add_attribute_vectors(model, batch_size, device, sub_dir_path, "Wearing_Lipstick");
        //add_attribute_vectors(model, batch_size, device, sub_dir_path, "High_Cheekbones");
        //add_attribute_vectors(model, batch_size, device, sub_dir_path, "Male");
        //add_attribute_vectors(model, batch_size, device, sub_dir_path, "Eyeglasses");
        //add_attribute_vectors(model, batch_size, device, sub_dir_path, "Blond_Hair");
    }

}

#if(UNIT_TEST)
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(TEST_main)
{
    std::cout << "main\n";
}

#else // UNIT_TEST

int main(int argc, const char* argv[])
{
    if (torch::cuda::cudnn_is_available())
    {
        std::cout << "Has CuDNN\n";
    }
    
    torch::NoGradGuard no_grad_guard {};

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

    //_/_/_/ Reconstruct faces
   
    //reconstruct_faces(model, dataset, device);

    //_/_/_/ Make latent space distribution

    //make_latent_space_distibution(model, dataset, device);
    
    //_/_/_/ Make newly generated faces
    
    //make_newly_generated_faces(model, device);
    
    //_/_/_/ Make attribute vectors

    //make_attribute_vectors(model, BATCH_SIZE, device, true);

    //_/_/_/ Add attribute vectors
    add_attribute_vectors(model, dataset, 10, device);
 
    return 0;
}


#endif  // UNIT_TEST
