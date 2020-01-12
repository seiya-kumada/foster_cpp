#include "make_attribute_vectors.h"
#include "custom_dataset_from_csv.h"
#include "variational_auto_encoder.h"
#include "csv.h"
#include <cpplinq.hpp>

namespace
{
    std::vector<torch::Tensor> extract_vectors(const torch::Tensor& target, const torch::Tensor& z_points, int flag)
    {
        int size = target.size(0);
        return cpplinq::range(0, size) 
            >> cpplinq::where([&target, flag](const auto& v){ return target[v].template item<std::int64_t>() == flag; })
            >> cpplinq::select([&z_points](const auto& v){ return z_points[v]; })
            >> cpplinq::to_vector();
    }
}

void make_attribute_vectors(
    VariationalAutoEncoder& model, 
    int                     batch_size, 
    const torch::Device&    device, 
    int                     iterations)
{
    const std::string csv_path = "/home/ubuntu/data/celeba/list_attr_celeba.csv";
    const std::string dir_path = "/home/ubuntu/data/celeba/img_align_celeba/";
    io::CSVReader<2> csv {csv_path};
    std::vector<int> input_size {128, 128};
    auto dataset = CustomDatasetFromCSV {dir_path, csv, input_size, "Attractive"}
        .map(torch::data::transforms::Normalize<>(0, 255.0))
        .map(torch::data::transforms::Stack<>());
    const auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

    torch::Tensor z_points {};
    torch::Tensor mu {};
    torch::Tensor log_var {};
    int current_n_pos = 0;
    for (auto& batch : *loader)
    {
        std::tie(z_points, mu, log_var) = model->predict(batch.data.to(device));
        const auto& target = batch.target;
        auto z_pos = extract_vectors(target, z_points, 1);
        auto z_neg = extract_vectors(target, z_points, -1);
 
        current_n_pos += 1;

        if (current_n_pos == iterations)
        {
            break;
        }
    }
}

#if(UNIT_TEST_make_attribute_vectors)
#include <boost/test/unit_test.hpp>
#include <fstream>

namespace fs = boost::filesystem;


namespace
{
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

    const std::string MODEL_DIR_PATH    {"/home/ubuntu/data/foster/ch03_05/"};
    void test_make_attribute_vectors()
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

        //_/_/_/ Define and load a model

        auto model = make_model(device);
        model->to(device);

        const auto model_path = fs::path(MODEL_DIR_PATH) / "model.pt";
        torch::load(model, model_path.string());
        model->eval();

        //_/_/_/ Make attribute vectors
       
        int batch_size = 500;
        int iterations = 1;
        make_attribute_vectors(model, batch_size, device, iterations);
    }

    void test_cpplinq()
    {
        int batch_size = 3;
        int z_dim = 2;
        torch::Tensor z_points = torch::ones({batch_size, z_dim});
        torch::Tensor target = torch::ones({batch_size, 1}, torch::kInt64);
        target[0] = -1;

        auto z_pos = extract_vectors(target, z_points, 1);
        BOOST_CHECK_EQUAL(2, z_pos.size());
        
        z_pos = extract_vectors(target, z_points, -1);
        BOOST_CHECK_EQUAL(1, z_pos.size());
    }
}

BOOST_AUTO_TEST_CASE(TEST_make_attribute_vectors)
{
    std::cout << "make_attribute_vectors\n";
    test_make_attribute_vectors();
    test_cpplinq();
}
#endif // UNIT_TEST_MAKE_ATTRIBUTE_VECTORS


