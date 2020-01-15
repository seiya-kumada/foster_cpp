#include "make_attribute_vectors.h"
#include "custom_dataset_from_csv.h"
#include "variational_auto_encoder.h"
#include "csv.h"
#include <cpplinq.hpp>
#include <boost/format.hpp>

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

    std::pair<torch::Tensor, float> update_position(
        torch::Tensor&                      current_sum,
        const std::vector<torch::Tensor>&   z,
        const torch::Tensor&                zeros,
        const torch::Tensor&                current_mean,
        int&                                current_n
    ) {
        current_sum += std::accumulate(std::begin(z), std::end(z), zeros);
        current_n += z.size();
        auto new_mean = current_sum / current_n;
        auto movement = torch::norm(new_mean - current_mean);
        return std::make_pair(new_mean, movement.item<float>());
    } 
}

void make_attribute_vectors(
    VariationalAutoEncoder& model, 
    int                     batch_size, 
    const torch::Device&    device, 
    int                     iterations,
    int                     z_dim,
    const std::string&      label)
{
    const std::string csv_path = "/home/ubuntu/data/celeba/list_attr_celeba.csv";
    const std::string dir_path = "/home/ubuntu/data/celeba/img_align_celeba/";
    io::CSVReader<2> csv {csv_path};
    std::vector<int> input_size {128, 128};
    auto dataset = CustomDatasetFromCSV {dir_path, csv, input_size, label}
        .map(torch::data::transforms::Normalize<>(0, 255.0))
        .map(torch::data::transforms::Stack<>());
    const auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

    torch::Tensor z_points {};
    torch::Tensor mu {};
    torch::Tensor log_var {};

    torch::Tensor zeros {torch::zeros({z_dim}, torch::kFloat)};
    
    int current_n_pos {0};
    torch::Tensor current_sum_pos {zeros};
    torch::Tensor current_mean_pos {zeros};
    torch::Tensor new_mean_pos {torch::empty({z_dim}, torch::kFloat)};
    float movement_pos {0.0f};
    
    int current_n_neg {0};
    torch::Tensor current_sum_neg {zeros};
    torch::Tensor current_mean_neg {zeros};
    torch::Tensor new_mean_neg {torch::empty({z_dim}, torch::kFloat)};
    float movement_neg {0.0f};

    torch::Tensor current_vector {torch::empty({z_dim}, torch::kFloat)};
    float current_dist = 0.0f;

    for (auto& batch : *loader)
    {
        std::tie(z_points, mu, log_var) = model->predict(batch.data.to(device));
        const auto& target = batch.target;
        auto z_pos = extract_vectors(target, z_points, 1);
        auto z_neg = extract_vectors(target, z_points, -1);
        
        if (!z_pos.empty())
        {
            std::tie(new_mean_pos, movement_pos) = update_position(
                current_sum_pos,
                z_pos,
                zeros,
                current_mean_pos,
                current_n_pos);
        }

        if (!z_neg.empty())
        {
             std::tie(new_mean_neg, movement_neg) = update_position(
                current_sum_neg,
                z_neg,
                zeros,
                current_mean_neg,
                current_n_neg);
        }

        current_vector = new_mean_pos - new_mean_neg;
        auto new_dist = torch::norm(current_vector);
        auto dist_change = new_dist - current_dist;

        std::cout << boost::format("%1% :%2% :%3% :%4% :%5%") 
            % current_n_pos 
            % movement_pos
            % movement_neg
            % new_dist
            % dist_change;
        
        current_mean_pos = new_mean_pos.clone();
        current_mean_neg = new_mean_neg.clone();
        current_dist = new_dist.item<float>();

        auto e = movement_pos + movement_neg;
        if (e < 0.08)
        {
            current_vector /= current_dist;
            std::cout << "Found the " << label << " vector\n";
            break;
        }

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
       
        int batch_size {500};
        int iterations {1};
        int z_dim {200};
        make_attribute_vectors(model, batch_size, device, iterations, z_dim, "Attractive");
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

    void test_sum()
    {
        torch::Tensor a {torch::ones({3}, torch::kFloat)};
        torch::Tensor b {torch::ones({3}, torch::kFloat)};
        torch::Tensor c = a + b;
        BOOST_CHECK_EQUAL(2, c[0].item<float>());
        BOOST_CHECK_EQUAL(2, c[1].item<float>());
        BOOST_CHECK_EQUAL(2, c[2].item<float>());
    }

    void test_dot()
    {
        torch::Tensor a {torch::ones({3}, torch::kFloat)};
        torch::Tensor b {torch::ones({3}, torch::kFloat)};
        torch::Tensor c = a + b;
        torch::Tensor d = torch::sqrt(c * a);
        BOOST_CHECK_CLOSE(std::sqrt(2.0), d[0].item<float>(), 1.0e-5);
        BOOST_CHECK_CLOSE(std::sqrt(2.0), d[1].item<float>(), 1.0e-5);
        BOOST_CHECK_CLOSE(std::sqrt(2.0), d[2].item<float>(), 1.0e-5);
    }

    void test_update_position()
    {
        int z_dim = 3;
        auto current_sum {torch::zeros({z_dim}, torch::kFloat)};
        auto z = std::vector<torch::Tensor>{
            2 * torch::ones({z_dim}, torch::kFloat), 
            3 * torch::ones({z_dim}, torch::kFloat), 
        };
        auto zeros {torch::zeros({z_dim}, torch::kFloat)};
        auto current_mean {zeros};
        int current_n = 0;
       
        auto new_mean {zeros};
        float movement {0.0f};
        std::tie(new_mean, movement) = update_position(current_sum, z, zeros, current_mean, current_n);
        BOOST_CHECK_CLOSE(2.5, new_mean[0].item<float>(), 1.0e-5);
        BOOST_CHECK_CLOSE(2.5, new_mean[1].item<float>(), 1.0e-5);
        BOOST_CHECK_CLOSE(2.5, new_mean[2].item<float>(), 1.0e-5);
        BOOST_CHECK_CLOSE(std::sqrt(3 * std::pow(2.5, 2)), movement, 1.0e-5);
    }
}

BOOST_AUTO_TEST_CASE(TEST_make_attribute_vectors)
{
    std::cout << "make_attribute_vectors\n";
    //test_make_attribute_vectors();
    test_cpplinq();
    test_sum();
    test_dot();
    test_update_position();
}
#endif // UNIT_TEST_MAKE_ATTRIBUTE_VECTORS


