#include "make_attribute_vectors.h"
#include "custom_dataset_from_csv.h"
#include "variational_auto_encoder.h"
#include "csv.h"
#include <cpplinq.hpp>
#include <boost/format.hpp>

namespace
{
    // test ok
    std::vector<torch::Tensor> extract_vectors(const torch::Tensor& target, const torch::Tensor& z_points, int flag)
    {
        int size = target.size(0);
        return cpplinq::range(0, size) 
            >> cpplinq::where([&target, flag](const auto& v){ return target[v].template item<std::int64_t>() == flag; })
            >> cpplinq::select([&z_points](const auto& v){ return z_points[v]; })
            >> cpplinq::to_vector();
    }

    // test ok
    class Group
    {
    private:
        int             current_n_;
        torch::Tensor   current_sum_;
        torch::Tensor   current_mean_;
        torch::Tensor   zeros_;
        torch::Tensor   new_mean_;
        float           movement_;

    public:
        Group(int z_dim, torch::Device device)
            : current_n_{0}
            , current_sum_{torch::zeros({z_dim}, torch::kFloat).to(device)}
            , current_mean_{torch::zeros({z_dim}, torch::kFloat).to(device)}
            , zeros_{torch::zeros({z_dim}, torch::kFloat).to(device)}
            , new_mean_{torch::zeros({z_dim}, torch::kFloat).to(device)}
            , movement_{0.0f}
        {}

        void update(const std::vector<torch::Tensor>& z)
        {
            current_sum_ += std::accumulate(std::begin(z), std::end(z), zeros_);
            current_n_ += z.size();
            new_mean_ = current_sum_ / current_n_;
            movement_ = torch::norm(new_mean_ - current_mean_).item<float>();
            current_mean_ = new_mean_.clone();
        }

        int get_current_n() const { return current_n_; }
        const torch::Tensor get_new_mean() const { return new_mean_; }
        float get_movement() const { return movement_; }
    };
}


torch::Tensor make_attribute_vectors(
    VariationalAutoEncoder& model, 
    int                     batch_size, 
    const torch::Device&    device, 
    int                     iterations,
    int                     z_dim,
    const std::string&      label,
    bool                    is_verbose)
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
    
    Group group_pos {z_dim, device};
    Group group_neg {z_dim, device};

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
            group_pos.update(z_pos);
        }

        if (!z_neg.empty())
        {
            group_neg.update(z_neg);
        }

        current_vector = group_pos.get_new_mean() - group_neg.get_new_mean();
        auto new_dist = torch::norm(current_vector).item<float>();
        auto dist_change = new_dist - current_dist;

        if (is_verbose)
        {
            std::cout << boost::format("%1% :%2% :%3% :%4% :%5%\n") 
                % group_pos.get_current_n() 
                % group_pos.get_movement()
                % group_neg.get_movement()
                % new_dist
                % dist_change;
        } 
        current_dist = new_dist;

        auto e = group_pos.get_movement() + group_neg.get_movement();
        if (e < 0.08)
        {
            current_vector /= current_dist;
            std::cout << "Found the " << label << " vector\n";
            break;
        }
        
        if (group_pos.get_current_n() == iterations)
        {
            break;
        }
    }
    return current_vector;
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
       
        int batch_size {50};
        int iterations {10000};
        int z_dim {200};
        auto v = make_attribute_vectors(model, batch_size, device, iterations, z_dim, "Attractive", false);
        auto d = torch::norm(v);
        BOOST_CHECK_CLOSE(d.item<float>(), 1, 1.0e-4);
        //std::cout << v << std::endl;
    }
    
    void test_extract_vectors()
    {
        int batch_size = 3;
        int z_dim = 2;
        torch::Tensor z_points = torch::ones({batch_size, z_dim});
        z_points[0][0] = 2;
        z_points[0][1] = 2;
        torch::Tensor target = torch::ones({batch_size, 1}, torch::kInt64);
        target[0] = -1;

        auto z_pos = extract_vectors(target, z_points, 1);
        BOOST_CHECK_EQUAL(2, z_pos.size());
        BOOST_CHECK_EQUAL(z_pos[0][0].item<float>(), 1);
        BOOST_CHECK_EQUAL(z_pos[0][1].item<float>(), 1);
        BOOST_CHECK_EQUAL(z_pos[1][0].item<float>(), 1);
        BOOST_CHECK_EQUAL(z_pos[1][1].item<float>(), 1);
        
        z_pos = extract_vectors(target, z_points, -1);
        BOOST_CHECK_EQUAL(1, z_pos.size());
        BOOST_CHECK_EQUAL(z_pos[0][0].item<float>(), 2);
        BOOST_CHECK_EQUAL(z_pos[0][1].item<float>(), 2);
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

    void test_zero()
    {
        torch::Tensor zeros {torch::zeros({3}, torch::kFloat)};
        torch::Tensor a {zeros.clone()};
        torch::Tensor b {zeros.clone()};
        a += torch::ones({3}); 
        b += torch::ones({3});
        BOOST_CHECK_EQUAL(a[0].item<float>(), 1);
        BOOST_CHECK_EQUAL(a[1].item<float>(), 1);
        BOOST_CHECK_EQUAL(a[2].item<float>(), 1);
        BOOST_CHECK_EQUAL(b[0].item<float>(), 1);
        BOOST_CHECK_EQUAL(b[1].item<float>(), 1);
        BOOST_CHECK_EQUAL(b[2].item<float>(), 1);
    }

    void test_group()
    {
        int z_dim {2};
        std::vector<torch::Tensor> pos_zs {};
        std::vector<torch::Tensor> neg_zs {};
        int s = 1000;
        pos_zs.reserve(s);
        neg_zs.reserve(s);
        for (auto i = 0; i < s; ++i)
        {
            auto pz = torch::empty({z_dim}, torch::kFloat).normal_() + torch::ones({z_dim}, torch::kFloat);
            pos_zs.emplace_back(std::move(pz));
            
            auto nz = torch::empty({z_dim}, torch::kFloat).normal_() + 5 * torch::ones({z_dim}, torch::kFloat);
            neg_zs.emplace_back(std::move(nz));
        }
        
        torch::Device device {torch::kCPU};
        Group pos_group {z_dim, device};
        Group neg_group {z_dim, device};

        pos_group.update(pos_zs);
        neg_group.update(neg_zs);

        auto pos_new_mean = pos_group.get_new_mean();
        auto neg_new_mean = neg_group.get_new_mean();
        
        for (auto i = 0; i < z_dim; ++i)
        {
            BOOST_CHECK(std::abs(pos_new_mean[i].item<float>() - 1.0) < 0.1);
            BOOST_CHECK(std::abs(neg_new_mean[i].item<float>() - 5.0) < 0.1);
        }

        BOOST_CHECK(std::abs(std::sqrt(2) - pos_group.get_movement()) < 0.1);
        BOOST_CHECK(std::abs(std::sqrt(2) * 5 - neg_group.get_movement()) < 0.1);

        for (auto i = 0; i < 10; ++i)
        {
            pos_zs.clear();
            neg_zs.clear();
            pos_zs.reserve(s);
            neg_zs.reserve(s);
            for (auto i = 0; i < s; ++i)
            {
                auto pz = torch::empty({z_dim}, torch::kFloat).normal_() + torch::ones({z_dim}, torch::kFloat);
                pos_zs.emplace_back(std::move(pz));
                
                auto nz = torch::empty({z_dim}, torch::kFloat).normal_() + 5 * torch::ones({z_dim}, torch::kFloat);
                neg_zs.emplace_back(std::move(nz));
            }
            BOOST_CHECK(pos_zs.size() == s);
            BOOST_CHECK(neg_zs.size() == s);

            pos_group.update(pos_zs);
            neg_group.update(neg_zs);

            //std::cout << "pos: " << pos_group.get_movement() << std::endl;
            //std::cout << "neg: " << neg_group.get_movement() << std::endl;
        }
        pos_new_mean = pos_group.get_new_mean();
        neg_new_mean = neg_group.get_new_mean();
        for (auto i = 0; i < z_dim; ++i)
        {
            BOOST_CHECK(std::abs(pos_new_mean[i].item<float>() - 1.0) < 0.1);
            BOOST_CHECK(std::abs(neg_new_mean[i].item<float>() - 5.0) < 0.1);
        }
    }
}

BOOST_AUTO_TEST_CASE(TEST_make_attribute_vectors)
{
    std::cout << "make_attribute_vectors\n";
    test_make_attribute_vectors();
    test_extract_vectors();
    test_sum();
    test_dot();
    test_zero();
    test_group();
}
#endif // UNIT_TEST_MAKE_ATTRIBUTE_VECTORS


