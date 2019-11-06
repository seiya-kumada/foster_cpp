#include "custom_dataset.h"
#include "dataset_reader.h"

namespace
{
    constexpr int IMAGE_SIZE {10000};

    torch::Tensor convert_to_tensor(const cv::Mat& image)
    {
        cv::Mat fimage{};
        image.convertTo(fimage, CV_32FC3);
        auto tensor = torch::from_blob(fimage.data, {image.rows, image.cols, 3}, torch::kFloat);
        tensor = tensor.permute({2, 0, 1});
        return tensor.clone();
    }

    torch::Tensor convert_to_tensor(int label)
    {   
        auto tensor = torch::empty(1, torch::kInt64);
        *reinterpret_cast<int64_t*>(tensor.data_ptr()) = label;
        return tensor;
    }
}

CustomDataset::CustomDataset(std::vector<std::ifstream>& ifss) 
    : images_{}
    , labels_{}
{
    auto size = ifss.size();

    images_.reserve(size * IMAGE_SIZE);
    labels_.reserve(size * IMAGE_SIZE);
    cv::Mat image;
    int label;
   
    for (auto& ifs : ifss)
    {
        DatasetReader reader{ifs};
        for (auto i = 0; i < IMAGE_SIZE; ++i)
        {
            std::tie(image, label) = reader.load_one_image(i);
            images_.emplace_back(convert_to_tensor(image));
            labels_.emplace_back(convert_to_tensor(label));
        }
    }
};

#if(UNIT_TEST_CustomDataset)
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <torch/torch.h>

namespace fs = boost::filesystem;

namespace
{
    const fs::path DIR_PATH = fs::path("/home/ubuntu/data/cifar-10/cifar-10-batches-bin/");

    const std::vector<fs::path> PATHS 
    {
        DIR_PATH / "data_batch_1.bin",
        DIR_PATH / "data_batch_2.bin",
        DIR_PATH / "data_batch_3.bin",
        DIR_PATH / "data_batch_4.bin",
        DIR_PATH / "data_batch_5.bin",
    };


    void test_0()
    {
        std::vector<std::ifstream> ifss{};
        boost::copy(
            PATHS | boost::adaptors::transformed(
                [](const auto& p)
                {
                    return std::ifstream{p.string(), std::ios::binary};
                }
            ), 
            std::back_inserter(ifss)
        );

        CustomDataset ds{ifss};
        auto size = ds.size().value();
        BOOST_REQUIRE_EQUAL(ifss.size() * IMAGE_SIZE, size);

        auto e = ds.get(0);
        const torch::Tensor& data = e.data;
        const torch::Tensor& label = e.target;
        BOOST_REQUIRE_EQUAL(data.size(0), 3);
        BOOST_REQUIRE_EQUAL(data.size(1), 32);
        BOOST_REQUIRE_EQUAL(data.size(2), 32);
        BOOST_REQUIRE_EQUAL(label.size(0), 1);
        BOOST_REQUIRE_EQUAL(data.sizes(), (std::vector<std::int64_t>{3, 32, 32}));
        BOOST_REQUIRE_EQUAL(label.sizes(), (std::vector<std::int64_t>{1}));
               
        const std::vector<int> answers = {6, 9, 9, 4, 1};
        for (auto i = 0; i < 5; ++i)
        {
            const auto& e = ds.get(i);
            const auto& l = e.target;
            BOOST_REQUIRE_EQUAL(l.item<int64_t>(), answers[i]); 
        }
    }
   
    template<typename T>
    struct Type;

    void test_1()
    {
        torch::manual_seed(1);
        std::vector<std::ifstream> ifss{};
        boost::copy(
            PATHS | boost::adaptors::transformed(
                [](const auto& p)
                {
                    return std::ifstream{p.string(), std::ios::binary};
                }
            ), 
            std::back_inserter(ifss)
        );

        auto ds = CustomDataset{ifss}.
            map(torch::data::transforms::Normalize<>(0, 255.0)).
            map(torch::data::transforms::Stack<>())
            ;

        constexpr int BATCH_SIZE = 32;

        auto data_loader = torch::data::make_data_loader(
                std::move(ds),
                torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2).drop_last(true));
       
        const std::vector<int> answers = {4, 5, 3, 4, 6};
        int c = 0;
        for (auto& batch : *data_loader)
        {
            auto batch_size = batch.data.size(0);
            BOOST_REQUIRE_EQUAL(batch.data.sizes(), (std::vector<std::int64_t>{32, 3, 32, 32}));
            BOOST_REQUIRE_EQUAL(BATCH_SIZE, batch_size);
            for (auto i = 0; i < batch_size; ++i)
            {
                const auto& target = batch.target[i];
                const auto& data = batch.data[i];
                BOOST_REQUIRE_EQUAL(3, batch.data[i].size(0));
                BOOST_REQUIRE_EQUAL(32, batch.data[i].size(1));
                BOOST_REQUIRE_EQUAL(32, batch.data[i].size(2));
                auto v =  data[0][0][0].item<float>();
                BOOST_CHECK(0.0 <= v && v <= 1.0);
                //std::cout << target.sizes() << std::endl;
                BOOST_CHECK(answers[c] == target.item<int64_t>());
                c += 1;
                if (c == 5)
                {
                    break;
                }
            }
            if (c == 5)
            {
                break;
            }
        }
    }

    void test_2()
    {
        torch::Tensor a = torch::ones({2, 3});
        std::cout << "a: " << a << std::endl;
        auto b = a;
        a[0,0] = 100;
        std::cout << "a: " << a << std::endl;
        std::cout << "b: " << b << std::endl;
    }


    void test_3()
    {
        torch::manual_seed(1);
        std::vector<std::ifstream> ifss{};
        boost::copy(
            PATHS | boost::adaptors::transformed(
                [](const auto& p)
                {
                    return std::ifstream{p.string(), std::ios::binary};
                }
            ), 
            std::back_inserter(ifss)
        );

        auto ds = CustomDataset{ifss}.
            map(torch::data::transforms::Normalize<>(0, 255.0))
            ;
        constexpr int BATCH_SIZE = 32;
        Type<decltype(ds)> h;
        auto data_loader = torch::data::make_data_loader(
                std::move(ds),
                torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2).drop_last(true));
        int c = 0; 
        const std::vector<int> answers = {4, 5, 3, 4, 6};
        for (auto& batch : *data_loader)
        {
            auto batch_size = batch.size();
            BOOST_CHECK_EQUAL(batch_size, BATCH_SIZE);
            for (auto i = 0; i < batch_size; ++i)
            {
                const auto& target = batch[i].target;
                const auto& data = batch[i].data;
                BOOST_REQUIRE_EQUAL(3, data.size(0));
                BOOST_REQUIRE_EQUAL(32, data.size(1));
                BOOST_REQUIRE_EQUAL(32, data.size(2));
                auto v =  data[0][0][0].item<float>();
                BOOST_CHECK(0.0 <= v && v <= 1.0);
            //    //std::cout << target.sizes() << std::endl;
                BOOST_CHECK(answers[c] == target.item<int64_t>());
                c += 1;
                if (c == 5)
                {
                    break;
                }
            }
            if (c == 5)
            {
                break;
            }
        }
        
    }
}

BOOST_AUTO_TEST_CASE(TEST_CustomDataset)
{
    std::cout << "CustomDataset\n";
    test_0();
    test_1();
    test_3();
}

#endif // UNIT_TEST_CustomDataset
