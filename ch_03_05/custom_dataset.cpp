#include "custom_dataset.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = boost::filesystem;

namespace
{
    void load_paths(const std::string& dir_path, std::vector<fs::path>& paths)
    {
        for (auto& p: fs::directory_iterator(dir_path))
        {
            if (p.path().extension() == ".jpg")
            {
                paths.emplace_back(p.path()); 
            }
        }
    }

    torch::Tensor convert_to_tensor(const cv::Mat& image)
    {
        cv::Mat fimage{};
        image.convertTo(fimage, CV_32FC3);
        auto tensor = torch::from_blob(fimage.data, {image.rows, image.cols, 3}, torch::kFloat);
        tensor = tensor.permute({2, 0, 1});
        return tensor.clone();
    }
}

CustomDataset::CustomDataset(const std::string& dir_path, const std::vector<int>& input_size)
    : input_size_{input_size[0], input_size[1]}
{
    load_paths(dir_path, paths_); 
}

torch::data::Example<> CustomDataset::get(std::size_t index)
{
    auto image = cv::imread(paths_[index].string());
    cv::resize(image, image, input_size_, cv::INTER_LINEAR);
    return {convert_to_tensor(image), torch::empty({})};

    //torch::Tensor tensor {};
    //torch::load(tensor, paths_[index].string());
    //return {tensor, torch::empty({})};
};

#if(UNIT_TEST_CustomDataset)
#include <boost/test/unit_test.hpp>

namespace
{
    void test_0()
    {
        std::vector<int> input_size {128, 128};
        CustomDataset dataset {"/home/ubuntu/data/celeba/img_align_celeba", input_size};
        const auto& paths = dataset.get_paths();
        BOOST_CHECK_EQUAL(202599, paths.size());
        if (dataset.size())
        {
            auto size = paths.size();
            BOOST_CHECK_EQUAL(202599, size);
        }
        auto e = dataset.get(0);
        auto data = e.data;
        BOOST_CHECK_EQUAL(data.sizes(), (std::vector<std::int64_t>{3, 128, 128}));
        BOOST_CHECK_EQUAL(192, data[0][0][0].item<float>());
    }

    void test_1()
    {
        std::vector<int> input_size {128, 128};
        auto dataset = CustomDataset{"/home/ubuntu/data/celeba/img_align_celeba", input_size}
            .map(torch::data::transforms::Normalize<>(0, 255.0))
            .map(torch::data::transforms::Stack<>());

        auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(2).workers(2));
    
        for (auto& batch : *loader)
        {
            auto data = batch.data;
            auto a = data[0][0][0][0].item<float>();
            auto b = data[0][1][0][0].item<float>();
            auto c = data[0][2][0][0].item<float>();
            BOOST_CHECK(0 <= a && a <= 1);
            BOOST_CHECK(0 <= b && b <= 1);
            BOOST_CHECK(0 <= c && c <= 1);
            break;
        }
    }
}

BOOST_AUTO_TEST_CASE(TEST_CustomDataset)
{
    std::cout << "CustomDataset\n";
    test_0();
    test_1();
}

#endif // UNIT_TEST_CustomDataset
