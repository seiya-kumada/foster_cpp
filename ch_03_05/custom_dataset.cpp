#include "custom_dataset.h"
#include <opencv2/highgui.hpp>

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

CustomDataset::CustomDataset(const std::string& dir_path)
{
    load_paths(dir_path, paths_); 
}

torch::data::Example<> CustomDataset::get(std::size_t index)
{
    auto image = cv::imread(paths_[index].string());
    return {convert_to_tensor(image), torch::empty({})};
};

#if(UNIT_TEST_CustomDataset)
#include <boost/test/unit_test.hpp>

namespace
{
    void test_0()
    {
        CustomDataset dataset {"/home/ubuntu/data/celeba/img_align_celeba"};
        const auto& paths = dataset.get_paths();
        BOOST_CHECK_EQUAL(202599, paths.size());
        if (dataset.size())
        {
            auto size = paths.size();
            BOOST_CHECK_EQUAL(202599, size);
        }
        auto e = dataset.get(0);
        auto data = e.data;
        BOOST_CHECK_EQUAL(data.sizes(), (std::vector<std::int64_t>{3, 218, 178}));
        BOOST_CHECK_EQUAL(189, data[0][0][0].item<float>());
    }
}

BOOST_AUTO_TEST_CASE(TEST_CustomDataset)
{
    std::cout << "CustomDataset\n";
    test_0();
}

#endif // UNIT_TEST_CustomDataset
