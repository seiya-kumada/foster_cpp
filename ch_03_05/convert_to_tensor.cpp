#include "convert_to_tensor.h"
#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>
#include <iostream>

namespace fs = boost::filesystem;

namespace
{
    torch::Tensor convert_image_to_tensor(const cv::Mat& image)
    {
        cv::Mat fimage{};
        image.convertTo(fimage, CV_32FC3);
        auto tensor = torch::from_blob(fimage.data, {image.rows, image.cols, 3}, torch::kFloat);
        tensor = tensor.permute({2, 0, 1});
        return tensor.clone();
    }
}

void convert_to_tensor(const std::string& src_dir_path, const std::string& dst_dir_path)
{
    auto dst_dpath = fs::path {dst_dir_path};
    auto input_size = cv::Size {128, 128};
    for (const auto& p: fs::directory_iterator(src_dir_path))
    {
        const auto& path = p.path();
        if (path.extension() == ".jpg")
        {
            auto image = cv::imread(path.string());
            cv::resize(image, image, input_size, cv::INTER_LINEAR);
            auto tensor = convert_image_to_tensor(image);
            auto name = path.filename();
            auto dst_path = dst_dpath / name;
            dst_path.replace_extension(".pt");
            torch::save(tensor, dst_path.string());
        }
    }
}


#if(UNIT_TEST_convert_to_tensor)
#include <boost/test/unit_test.hpp>

namespace
{
    void test0()
    {
        //convert_to_tensor(
        //    "/home/ubuntu/data/celeba/img_align_celeba", 
        //    "/home/ubuntu/data/celeba/img_align_celeba_tensor");
        //torch::Tensor tensor {};
        //torch::load(tensor, "/home/ubuntu/data/celeba/img_align_celeba_tensor/048255.pt");
        //std::cout << tensor.sizes() << std::endl;
    }
}

BOOST_AUTO_TEST_CASE(TEST_convert_to_tensor)
{
    std::cout << "convert_to_tensor\n";
    test0();
}

#endif // UNIT_TEST_convert_to_tensor
