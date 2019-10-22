#include "dataset_reader.h"
#include <string>
#include <map>
#include <iostream>
#include <opencv2/highgui.hpp>

namespace
{
    std::map<int, std::string> INDEX2CATEGORY_NAME 
    {
        {0, "airplane"},
        {1, "automobile"},
        {2, "bird"},
        {3, "cat"},
        {4, "deer"},
        {5, "dog"},
        {6, "frog"},
        {7, "horse"},
        {8, "ship"},
        {9, "truck"},
    };

    constexpr int IMAGE_SIZE    {10000};
    constexpr int LABEL_SIZE    {1}; // bytes
    constexpr int ROWS          {32};
    constexpr int COLS          {32};
    constexpr int CHANNEL_SIZE  {ROWS * COLS};
    constexpr int CHANNEL_NUM   {3};
    constexpr int FILE_SIZE     {CHANNEL_NUM * CHANNEL_SIZE + LABEL_SIZE}; // bytes
}

DatasetReader::DatasetReader(std::ifstream& ifs)
    : ifs_{ifs} {}

std::pair<cv::Mat, int> DatasetReader::load_one_image(int index) const
{
    int position = index * FILE_SIZE;
    ifs_.seekg(position, std::ios_base::beg);

    char buffer[FILE_SIZE];
    ifs_.read(buffer, FILE_SIZE);

    const auto label = static_cast<int>(buffer[0]);    
    // std::cout << "label: " << INDEX2CATEGORY_NAME[label] << std::endl;

    char red_channel[CHANNEL_SIZE];
    char green_channel[CHANNEL_SIZE];
    char blue_channel[CHANNEL_SIZE];
   
    auto start = buffer + 1;
    auto end = start + CHANNEL_SIZE;
    std::copy(start, end, red_channel);
    
    start = end;
    end = start + CHANNEL_SIZE;
    std::copy(start, end, green_channel);
    
    start = end;
    end = start + CHANNEL_SIZE;
    std::copy(start, end, blue_channel);
    
    cv::Mat image(ROWS, COLS, CV_8UC3);
    for (auto row = 0; row < ROWS; ++row)
    {
        const auto p = COLS * row;
        for (auto col = 0; col < COLS; ++col)
        {
            const auto q = p + col;
            const auto r = static_cast<std::uint8_t>(red_channel[q]);
            const auto g = static_cast<std::uint8_t>(green_channel[q]);
            const auto b = static_cast<std::uint8_t>(blue_channel[q]);
            image.at<cv::Vec3b>(row, col) = cv::Vec3b(b, g, r);
        }
    }
    return {image, label};
}
#if(UNIT_TEST_DatasetReader)
#include <boost/test/unit_test.hpp>
#include <opencv2/core.hpp>

namespace
{
    const std::string PATH {"/home/ubuntu/data/cifar-10/cifar-10-batches-bin/data_batch_1.bin"};
    
    void test_DatasetReader(const std::string& path)
    {
        std::ifstream ifs {path, std::ios::binary};
        if (!ifs)
        {
            throw std::runtime_error("unvalid data path");
        }
        DatasetReader reader{ifs};
        cv::Mat image;
        int label;
        std::tie(image, label) = reader.load_one_image(0);
        BOOST_REQUIRE_EQUAL(6, label);
        cv::imwrite(std::string("image_0.jpg"), image);

        std::tie(image, label) = reader.load_one_image(1);
        BOOST_REQUIRE_EQUAL(9, label);
        cv::imwrite(std::string("image_1.jpg"), image);
    }
}

BOOST_AUTO_TEST_CASE(TEST_DatasetReader)
{
    std::cout << "DatasetReader\n";
    test_DatasetReader(PATH);
}

#endif // UNIT_TEST_DatasetReader
