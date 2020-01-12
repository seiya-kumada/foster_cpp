#include "custom_dataset_from_csv.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = boost::filesystem;

namespace
{
    std::vector<FilenameInfo> read_names_from_csv(io::CSVReader<2>&  csv, const std::string& label)
    {
        csv.read_header(io::ignore_extra_column, "image_id", label);
        std::string image_id {};
        std::string flag {};
        std::vector<FilenameInfo> infos {};
        while (csv.read_row(image_id, flag))
        {
            infos.emplace_back(image_id, flag);
        }
        return infos;
    }

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

CustomDatasetFromCSV::CustomDatasetFromCSV(
    const std::string& dir_path, io::CSVReader<2>& csv, const std::vector<int>& input_size, const std::string& label)
    : dir_path_{fs::path{dir_path}}
    , filename_infos_{read_names_from_csv(csv, label)}
    , input_size_{input_size[0], input_size[1]}
{
}

torch::data::Example<> CustomDatasetFromCSV::get(std::size_t index)
{
    auto path = dir_path_ / filename_infos_[index].image_id_;
    auto image = cv::imread(path.string());
    int flag = std::stoi(filename_infos_[index].flag_);
    cv::resize(image, image, input_size_, cv::INTER_LINEAR);
    return {convert_to_tensor(image), convert_to_tensor(flag)};
};

#if(UNIT_TEST_CustomDatasetFromCSV)
#include <boost/test/unit_test.hpp>

namespace
{
    void test_read_names_from_csv()
    {
        const std::string csv_path = "/home/ubuntu/data/celeba/list_attr_celeba.csv";
        io::CSVReader<2> csv {csv_path};
        auto infos = read_names_from_csv(csv, "Attractive");
        BOOST_CHECK_EQUAL(202599, infos.size());
    }

    void test_get()
    {
        const std::string csv_path = "/home/ubuntu/data/celeba/list_attr_celeba.csv";
        const std::string dir_path = "/home/ubuntu/data/celeba/img_align_celeba/";
        io::CSVReader<2> csv {csv_path};
        std::vector<int> input_size {128, 128};
        CustomDatasetFromCSV dataset {dir_path, csv, input_size, "Attractive"};
        const auto vs = dataset.get(0);
        const auto& x = vs.data;
        const auto& y = vs.target;
        BOOST_REQUIRE_EQUAL((std::vector<std::int64_t>{3, 128, 128}), x.sizes());
        BOOST_REQUIRE_EQUAL((std::vector<std::int64_t>{1}), y.sizes());
    }
}

BOOST_AUTO_TEST_CASE(TEST_CustomDatasetFromCSV)
{
    std::cout << "CustomDatasetFromCSV\n";
    test_read_names_from_csv();
    test_get();
}

#endif // UNIT_TEST_CustomDatasetFromCSV
