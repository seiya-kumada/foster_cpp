#include "custom_dataset.h"
#include <npy.hpp>

namespace fs = boost::filesystem;

namespace
{
}

CustomDataset::CustomDataset(const std::string& path)
{
    bool fortran_order {};
    npy::LoadArrayFromNumpy(path, shape_, fortran_order, data_);
}

torch::data::Example<> CustomDataset::get(std::size_t j)
{
    const auto offset = j * shape_[1];
    const auto begin = std::begin(data_) + offset;
    const auto end = begin + shape_[1];
    std::vector<uint8_t> vs {};
    std::copy(begin, end, std::back_inserter(vs));
    return {torch::from_blob(vs.data(), {1, 28, 28}, torch::kUInt8).to(torch::kFloat), torch::empty({})};
};

#if(UNIT_TEST_CustomDataset)
#include <boost/test/unit_test.hpp>

namespace
{
    const std::string PATH {"/home/ubuntu/projects/GDL_code/data/camel/full_numpy_bitmap_camel.npy"};

    void test_npy()
    {
        std::vector<std::uint64_t> shape {};
        bool fortran_order {};
        std::vector<std::uint8_t> data {};
        npy::LoadArrayFromNumpy(PATH, shape, fortran_order, data);

        auto rows = shape[0];
        auto cols = shape[1];
        BOOST_CHECK_EQUAL(rows, 121399); 
        BOOST_CHECK_EQUAL(cols, 784); 
        BOOST_CHECK_EQUAL(data.size(), rows * cols); 
        BOOST_CHECK_EQUAL(fortran_order, 0); 
    }

    void test_get()
    {
        CustomDataset dataset {PATH};
        auto size = dataset.size();
        BOOST_CHECK_EQUAL(121399, size.value());
        torch::Tensor data {};
        torch::Tensor target {};
        auto v = dataset.get(0);
        BOOST_CHECK_EQUAL(v.data.sizes(), (std::vector<int64_t>{1, 28, 28}));
        BOOST_CHECK_EQUAL(v.data[0][7][4].item<float>(), 38);
        BOOST_CHECK_EQUAL(v.data[0][14][8].item<float>(), 141);

        v = dataset.get(size.value() - 1);
        BOOST_CHECK_EQUAL(v.data[0][10][20].item<float>(), 248);
        BOOST_CHECK_EQUAL(v.data[0][17][24].item<float>(), 82);
    }

}

BOOST_AUTO_TEST_CASE(TEST_CustomDataset)
{
    std::cout << "CustomDataset\n";
    test_npy();
    test_get();
}

#endif // UNIT_TEST_CustomDataset
