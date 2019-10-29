#include "architecture.h"

namespace
{
    constexpr int CONV1_IN_CHANNELS     {3};
    constexpr int CONV1_OUT_CHANNELS    {32};
    constexpr int CONV1_KERNEL          {3};
    constexpr int CONV2_OUT_CHANNELS    {32};
    constexpr int CONV2_KERNEL          {3};
    constexpr int CONV3_OUT_CHANNELS    {64};
    constexpr int CONV3_KERNEL          {3};
    constexpr int CONV4_OUT_CHANNELS    {64};
    constexpr int CONV4_KERNEL          {3};
    constexpr int LAST_SIZE             {8};
    constexpr int N_HIDDEN1             {128};
    constexpr int N_CLASSES             {10};
    constexpr double DROPOUT_RATE             {0.5};
}

ArchitectureImpl::ArchitectureImpl()
    : conv1_{torch::nn::Conv2dOptions(CONV1_IN_CHANNELS,  CONV1_OUT_CHANNELS, CONV1_KERNEL).padding(1).stride(1)}
    , batch_norm1_{CONV1_OUT_CHANNELS}
    , conv2_{torch::nn::Conv2dOptions(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, CONV2_KERNEL).padding(1).stride(2)}
    , batch_norm2_{CONV2_OUT_CHANNELS}
    , conv3_{torch::nn::Conv2dOptions(CONV2_OUT_CHANNELS, CONV3_OUT_CHANNELS, CONV3_KERNEL).padding(1).stride(1)}
    , batch_norm3_{CONV3_OUT_CHANNELS}
    , conv4_{torch::nn::Conv2dOptions(CONV3_OUT_CHANNELS, CONV4_OUT_CHANNELS, CONV4_KERNEL).padding(1).stride(2)}
    , batch_norm4_{CONV4_OUT_CHANNELS}
    , dense1_{LAST_SIZE * LAST_SIZE * CONV4_OUT_CHANNELS, N_HIDDEN1}
    , batch_norm5_{N_HIDDEN1}
    , dense2_{N_HIDDEN1, N_CLASSES}
    , dropout_{DROPOUT_RATE}
{
    register_module("conv1_", conv1_);
    register_module("conv2_", conv2_);
    register_module("conv3_", conv3_);
    register_module("conv4_", conv4_);
    register_module("dense1_", dense1_);
    register_module("dense2_", dense2_);
}


torch::Tensor ArchitectureImpl::forward(torch::Tensor x)
{
    x = torch::leaky_relu(barch_norm1_->forward(conv1_->forward(x)))
    x = torch::leaky_relu(barch_norm2_->forward(conv2_->forward(x)))
    x = torch::leaky_relu(barch_norm3_->forward(conv3_->forward(x)))
    x = torch::leaky_relu(barch_norm4_->forward(conv4_->forward(x)))
    x = torch::flatten(x, 1); 
    x = dropput_->forward(batch_norm5->forward(dense1_->forward(x)));
    x = dense2_->forward(x)
    return torch::log_softmax(x, 1);
}

#if(UNIT_TEST_DatasetReader)
#include <boost/test/unit_test.hpp>

namespace
{
    void test_0()
    {
        int in_features = 100;
        int out_features = 10;

        Architecture architecture{in_features, out_features};
        auto s = architecture->parameters().size();
        BOOST_CHECK_EQUAL(s, 6);
    }

    void test_1()
    {
        auto batch_size = 20;
        auto row = 12;
        auto col = 13;
        auto cha = 3;
        auto x = torch::ones({batch_size, cha, row, col});

        int in_features = row * col * cha;
        int out_features = 10;
        Architecture architecture{in_features, out_features};
        auto y = architecture->forward(x);
        BOOST_CHECK_EQUAL((std::vector<int64_t>{batch_size, out_features}), y.sizes());
        torch::save(architecture, "model.pt");
    }

    void test_2()
    {
        auto row = 12;
        auto col = 13;
        auto cha = 3;
        int in_features = row * col * cha;
        int out_features = 10;
        Architecture architecture{in_features, out_features};
        torch::load(architecture, "model.pt");
    }
}

BOOST_AUTO_TEST_CASE(TEST_Architecture)
{
    std::cout << "Architecture\n";
    test_0();
    test_1();
}

#endif // UNIT_TEST_DatasetReader
