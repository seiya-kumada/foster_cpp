#include "architecture.h"

namespace
{
    constexpr int N_HIDDEN1 = 200;
    constexpr int N_HIDDEN2 = 150;
}

Architecture::Architecture(int in_features, int out_features)
    : dense1_{in_features, N_HIDDEN1}
    , dense2_{N_HIDDEN1, N_HIDDEN2}
    , dense3_{N_HIDDEN2, out_features}
{
    register_module("dense1_", dense1_);
    register_module("dense2_", dense2_);
    register_module("dense3_", dense3_);
}


torch::Tensor Architecture::forward(torch::Tensor x)
{
    x = torch::flatten(x, 1); 
    x = torch::relu(dense1_->forward(x));
    x = torch::relu(dense2_->forward(x));
    x = dense3_->forward(x);
    return torch::log_softmax(x, 1);
}

#if(UNIT_TEST_DatasetReader)
#include <boost/test/unit_test.hpp>

namespace
{
    void test()
    {
        int in_features = 100;
        int out_features = 10;

        Architecture architecture{in_features, out_features};
        auto s = architecture.parameters().size();
        BOOST_CHECK_EQUAL(s, 6);
    }
}

BOOST_AUTO_TEST_CASE(TEST_Architecture)
{
    std::cout << "Architecture\n";
    test();
}

#endif // UNIT_TEST_DatasetReader
