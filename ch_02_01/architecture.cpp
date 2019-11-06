#include "architecture.h"

namespace
{
    constexpr int N_HIDDEN1 = 200;
    constexpr int N_HIDDEN2 = 150;
}

ArchitectureImpl::ArchitectureImpl(int in_features, int out_features)
    //: dense1_{in_features, N_HIDDEN1}
    //, dense2_{N_HIDDEN1, N_HIDDEN2}
    //, dense3_{N_HIDDEN2, out_features}
    : dense1_{register_module("dense1_", torch::nn::Linear{in_features, N_HIDDEN1})}
    , dense2_{N_HIDDEN1, N_HIDDEN2}
    , dense3_{N_HIDDEN2, out_features}
{
    //register_module("dense1_", dense1_);
    register_module("dense2_", dense2_);
    register_module("dense3_", dense3_);
}


torch::Tensor ArchitectureImpl::forward(torch::Tensor x)
{
    x = torch::flatten(x, 1); // [batch_size, in_features]
    x = torch::relu(dense1_->forward(x)); // [batch_size, N_HIDDEN1]
    x = torch::relu(dense2_->forward(x)); // [batch_size, N_HIDDEN2]
    x = dense3_->forward(x); // [batch_size, out_features]
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

    template<typename Derived>
    class Base
    {
    public:
        void interface()
        {
            static_cast<Derived*>(this)->implementation();
        }

        void implementation()
        {
            std::cout << "Base::implementation" << std::endl;
        }
    };


    class Derived_1 : public Base<Derived_1>
    {
    public:
        void implementation()
        {
            std::cout << "Derived_1::implementation" << std::endl;
        }
    };

    class Derived_2 : public Base<Derived_2>
    {
    public:
    };


    void test_3()
    {
        Derived_1 d1{};
        d1.interface();
        Derived_2 d2{};
        d2.interface();
    }
}

BOOST_AUTO_TEST_CASE(TEST_Architecture)
{
    std::cout << "Architecture\n";
    test_0();
    test_1();
}

#endif // UNIT_TEST_DatasetReader
