#include "gan.h"

namespace
{
    void print_parameters(const GAN& model)
    {
        int s {0};
        for (const auto& pair : model->named_parameters())
        {
            const auto& key = pair.key();
            const auto& value = pair.value();
            //std::cout << ": " << value.sizes() << std::endl;
            auto c = 1;
            for (const auto& v : value.sizes())
            {
                c *= v; 
            }
            std::cout << key << ": " << pair.value().sizes() << " -> " << c << std::endl;
            s += c;
        }
        std::cout << "total number of parameters: " << s << std::endl;
    }

    inline int64_t prod(const std::vector<int64_t>& s)
    {
        return std::accumulate(
            std::begin(s), 
            std::end(s), 
            1, [
            ](int64_t init, int64_t v){ return init * v; });
    }
}

GANImpl::Params::Params(
    int                             start_channels,
    const std::vector<int64_t>&     before_flatten_shape,
    const std::vector<int>&         conv_filters,
    const std::vector<int>&         kernel_size,
    const std::vector<int>&         strides,
    const boost::optional<double>&  batch_norm_momentum,
    const std::string&              activation,
    const boost::optional<double>&  dropout_rate,
    const double&                   learning_rate
)
    : start_channels_{start_channels}
    , before_flatten_shape_{before_flatten_shape}
    , flatten_size_{prod(before_flatten_shape)}
    , conv_filters_{conv_filters}
    , kernel_size_{kernel_size}
    , strides_{strides}
    , batch_norm_momentum_{batch_norm_momentum}
    , activation_{activation}
    , dropout_rate_{dropout_rate}
    , learning_rate_{learning_rate}
    , n_layers_{conv_filters.size()}
{

}

GANImpl::GANImpl(
    const Params&           discriminator_params,
    const Params&           generator_params,
    const std::vector<int>& generator_initial_dense_layer_size,
    const std::vector<int>& generator_upsample,
    const std::string&      optimizer,
    int                     z_dim,
    const torch::Device&    device
)
    : discriminator_params_{discriminator_params}
    , generator_params_{generator_params}
    , generator_initial_dense_layer_size_{generator_initial_dense_layer_size}
    , generator_upsample_{generator_upsample}
    , optimizer_{optimizer}
    , z_dim_{z_dim}
    , device_{device} 
{
    build();
}

void GANImpl::build()
{
    discriminator_ = build_discriminator();
    register_module("discriminator", discriminator_);

    build_generator();
    build_adversarial();
}

namespace
{
    inline torch::nn::Functional get_activation(const std::string& name)
    {
        if (name == "leaky_relu")
        {
            return torch::nn::Functional(torch::leaky_relu, 0.2);
        }
        else
        {
            return torch::nn::Functional(torch::relu);
        }
    }
}

torch::nn::Sequential GANImpl::build_discriminator()
{
    torch::nn::Sequential discriminator {};
    auto in_channels = discriminator_params_.start_channels_;
    //auto out_channels = 0;
    for (auto i = 0; i < discriminator_params_.n_layers_; ++i)
    {
        auto out_channels = discriminator_params_.conv_filters_[i];
        auto c = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, discriminator_params_.kernel_size_[i])
                .stride(discriminator_params_.strides_[i])
                .padding(2)
        );
        torch::nn::init::normal_(c->weight, 0, 0.2);
        torch::nn::init::zeros_(c->bias);
        discriminator->push_back("conv2d_" + std::to_string(i), std::move(c));
        
        if (discriminator_params_.batch_norm_momentum_ && i > 0)
        {
            auto b = torch::nn::BatchNorm(
                torch::nn::BatchNormOptions(out_channels).momentum(discriminator_params_.batch_norm_momentum_.value())
            );
            discriminator->push_back("batch_norm_" + std::to_string(i), std::move(b));
        }
        discriminator->push_back("activation_" + std::to_string(i), get_activation(discriminator_params_.activation_));

        if (discriminator_params_.dropout_rate_)
        {
            discriminator->push_back("dropout_" + std::to_string(i), torch::nn::Dropout(discriminator_params_.dropout_rate_.value()));
        }
        in_channels = out_channels;
    }
    
    discriminator->push_back("flatten", torch::nn::Functional(torch::flatten, 1, -1));
    
    auto l = torch::nn::Linear(discriminator_params_.flatten_size_, 1);
    torch::nn::init::normal_(l->weight, 0, 0.2);
    torch::nn::init::zeros_(l->bias);
    discriminator->push_back("linear", std::move(l));
    
    discriminator->push_back("sigmoid", torch::nn::Functional(torch::sigmoid));

    return discriminator;
}

void GANImpl::build_generator()
{
}

void GANImpl::build_adversarial()
{
}

#if(UNIT_TEST_GAN)
#include <boost/test/unit_test.hpp>

namespace
{
    template<typename T>
    struct Type;

    void test_0()
    {
        GANImpl::Params discriminator_params {
            1, // start_channels
            {128, 4, 4},
            {64, 64, 128, 128}, // conv_filters
            {5, 5, 5, 5}, // kernel_size
            {2, 2, 2, 1}, // strides             
            boost::none, // batch_norm_momentum
            "relu", // activation          
            0.4, // dropout_rate       
            0.0008 // learning_rate  
        };

        GANImpl::Params generator_params {
            1, // start_channels
            {128, 64, 64, 1}, // conv_filters
            {128, 4, 4}, // dummy
            {5, 5, 5, 5}, // kernel_size
            {1, 1, 1, 1}, // strides             
            0.9, // batch_norm_momentum
            "relu", // activation          
            boost::none, // dropout_rate       
            0.0004 // learning_rate  
        };

        GAN gan {
            discriminator_params,
            generator_params,
            std::vector<int>{7, 7, 64}, // generator_initial_dense_layer_size
            std::vector<int>{2, 2, 1, 1}, // generator_upsample,
            "rmsprop", // optimizer,
            100, // z_dim,
            torch::kCPU
        };

        auto& d = gan->get_discriminator();
        auto batch_size = 10;
        auto rows = 28;
        auto cols = 28;
        auto channels = 1;
        auto x = torch::ones({batch_size, channels, rows, cols});
        auto y = d->forward(x);
        BOOST_CHECK_EQUAL(y.sizes(), (std::vector<int64_t>{batch_size, 1})); 
        //for (auto i = 0; i < d->size(); ++i)
        //{
        //    std::cout << d->ptr(i)->name() << std::endl;
        //}
        //print_parameters(gan);
    }
}

BOOST_AUTO_TEST_CASE(TEST_GAN)
{
    std::cout << "GAN\n";
    test_0();
}

#endif // UNIT_TEST_GAN
