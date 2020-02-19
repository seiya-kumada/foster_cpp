#include "gan.h"

GANImpl::Params::Params(
    int                     start_channels,
    const std::vector<int>& conv_filters,
    const std::vector<int>& kernel_size,
    const std::vector<int>& strides,
    const double&           batch_norm_momentum,
    const std::string&      activation,
    const double&           dropout_rate,
    const double&           learning_rate
)
    : start_channels_{start_channels}
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
    int                     generator_initial_dense_layer_size,
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
    build_discriminator();
    build_generator();
    build_adversarial();
}

void GANImpl::build_discriminator()
{
    torch::nn::Sequential discriminator {};
    auto in_channels = discriminator_params_.start_channels_;
    for (auto i = 0; i < discriminator_params_.n_layers_; ++i)
    {
        auto out_channels = discriminator_params_.conv_filters_[i];
        auto c = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, discriminator_params_.kernel_size_[i])
                .stride(discriminator_params_.strides_[i])
                .padding(1)
        );
        torch::nn::init::normal_(c->weight, 0, 0.2);
        torch::nn::init::zeros_(c->bias);
    }
}

void GANImpl::build_generator()
{
}

void GANImpl::build_adversarial()
{
}

#if(UNIT_TEST_GAN)
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(TEST_GAN)
{
    std::cout << "GAN\n";
}

#endif // UNIT_TEST_GAN
