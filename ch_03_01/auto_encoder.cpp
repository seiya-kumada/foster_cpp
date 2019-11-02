#include "auto_encoder.h"

AutoEncoderImpl::AutoEncoderImpl(
    std::vector<int>    encoder_conv_filters,
    std::vector<int>    encoder_conv_kernel_sizes,
    std::vector<int>    encoder_conv_strides,
    std::vector<int>    decoder_conv_filters,
    std::vector<int>    decoder_conv_kernel_sizes,
    std::vector<int>    decoder_conv_strides,
    int                 z_dim,
    bool                uses_batch_norm,
    bool                uses_dropout
)
    : encoder_conv_filters_{encoder_conv_filters}
    , encoder_conv_kernel_sizes_{encoder_conv_kernel_sizes}
    , encoder_conv_strides_{encoder_conv_strides}
    , decoder_conv_filters_{decoder_conv_filters}
    , decoder_conv_kernel_sizes_{decoder_conv_kernel_sizes}
    , decoder_conv_strides_{decoder_conv_strides}
    , z_dim_{z_dim}
    , uses_batch_norm_{uses_batch_norm}
    , uses_dropout_{uses_dropout} 
    , n_layers_encoder_{encoder_conv_filters.size()}
    , n_layers_decoder_{decoder_conv_filters.size()}

{

}

void AutoEncoderImpl::build()
{
    // (batch_size, channels, rows, cols)
    torch::nn::Sequential encoder {};
    for (auto i = 0; i < n_layers_encoder_; ++i)
    {
        encoder->push_back(
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(3, encoder_conv_filters_[i], encoder_conv_kernel_sizes_[i])
                    .stride(encoder_conv_strides_[i])
                    .padding(1)
            )
        );

        encoder->push_back(
            torch::nn::Functional(torch::leaky_relu, 0.3)
        );

        if (uses_batch_norm_)
        {
            encoder->push_back(
                torch::nn::BatchNorm(encoder_conv_filters_[i])
            ); 
        }

        if (uses_dropout_)
        {
            encoder->push_back(
                torch::nn::Dropout(0.25)
            ); 
        }
    }
}


#if(UNIT_TEST_AutoEncoder)
#include <boost/test/unit_test.hpp>

namespace
{
    void test_0()
    {
        std::vector<int> encoder_conv_filters       {32, 64, 64,  6};
        std::vector<int> encoder_conv_kernel_sizes  { 3,  3,  3,  3};
        std::vector<int> encoder_conv_strides       { 1,  2,  2,  1};
        std::vector<int> decoder_conv_filters       {64, 64, 32,  1};
        std::vector<int> decoder_conv_kernel_sizes  { 3,  3,  3,  3};
        std::vector<int> decoder_conv_strides       { 1,  2,  2,  1};
        int z_dim {2};

        AutoEncoder ae {  
            std::move(encoder_conv_filters),
            std::move(encoder_conv_kernel_sizes),
            std::move(encoder_conv_strides),
            std::move(decoder_conv_filters),
            std::move(decoder_conv_kernel_sizes),
            std::move(decoder_conv_strides),
            z_dim, 
        };
    }
}

BOOST_AUTO_TEST_CASE(TEST_AutoEncoder)
{
    std::cout << "AutoEncoder\n";
    test_0();
}

#endif // UNIT_TEST_AutoEncoder
