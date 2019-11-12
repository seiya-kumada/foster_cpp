#include "auto_encoder.h"
#include <iostream>

namespace
{
    constexpr int ENCODER_START_CHANNELS        {1};
    constexpr int DECODER_START_CHANNELS        {64};
    constexpr int FLATTEN_SIZE                  {3136};
    const std::vector<int64_t> BEFORE_FLATTEN_SIZE  {64, 7, 7};
}

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
    , encoder_{nullptr}
    , decoder_{nullptr}
{
    build();
}

torch::nn::Sequential& AutoEncoderImpl::get_encoder()
{
    return encoder_;
}

torch::nn::Sequential& AutoEncoderImpl::get_decoder()
{
    return decoder_;
}

void AutoEncoderImpl::build()
{
    encoder_ = build_encoder();
    register_module("encoder_", encoder_);
    decoder_ = build_decoder();
    register_module("decoder_", decoder_);
}

torch::nn::Sequential AutoEncoderImpl::build_encoder()
{
    // (batch_size, channels, rows, cols)
    torch::nn::Sequential encoder {};
    int in_channels = ENCODER_START_CHANNELS;
    for (auto i = 0; i < n_layers_encoder_; ++i)
    {
        auto out_channels = encoder_conv_filters_[i];
        encoder->push_back(
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, encoder_conv_kernel_sizes_[i])
                    .stride(encoder_conv_strides_[i])
                    .padding(1)
            )
        );
        in_channels = out_channels;

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

    encoder->push_back(
        torch::nn::Functional(torch::flatten, 1, -1)
    );

    encoder->push_back(
        torch::nn::Linear(FLATTEN_SIZE, z_dim_)
    );
    return encoder;
}

namespace
{
    inline torch::Tensor reshape(torch::Tensor x, torch::IntArrayRef shape)
    {
        const auto s = x.sizes(); // batch size
        return x.reshape({s[0], shape[0], shape[1], shape[2]});
    }
}

torch::nn::Sequential AutoEncoderImpl::build_decoder()
{
    // (batch_size, z_dim)
    torch::nn::Sequential decoder {};

    decoder->push_back(
        torch::nn::Linear(z_dim_, FLATTEN_SIZE)
    );
   
    decoder->push_back(
        torch::nn::Functional(reshape, BEFORE_FLATTEN_SIZE)
    );

    int in_channels = DECODER_START_CHANNELS;
    for (auto i = 0; i < n_layers_decoder_; ++i)
    {
        auto out_channels = decoder_conv_filters_[i];
        decoder->push_back(
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, decoder_conv_kernel_sizes_[i])
                    .stride(decoder_conv_strides_[i])
                    .padding(1)
                    .output_padding(decoder_conv_strides_[i] + 2 * 1 - decoder_conv_kernel_sizes_[i])
                    .transposed(true)
            )
        );
        in_channels = out_channels;
        
        if (i < n_layers_decoder_ - 1)
        {
            decoder->push_back(
                torch::nn::Functional(torch::leaky_relu, 0.3)
            );
            if (uses_batch_norm_)
            {
                decoder->push_back(
                    torch::nn::BatchNorm(decoder_conv_filters_[i])
                );
            }
            if (uses_dropout_)
            {
                decoder->push_back(
                    torch::nn::Dropout(0.25)
                );
            }
        }
        else
        {
            decoder->push_back(
                torch::nn::Functional(torch::sigmoid)
            );
        }
    }

    return decoder;
}

torch::Tensor AutoEncoderImpl::forward(torch::Tensor x)
{
    x = encoder_->forward(x);
    return decoder_->forward(x);
}

#if(UNIT_TEST_AutoEncoder)
#include <boost/test/unit_test.hpp>

namespace
{
    void print_parameters(const AutoEncoder& model)
    {
        for (const auto& pair : model->named_parameters())
        {
            const auto& key = pair.key();
            const auto& value = pair.value();
            //<< ": " << pair.value().sizes() << std::endl;
            auto c = 1;
            for (const auto& v : value.sizes())
            {
                c *= v; 
            }
            std::cout << key << ": " << pair.value().sizes() << " -> " << c << std::endl;
        }
    }

    void test_0()
    {
        std::vector<int> encoder_conv_filters       {32, 64, 64,  64};
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

        int batch_size {1};
        int cha {1};
        int row {28};
        int col {28};
        auto x = torch::zeros({batch_size, cha, row, col});
        auto y = ae->get_encoder()->forward(x);
        BOOST_CHECK_EQUAL(y.sizes(), (std::vector<int64_t>{batch_size, z_dim})); 
        
        auto s = 2 * torch::ones({batch_size, z_dim});
        auto t = ae->get_decoder()->forward(s);
        BOOST_CHECK_EQUAL(t.sizes(), (std::vector<int64_t>{batch_size, cha, row, col})); 
        
        auto u = ae->forward(x);
        BOOST_CHECK_EQUAL(u.sizes(), (std::vector<int64_t>{batch_size, cha, row, col})); 
    }
}

BOOST_AUTO_TEST_CASE(TEST_AutoEncoder)
{
    std::cout << "AutoEncoder\n";
    test_0();
}

#endif // UNIT_TEST_AutoEncoder
