#include "variational_auto_encoder.h"

namespace
{
    constexpr int ENCODER_START_CHANNELS    {1};
    constexpr int FLATTEN_SIZE              {3136};
}

VariationalAutoEncoderImpl::VariationalAutoEncoderImpl(
    const std::vector<int>& encoder_conv_filters,
    const std::vector<int>& encoder_conv_kernel_sizes,
    const std::vector<int>& encoder_conv_strides,
    const std::vector<int>& decoder_conv_filters,
    const std::vector<int>& decoder_conv_kernel_sizes,
    const std::vector<int>& decoder_conv_strides,
    int                     z_dim,
    bool                    uses_batch_norm,
    bool                    uses_dropout
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
    , mu_linear_{register_module("mu_linear", torch::nn::Linear{FLATTEN_SIZE, z_dim})}
    , log_var_linear_{register_module("log_var_linear", torch::nn::Linear{FLATTEN_SIZE, z_dim})}
{
    build();
}

torch::nn::Sequential& VariationalAutoEncoderImpl::get_encoder()
{
    return encoder_;
}

torch::nn::Sequential& VariationalAutoEncoderImpl::get_decoder()
{
    return decoder_;
}

void VariationalAutoEncoderImpl::build()
{
    encoder_ = build_encoder();

}

namespace
{
    inline torch::Tensor sample(torch::Tensor mu, torch::Tensor log_var, int z_dim)
    {
        // 2-dim standard normal distribution
        const auto epsilon = torch::empty(z_dim).normal_();
        return mu + torch::exp(log_var / 2) * epsilon;
    }
}

torch::Tensor VariationalAutoEncoderImpl::forward(torch::Tensor x)
{
    x = encoder_->forward(x);
    auto mu = mu_linear_->forward(x);
    auto log_var = log_var_linear_->forward(x);
    x = sample(mu, log_var, z_dim_);
    return x;
}

torch::nn::Sequential VariationalAutoEncoderImpl::build_encoder()
{
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
        
        if (uses_batch_norm_)
        {
            encoder->push_back(
                torch::nn::BatchNorm(out_channels)
            ); 
        }

        encoder->push_back(
            torch::nn::Functional(torch::leaky_relu, 0.3)
        );

        if (uses_dropout_)
        {
            encoder->push_back(
                torch::nn::Dropout(0.25)
            ); 
        }
        in_channels = out_channels;
    }

    encoder->push_back(
        torch::nn::Functional(torch::flatten, 1, -1)
    );
    return encoder;
}
