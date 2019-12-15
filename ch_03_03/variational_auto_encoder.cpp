#include "variational_auto_encoder.h"
#include <iostream>

namespace
{
    inline int64_t prod(const std::vector<int64_t>& s)
    {
        return std::accumulate(
            std::begin(s), 
            std::end(s), 
            1, [
            ](int64_t init, int64_t v){ return init * v; });
    }
}

VariationalAutoEncoderImpl::VariationalAutoEncoderImpl(
     int                    encoder_start_channels,
    const std::vector<int64_t>& before_flatten_size,
    const std::vector<int>& encoder_conv_filters,
    const std::vector<int>& encoder_conv_kernel_sizes,
    const std::vector<int>& encoder_conv_strides,
    const std::vector<int>& decoder_conv_filters,
    const std::vector<int>& decoder_conv_kernel_sizes,
    const std::vector<int>& decoder_conv_strides,
    int                     z_dim,
    const torch::Device&    device,
    bool                    uses_batch_norm,
    bool                    uses_dropout
)
    : encoder_start_channels_{encoder_start_channels}
    , before_flatten_size_{before_flatten_size}
    , flatten_size_{prod(before_flatten_size)}
    , encoder_conv_filters_{encoder_conv_filters}
    , encoder_conv_kernel_sizes_{encoder_conv_kernel_sizes}
    , encoder_conv_strides_{encoder_conv_strides}
    , decoder_conv_filters_{decoder_conv_filters}
    , decoder_conv_kernel_sizes_{decoder_conv_kernel_sizes}
    , decoder_conv_strides_{decoder_conv_strides}
    , z_dim_{z_dim}
    , device_{device}
    , uses_batch_norm_{uses_batch_norm}
    , uses_dropout_{uses_dropout} 
    , n_layers_encoder_{encoder_conv_filters.size()}
    , n_layers_decoder_{decoder_conv_filters.size()}
    , encoder_{nullptr}
    , decoder_{nullptr}
    , mu_linear_{register_module("mu_linear", torch::nn::Linear{flatten_size_, z_dim})}
    , log_var_linear_{register_module("log_var_linear", torch::nn::Linear{flatten_size_, z_dim})}
{
    build(device);
}

torch::nn::Sequential& VariationalAutoEncoderImpl::get_encoder()
{
    return encoder_;
}

torch::nn::Sequential& VariationalAutoEncoderImpl::get_decoder()
{
    return decoder_;
}

torch::nn::Linear& VariationalAutoEncoderImpl::get_mu_linear()
{
    return mu_linear_;
}

torch::nn::Linear& VariationalAutoEncoderImpl::get_log_var_linear()
{
    return log_var_linear_;
}

void VariationalAutoEncoderImpl::build(const torch::Device& device)
{
    encoder_ = build_encoder(device);
    register_module("encoder", encoder_);
    decoder_ = build_decoder(device);
    register_module("decoder", decoder_);
}

namespace
{
    inline torch::Tensor sample(
        torch::Tensor           mu, 
        torch::Tensor           log_var, 
        int                     z_dim, 
        const torch::Device&    device)
    {
        // 2-dim standard normal distribution
        const auto epsilon = torch::empty(mu.sizes()).normal_().to(device);
        return mu + torch::exp(log_var / 2) * epsilon;
    }
}

auto VariationalAutoEncoderImpl::predict(torch::Tensor x)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
{
    x = encoder_->forward(x);
    const auto mu = mu_linear_->forward(x);
    const auto log_var = log_var_linear_->forward(x);
    x = sample(mu, log_var, z_dim_, device_);
    return std::make_tuple(x, mu, log_var);

}

auto VariationalAutoEncoderImpl::forward(torch::Tensor x) 
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
{
    auto out = predict(x);
    x = decoder_->forward(std::get<0>(out));
    std::get<0>(out) = x;
    return out; 
}

torch::nn::Sequential VariationalAutoEncoderImpl::build_encoder(const torch::Device& device)
{
    torch::nn::Sequential encoder {};
    int in_channels = encoder_start_channels_;
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

namespace
{
    inline torch::Tensor reshape(torch::Tensor x, torch::IntArrayRef shape)
    {
        const auto s = x.sizes(); // batch size
        return x.reshape({s[0], shape[0], shape[1], shape[2]});
    }
}

torch::nn::Sequential VariationalAutoEncoderImpl::build_decoder(const torch::Device& device)
{
    // (batch_size, z_dim)
    torch::nn::Sequential decoder {};

    decoder->push_back(
        torch::nn::Linear(z_dim_, flatten_size_)
    );
   
    decoder->push_back(
        torch::nn::Functional(reshape, before_flatten_size_)
    );

    int in_channels = encoder_conv_filters_.back();
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
        
        if (i < n_layers_decoder_ - 1)
        {
            if (uses_batch_norm_)
            {
                decoder->push_back(
                    torch::nn::BatchNorm(decoder_conv_filters_[i])
                );
            }

            decoder->push_back(
                torch::nn::Functional(torch::leaky_relu, 0.3)
            );
            
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
        in_channels = out_channels;
    }
    return decoder;
}

#if(UNIT_TEST_VariationalAutoEncoder)
#include <boost/test/unit_test.hpp>
#include <vector>

namespace
{

    const std::vector<int64_t> BEFORE_FLATTEN_SIZE  {64, 7, 7};
    constexpr int FLATTEN_SIZE = 64 * 49;
    
    template<typename M>
    void print_parameters(const M& model) //VariationalAutoEncoder& model)
    {
        int s {0};
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
            s += c;
        }
        std::cout << "> total number of parameters: " << s << std::endl;
    }

    void print_parameters(const VariationalAutoEncoder& model)
    {
        int s {0};
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
            s += c;
        }
        std::cout << "total number of parameters: " << s << std::endl;
    }

    void test_0()
    {
        int              encoder_start_channles     {1};
        std::vector<int64_t> before_flatten_size    {64, 7, 7};
        std::vector<int> encoder_conv_filters       {32, 64, 64,  64};
        std::vector<int> encoder_conv_kernel_sizes  { 3,  3,  3,  3};
        std::vector<int> encoder_conv_strides       { 1,  2,  2,  1};
        std::vector<int> decoder_conv_filters       {64, 64, 32,  1};
        std::vector<int> decoder_conv_kernel_sizes  { 3,  3,  3,  3};
        std::vector<int> decoder_conv_strides       { 1,  2,  2,  1};
        int z_dim {2};
        torch::Device device{torch::kCPU};


        VariationalAutoEncoder vae {  
            encoder_start_channles,
            before_flatten_size,
            std::move(encoder_conv_filters),
            std::move(encoder_conv_kernel_sizes),
            std::move(encoder_conv_strides),
            std::move(decoder_conv_filters),
            std::move(decoder_conv_kernel_sizes),
            std::move(decoder_conv_strides),
            z_dim,
            device
        };

        //print_parameters(vae);

        int batch_size {2};
        int cha {1};
        int row {28};
        int col {28};
        auto x = torch::zeros({batch_size, cha, row, col});
        auto y = vae->get_encoder()->forward(x);
        // 7x7x64=3136
        BOOST_CHECK_EQUAL(y.sizes(), (std::vector<int64_t>{batch_size, FLATTEN_SIZE})); 

        float a = y[0][0].item<float>();
        auto mu = vae->get_mu_linear()->forward(y);
        float b = y[0][0].item<float>();
        BOOST_CHECK_CLOSE(a, b, 1.0e-5);
        BOOST_CHECK_EQUAL(mu.sizes(), (std::vector<int64_t>{batch_size, z_dim})); 
        auto log_var = vae->get_log_var_linear()->forward(y);
        BOOST_CHECK_EQUAL(log_var.sizes(), (std::vector<int64_t>{batch_size, z_dim})); 

        mu = torch::zeros({batch_size, z_dim});
        log_var = torch::zeros({batch_size, z_dim});
        //std::cout << mu << std::endl;
        //std::cout << log_var << std::endl;
        
        auto z = sample(mu, log_var, z_dim, device);
        BOOST_CHECK_EQUAL(z.sizes(), (std::vector<int64_t>{batch_size, z_dim})); 
        
        z = sample(mu, log_var, z_dim, device);
        BOOST_CHECK_EQUAL(z.sizes(), (std::vector<int64_t>{batch_size, z_dim})); 
        
        x = torch::zeros({batch_size, FLATTEN_SIZE});
        y = torch::nn::Functional(reshape, BEFORE_FLATTEN_SIZE)(x);
        BOOST_CHECK_EQUAL(y.sizes(), (std::vector<int64_t>{batch_size, 64, 7, 7})); 

        torch::Tensor dummy1 {};
        torch::Tensor dummy2 {};
        x = torch::zeros({batch_size, cha, row, col});
        std::tie(y, dummy1, dummy2) = vae->forward(x);
        BOOST_CHECK_EQUAL(y.sizes(), (std::vector<int64_t>{batch_size, cha, row, col})); 
        auto u = torch::mse_loss(y, x);
        BOOST_CHECK_EQUAL(u.sizes(), (std::vector<int64_t>{})); 


        x = 2 * torch::ones({batch_size, z_dim});
        auto w = x.sum({1});
        BOOST_CHECK_EQUAL(w.sizes(), (std::vector<int64_t>{batch_size})); 

        w = x * x;
        BOOST_CHECK_EQUAL(w.sizes(), (std::vector<int64_t>{batch_size, z_dim})); 
        BOOST_CHECK_EQUAL(4, w[0][0].item<double>());
        BOOST_CHECK_EQUAL(4, w[0][1].item<double>());

        x = torch::zeros({batch_size, z_dim});
        y = torch::exp(x);
        BOOST_CHECK_EQUAL(x.sizes(), (std::vector<int64_t>{batch_size, z_dim})); 
    }

    torch::Tensor func()
    {
        auto a = torch::ones({3});
        return a;
    }

    struct TensorTest
    {
        TensorTest()
            : tensor_{torch::ones({3})} {}

        torch::Tensor tensor_;

        torch::Tensor get() { return tensor_; }
    };

    void test_1()
    {
        auto a = func();
        BOOST_CHECK_EQUAL(1, a[0].item<float>());
        BOOST_CHECK_EQUAL(1, a[1].item<float>());
        BOOST_CHECK_EQUAL(1, a[2].item<float>());
        const auto e = torch::empty({3, 2});
        BOOST_CHECK_EQUAL(e.sizes(), (std::vector<int64_t>{3, 2}));

        TensorTest test{};
        auto t = test.get();
        t[0] = 4; 
        auto s = test.get();
        BOOST_CHECK_EQUAL(4, s[0].item<float>());
    }
}


BOOST_AUTO_TEST_CASE(TEST_VariationalAutoEncoder)
{
    std::cout << "VariationalAutoEncoder\n";
    test_0();
    test_1();
}

#endif // UNIT_TEST_VariationalAutoEncoder
