#ifndef AUTO_ENCODERE
#define AUTO_ENCODERE
#include <torch/torch.h>

class AutoEncoderImpl : public torch::nn::Module
{
public:
    AutoEncoderImpl(
        std::vector<int>    encoder_conv_filters,
        std::vector<int>    encoder_conv_kernel_sizes,
        std::vector<int>    encoder_conv_strides,
        std::vector<int>    decoder_conv_filters,
        std::vector<int>    decoder_conv_kernel_sizes,
        std::vector<int>    decoder_conv_strides,
        int                 z_dim,
        bool                uses_batch_norm = false,
        bool                uses_dropout = false
    );

    AutoEncoderImpl(const AutoEncoderImpl&) = delete;
    AutoEncoderImpl& operator=(const AutoEncoderImpl&) = delete;

    //torch::Tensor forward(torch::Tensor x);
    torch::nn::Sequential& get_encoder();
    torch::nn::Sequential& get_decoder();

private:
    std::vector<int>    encoder_conv_filters_;
    std::vector<int>    encoder_conv_kernel_sizes_;
    std::vector<int>    encoder_conv_strides_;
    std::vector<int>    decoder_conv_filters_;
    std::vector<int>    decoder_conv_kernel_sizes_;
    std::vector<int>    decoder_conv_strides_;
    int                 z_dim_;
    bool                uses_batch_norm_;
    bool                uses_dropout_;
    std::size_t         n_layers_encoder_;
    std::size_t         n_layers_decoder_;

    torch::nn::Sequential encoder_;
    torch::nn::Sequential decoder_;

    void build();
    torch::nn::Sequential build_encoder();
    torch::nn::Sequential build_decoder();
};

TORCH_MODULE(AutoEncoder);

#endif // AUTO_ENCODERE
