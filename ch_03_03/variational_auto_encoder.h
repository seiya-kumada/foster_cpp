#ifndef VARIATIONAL_AUTO_ENCODER
#define VARIATIONAL_AUTO_ENCODER
#include <torch/torch.h>

class VariationalAutoEncoderImpl : public torch::nn::Module
{
public:
    VariationalAutoEncoderImpl(
        const std::vector<int>& encoder_conv_filters,
        const std::vector<int>& encoder_conv_kernel_sizes,
        const std::vector<int>& encoder_conv_strides,
        const std::vector<int>& decoder_conv_filters,
        const std::vector<int>& decoder_conv_kernel_sizes,
        const std::vector<int>& decoder_conv_strides,
        int                     z_dim,
        bool                    uses_batch_norm = false,
        bool                    uses_dropout = false
    );

    VariationalAutoEncoderImpl(const VariationalAutoEncoderImpl&) = delete;
    VariationalAutoEncoderImpl& operator=(const VariationalAutoEncoderImpl&) = delete;

    torch::Tensor forward(torch::Tensor x);
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

    torch::nn::Sequential   encoder_;
    torch::nn::Sequential   decoder_;
    torch::nn::Linear       mu_linear_;
    torch::nn::Linear       log_var_linear_;

    void build();
    torch::nn::Sequential build_encoder();
    //torch::nn::Sequential build_decoder();
};

TORCH_MODULE(VariationalAutoEncoder);

#endif // VARIATIONAL_AUTO_ENCODER
