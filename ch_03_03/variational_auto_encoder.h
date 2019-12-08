#ifndef VARIATIONAL_AUTO_ENCODER
#define VARIATIONAL_AUTO_ENCODER
#include <torch/torch.h>

class VariationalAutoEncoderImpl : public torch::nn::Module
{
public:
    VariationalAutoEncoderImpl(
        int                     encoder_start_channels,
        const std::vector<int64_t>& before_flatten_size,
        const std::vector<int>& encoder_conv_filters,
        const std::vector<int>& encoder_conv_kernel_sizes,
        const std::vector<int>& encoder_conv_strides,
        const std::vector<int>& decoder_conv_filters,
        const std::vector<int>& decoder_conv_kernel_sizes,
        const std::vector<int>& decoder_conv_strides,
        int                     z_dim,
        const torch::Device&    device,
        bool                    uses_batch_norm = false,
        bool                    uses_dropout = false
    );

    VariationalAutoEncoderImpl(const VariationalAutoEncoderImpl&) = delete;
    VariationalAutoEncoderImpl& operator=(const VariationalAutoEncoderImpl&) = delete;

    auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
    auto predict(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
    torch::nn::Sequential& get_encoder();
    torch::nn::Sequential& get_decoder();
    torch::nn::Linear& get_mu_linear();
    torch::nn::Linear& get_log_var_linear();

private:
    int                 encoder_start_channels_;
    std::vector<int64_t>    before_flatten_size_;
    int64_t             flatten_size_;
    std::vector<int>    encoder_conv_filters_;
    std::vector<int>    encoder_conv_kernel_sizes_;
    std::vector<int>    encoder_conv_strides_;
    std::vector<int>    decoder_conv_filters_;
    std::vector<int>    decoder_conv_kernel_sizes_;
    std::vector<int>    decoder_conv_strides_;
    int                 z_dim_;
    torch::Device       device_;
    bool                uses_batch_norm_;
    bool                uses_dropout_;
    std::size_t         n_layers_encoder_;
    std::size_t         n_layers_decoder_;

    torch::nn::Sequential   encoder_;
    torch::nn::Sequential   decoder_;
    torch::nn::Linear       mu_linear_;
    torch::nn::Linear       log_var_linear_;

    void build(const torch::Device& device);
    torch::nn::Sequential build_encoder(const torch::Device& device);
    torch::nn::Sequential build_decoder(const torch::Device& device);
};

TORCH_MODULE(VariationalAutoEncoder);

#endif // VARIATIONAL_AUTO_ENCODER
