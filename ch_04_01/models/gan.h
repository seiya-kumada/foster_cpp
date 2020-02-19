#ifndef GAN_INCLUDE 
#define GAN_INCLUDE
#include <torch/torch.h>

class GANImpl : public torch::nn::Module
{
public:
    struct Params
    {
        int                 start_channels_;
        std::vector<int>    conv_filters_;
        std::vector<int>    kernel_size_;
        std::vector<int>    strides_;
        double              batch_norm_momentum_;
        std::string         activation_;
        double              dropout_rate_;
        double              learning_rate_;
        std::size_t         n_layers_;

        Params(
            int                     start_channels,
            const std::vector<int>& conv_filters,
            const std::vector<int>& kernel_size,
            const std::vector<int>& strides,
            const double&           batch_norm_momentum,
            const std::string&      activation,
            const double&           dropout_rate,
            const double&           learning_rate);
    };

    GANImpl(
        const Params&           discriminator_params,
        const Params&           generator_params,
        int                     generator_initial_dense_layer_size,
        const std::vector<int>& generator_upsample,
        const std::string&      optimizer,
        int                     z_dim,
        const torch::Device&    device
    );
    
private:
    Params              discriminator_params_;
    Params              generator_params_;
    int                 generator_initial_dense_layer_size_;
    std::vector<int>    generator_upsample_;
    std::string         optimizer_;
    int                 z_dim_;
    torch::Device       device_;
    
    void build();
    void build_discriminator();
    void build_generator();
    void build_adversarial();
};
TORCH_MODULE(GAN);
#endif // GAN_INCLUDE
