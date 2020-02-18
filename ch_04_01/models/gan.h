#ifndef GAN_INCLUDE 
#define GAN_INCLUDE
#include <torch/torch.h>

class GANImpl : public torch::nn::Module
{
public:
    struct Params
    {
        std::vector<int>    conv_filters_;
        std::vector<int>    kernel_size_;
        std::vector<int>    strides_;
        double              batch_norm_momentum_;
        std::string         activation_;
        double              dropout_rate_;
        double              learning_rate_;
    };

    GANImpl(
        const Params&           dicriminator_params,
        const Params&           generator_params,
        int                     generator_initial_dense_layer_size,
        const std::vector<int>& generator_upsample,
        const std::string&      optimizer,
        int                     z_dim,
        const torch::Device&    device
    );
    
private:
    Params              dicriminator_params_;
    Params              generator_params_;
    int                 generator_initial_dense_layer_size_;
    std::vector<int>    generator_upsample_;
    std::string         optimizer_;
    int                 z_dim_;
    torch::Device       device_;
};
TORCH_MODULE(GAN);
#endif // GAN_INCLUDE
