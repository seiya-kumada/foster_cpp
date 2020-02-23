#ifndef GAN_INCLUDE 
#define GAN_INCLUDE
#include <torch/torch.h>
#include <boost/optional.hpp>
#include <initializer_list>

class GANImpl : public torch::nn::Module
{
public:
    struct Params
    {
        int                     start_channels_;
        std::vector<int64_t>    flatten_shape_;
        int64_t                 flatten_size_;
        std::vector<int>        conv_filters_;
        std::vector<int>        kernel_size_;
        std::vector<int>        strides_;
        boost::optional<double> batch_norm_momentum_;
        std::string             activation_;
        boost::optional<double> dropout_rate_;
        double                  learning_rate_;
        int64_t                 n_layers_;

        Params(
            int                             start_channels,
            const std::vector<int64_t>&     flatten_shape_,
            const std::vector<int>&         conv_filters,
            const std::vector<int>&         kernel_size,
            const std::vector<int>&         strides,
            const boost::optional<double>&  batch_norm_momentum,
            const std::string&              activation,
            const boost::optional<double>&  dropout_rate,
            const double&                   learning_rate);
    };

    GANImpl(
        const Params&           discriminator_params,
        const Params&           generator_params,
        const std::vector<int>& generator_upsample,
        const std::string&      optimizer,
        int                     z_dim,
        const torch::Device&    device
    );

    const torch::nn::Sequential& get_discriminator() const { return discriminator_; }
    torch::nn::Sequential& get_discriminator() { return discriminator_; }

    const torch::nn::Sequential& get_generator() const { return generator_; }
    torch::nn::Sequential& get_generator() { return generator_; }
    
    const torch::nn::Sequential& get_adversarial() const { return adversarial_; }
    torch::nn::Sequential& get_adversarial() { return adversarial_; }

private:
    Params              discriminator_params_;
    Params              generator_params_;
    std::vector<int>    generator_upsample_;
    std::string         optimizer_;
    int                 z_dim_;
    torch::Device       device_;

    torch::nn::Sequential   discriminator_;
    torch::nn::Sequential   generator_;
    torch::nn::Sequential   adversarial_;

    void build();
    torch::nn::Sequential build_discriminator();
    torch::nn::Sequential build_generator();
    void build_adversarial();
};
TORCH_MODULE(GAN);
#endif // GAN_INCLUDE
