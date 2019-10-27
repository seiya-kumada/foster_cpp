#ifndef ARCHITECTURE
#define ARCHITECTURE
#include <torch/torch.h>

class ArchitectureImpl : public torch::nn::Module
{
public:
    ArchitectureImpl(int in_feature, int out_features);
    ArchitectureImpl(const ArchitectureImpl&) = delete;
    ArchitectureImpl& operator=(const ArchitectureImpl&) = delete;

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear   dense1_;
    torch::nn::Linear   dense2_;
    torch::nn::Linear   dense3_;
};

TORCH_MODULE(Architecture);

#endif // ARCHITECTURE
