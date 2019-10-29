#ifndef ARCHITECTURE
#define ARCHITECTURE
#include <torch/torch.h>

class ArchitectureImpl : public torch::nn::Module
{
public:
    ArchitectureImpl();
    ArchitectureImpl(const ArchitectureImpl&) = delete;
    ArchitectureImpl& operator=(const ArchitectureImpl&) = delete;

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1_;
    torch::nn::Conv2d conv2_;
    torch::nn::Conv2d conv3_;
    torch::nn::Conv2d conv4_;
    torch::nn::Linear dense1_;
    torch::nn::Linear dense2_;
};

TORCH_MODULE(Architecture);

#endif // ARCHITECTURE
