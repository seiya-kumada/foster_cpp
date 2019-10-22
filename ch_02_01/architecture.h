#ifndef ARCHITECTURE
#define ARCHITECTURE
#include <torch/torch.h>

class Architecture : public torch::nn::Module
{
public:
    Architecture(int in_feature, int out_features);
    Architecture(const Architecture&) = delete;
    Architecture& operator=(const Architecture&) = delete;

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear   dense1_;
    torch::nn::Linear   dense2_;
    torch::nn::Linear   dense3_;
};
#endif // ARCHITECTURE
