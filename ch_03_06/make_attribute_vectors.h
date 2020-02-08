#ifndef MAKE_ATTRIBUTE_VECTORS_H
#define MAKE_ATTRIBUTE_VECTORS_H
#include <torch/torch.h>

class VariationalAutoEncoder;

torch::Tensor make_attribute_vectors(
    VariationalAutoEncoder& model, 
    int                     batch_size, 
    const torch::Device&    device, 
    int                     iterations,
    int                     z_dim,
    const std::string&      label,
    bool                    is_verbose=false);

#endif // MAKE_ATTRIBUTE_VECTORS_H
 

