#ifndef MAKE_ATTRIBUTE_VECTORS_H
#define MAKE_ATTRIBUTE_VECTORS_H
#include <torch/torch.h>

class VariationalAutoEncoder;

void make_attribute_vectors(
    VariationalAutoEncoder& model, 
    int                     batch_size, 
    const torch::Device&    device);

#endif // MAKE_ATTRIBUTE_VECTORS_H
 

