#include "gan.h"

GANImpl::GANImpl(
    const Params&           dicriminator_params,
    const Params&           generator_params,
    int                     generator_initial_dense_layer_size,
    const std::vector<int>& generator_upsample,
    const std::string&      optimizer,
    int                     z_dim,
    const torch::Device&    device
)
    : dicriminator_params_{dicriminator_params}
    , generator_params_{generator_params}
    , generator_initial_dense_layer_size_{generator_initial_dense_layer_size}
    , generator_upsample_{generator_upsample}
    , optimizer_{optimizer}
    , z_dim_{z_dim}
    , device_{device} {}
