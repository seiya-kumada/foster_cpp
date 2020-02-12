#ifndef ADD_ATTRIBUTE_VECTORS_H
#define ADD_ATTRIBUTE_VECTORS_H
#include <torch/torch.h>
#include <vector>
#include "variational_auto_encoder.h"
#include <torch/torch.h>

template<typename Dataset>
torch::Tensor add_attribute_vectors(
    VariationalAutoEncoder& model, 
    const Dataset&          dataset,
    int                     batch_size, 
    const torch::Device&    device, 
    const torch::Tensor&    feature_vec)
{
    const auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
  
    auto data = std::begin(*loader)->data.to(device);

    torch::Tensor z_points {};
    torch::Tensor mu {};
    torch::Tensor log_var {};
    std::tie(z_points, mu, log_var) = model->predict(data);        

    std::vector<int> factors = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
    int total_size = batch_size * factors.size();
    torch::Tensor images {torch::empty({total_size, 3, 128, 128}, torch::kFloat)};
    int c = 0;
    for (auto i = 0; i < batch_size; ++i)
    {
        const auto& z_point = z_points[i].to(torch::kCPU);
        for (const auto& factor : factors)
        {
            const auto changed_z_point = torch::unsqueeze(z_point + feature_vec * factor, 0); 
            const auto changed_image = model->get_decoder()->forward(changed_z_point.to(device));
            images[c] = torch::squeeze(changed_image);
            c += 1;
        } 
    }
    return images;
}

#endif // ADD_ATTRIBUTE_VECTORS_H
 
