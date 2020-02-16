#ifndef MORPH_FACES
#define MORPH_FACES

#include <torch/torch.h>
#include <vector>
#include "variational_auto_encoder.h"

template<typename Dataset>
auto morph_faces_(
    VariationalAutoEncoder& model, 
    const Dataset&          dataset,
    const torch::Device&    device)  
    -> std::pair<torch::Tensor, torch::Tensor> 
{
    const auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(2).workers(2));
  
    auto data = std::begin(*loader)->data.to(device);

    torch::Tensor z_points {};
    torch::Tensor mu {};
    torch::Tensor log_var {};
    std::tie(z_points, mu, log_var) = model->predict(data);        
    //std::cout << z_points.size(0) << std::endl;
    auto factors = torch::arange(0, 1, 0.1);
    //std::cout << range << std::endl;
    
    const auto& start_z_point = z_points[0];
    const auto& end_z_point = z_points[1];
    //std::cout << start_z.sizes() << std::endl;
    torch::Tensor changed_images {torch::empty({factors.size(0), 3, 128, 128}, torch::kFloat)};
    for (auto i = 0; i < factors.size(0); ++i)
    {
        const auto& factor = factors[i].item<float>();
        auto changed_z_point = (1.0 - factor) * start_z_point + factor * end_z_point;
        changed_z_point = torch::unsqueeze(changed_z_point, 0); 
        const auto changed_image = model->get_decoder()->forward(changed_z_point.to(device));
        changed_images[i] = torch::squeeze(changed_image);
    }
    return {data, changed_images};
}

#endif // MORPH_FACES 
