#ifndef CUSTOM_DATASET
#define CUSTOM_DATASET
#include <torch/torch.h>
#include <vector>
#include <fstream>

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
	// Declare 2 vectors of tensors for images and labels
	std::vector<torch::Tensor> images_;
	std::vector<torch::Tensor> labels_;

public:
	// Constructor
	CustomDataset(std::vector<std::ifstream>& ifs);

	// Override get() function to return tensor at location index
	torch::data::Example<> get(std::size_t index) override 
	{
	  torch::Tensor sample_img = images_.at(index);
	  torch::Tensor sample_label = labels_.at(index);
	  return {sample_img.clone(), sample_label.clone()};
	};

	// Return the length of data
	torch::optional<std::size_t> size() const override 
	{
	  return labels_.size();
	};
};
#endif // CUSTOM_DATASET
