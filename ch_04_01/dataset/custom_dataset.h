#ifndef CUSTOM_DATASET
#define CUSTOM_DATASET
#include <torch/torch.h>
#include <vector>
#include <boost/filesystem.hpp>
//#include <opencv2/core.hpp>
#include <string>

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    std::vector<std::uint64_t>  shape_;
    std::vector<std::uint8_t>   data_;
 
/*
 * 1. npyファイルを読む。
 * 2. batch_size枚の画像(rows, cols)を作る。
 * 3. torch::Tensorに変換する。ラベル部分は空テンソル。
 */
public:
    CustomDataset(const std::string& path, int upper_size);
	
    // Override get() function to return tensor at location index
	torch::data::Example<> get(std::size_t index) override;

	// Return the length of data
	torch::optional<std::size_t> size() const override 
	{
	  return shape_[0];
	};
};
#endif // CUSTOM_DATASET
