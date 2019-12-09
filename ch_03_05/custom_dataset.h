#ifndef CUSTOM_DATASET
#define CUSTOM_DATASET
#include <torch/torch.h>
#include <vector>
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <string>

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    std::vector<boost::filesystem::path> paths_;
/*
 * 1. ディレクトリ内の画像名を全て読み込む。file_names__
 * 2. file_names_をシャッフルする。
 * 3. batch_size_分だけ画像を読み込む。
 */
public:
    CustomDataset(const std::string& dir_path, const std::vector<int>& input_size);
	
    // Override get() function to return tensor at location index
	torch::data::Example<> get(std::size_t index) override;

	// Return the length of data
	torch::optional<std::size_t> size() const override 
	{
	  return paths_.size();
	};

    const std::vector<boost::filesystem::path>& get_paths() const
    {
        return paths_;
    }

private:
    cv::Size input_size_;
};
#endif // CUSTOM_DATASET
