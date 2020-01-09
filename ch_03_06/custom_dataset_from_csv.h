#ifndef CUSTOM_DATASET_FROM_CSV
#define CUSTOM_DATASET_FROM_CSV
#include <torch/torch.h>
#include <vector>
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <string>
#include "csv.h"

struct FilenameInfo
{
    std::string image_id_;
    std::string flag_;
    FilenameInfo(const std::string& image_id, const std::string& flag)
        : image_id_{image_id}
        , flag_{flag} {}
};


class CustomDatasetFromCSV : public torch::data::Dataset<CustomDatasetFromCSV> {
private:
    boost::filesystem::path dir_path_;    
    std::vector<FilenameInfo> filename_infos_;

public:
    CustomDatasetFromCSV(
        const std::string& dir_path, io::CSVReader<2>& csv, const std::vector<int>& input_size, const std::string& label);
	
    // Override get() function to return tensor at location index
	torch::data::Example<> get(std::size_t index) override;

	// Return the length of data
	torch::optional<std::size_t> size() const override 
	{
	  return filename_infos_.size();
	};

    const std::vector<FilenameInfo>& get_filename_infos() const
    {
        return filename_infos_;
    }

private:
    cv::Size input_size_;
};
#endif // CUSTOM_DATASET_FROM_CSV
