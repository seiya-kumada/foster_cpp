#ifndef DATASET_READER
#define DATASET_READER

#include <opencv2/core.hpp>
#include <fstream>
#include <utility>

class DatasetReader
{
public:
    DatasetReader(std::ifstream& ifs);
    DatasetReader(const DatasetReader&) = delete;
    DatasetReader& operator=(const DatasetReader&) = delete;

    std::pair<cv::Mat, int> load_one_image(int index) const;

private:
    std::ifstream& ifs_;
};
#endif // DATASET_READER 
