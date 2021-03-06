cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(PROJ_NAME ch_04_01)
add_executable(${PROJ_NAME} 
    ./main.cpp
    ./trainer.cpp
    ./models/gan.cpp
    ./dataset/custom_dataset.cpp
    #./add_attribute_vectors.cpp
    #../ch_03_05/custom_dataset.cpp
    #./custom_dataset_from_csv.cpp
    #../ch_03_03/variational_auto_encoder.cpp
    )

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(Boost 1.71.0 COMPONENTS filesystem program_options unit_test_framework REQUIRED)

target_compile_definitions(
    ${PROJ_NAME} PRIVATE 
    UNIT_TEST=1
    UNIT_TEST_GAN=1
    UNIT_TEST_CustomDataset=1
    UNIT_TEST_TRAINER=1
)

set(INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ./
    ./models
    ./dataset
    /usr/local/libnpy
    #/usr/local/cpplinq/CppLinq/
    #../ch_03_03/
    #../ch_03_05/
)

set(LIBRARIES
    ${TORCH_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
)


include_directories(${INCLUDE_DIRS})
target_link_libraries (${PROJ_NAME} ${LIBRARIES})

set_property(TARGET ${PROJ_NAME} PROPERTY CXX_STANDARD 14)
message(STATUS ${Boost_INCLUDE_DIRS})
message(STATUS ${Boost_LIBRARIES})
message(STATUS ${OpenCV_INCLUDE_DIRS})
message(STATUS ${OpenCV_LIBRARIES})
message(STATUS ${LIBRARIES})

