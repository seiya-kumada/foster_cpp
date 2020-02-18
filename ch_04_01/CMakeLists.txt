cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(PROJ_NAME ch_04_01)
add_executable(${PROJ_NAME} 
    ./main.cpp
    ./models/gan.cpp
    #./add_attribute_vectors.cpp
    #../ch_03_05/custom_dataset.cpp
    #./custom_dataset_from_csv.cpp
    #../ch_03_03/variational_auto_encoder.cpp
    )

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework filesystem program_options)

target_compile_definitions(
    ${PROJ_NAME} PRIVATE 
    #UNIT_TEST=0
    #UNIT_TEST_CustomDatasetFromCSV=0
    #UNIT_TEST_make_attribute_vectors=0
)

set(INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ./
    ./models
    #/usr/local/cpplinq/CppLinq/
    #../ch_03_03/
    #../ch_03_05/
)

set(LIBRARIES
    ${OpenCV_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${Boost_LIBRARIES}
)

include_directories(${INCLUDE_DIRS})
target_link_libraries (${PROJ_NAME} ${LIBRARIES})

set_property(TARGET ${PROJ_NAME} PROPERTY CXX_STANDARD 14)
message(STATUS ${Boost_INCLUDE_DIRS})
message(STATUS ${Boost_LIBRARIES})
message(STATUS ${Boost_LIBRARY_DIRS})

