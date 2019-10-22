cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(PROJ_NAME ch_02_01)
add_executable(${PROJ_NAME} 
    ./main.cpp 
    ./custom_dataset.cpp
    ./dataset_reader.cpp
    ./architecture.cpp)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework filesystem)

target_compile_definitions(
    ${PROJ_NAME} PRIVATE 
    UNIT_TEST=1
    UNIT_TEST_DatasetReader=1
    UNIT_TEST_CustomDataset=1
    UNIT_TEST_Architecture=1
)

set(INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ./
)

set(LIBRARIES
    ${OpenCV_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${Boost_LIBRARIES}
)

include_directories(${INCLUDE_DIRS})
target_link_libraries (${PROJ_NAME} ${LIBRARIES})

set_property(TARGET ch_02_01 PROPERTY CXX_STANDARD 14)
message(STATUS ${Boost_INCLUDE_DIRS})
message(STATUS ${Boost_LIBRARIES})
message(STATUS ${Boost_LIBRARY_DIRS})

