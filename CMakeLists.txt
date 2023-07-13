cmake_minimum_required(VERSION 3.11-3.18)

project(conformalResizing)

find_package( OpenCV REQUIRED )
add_executable(conformalResizing main.cpp)

target_link_libraries(conformalResizing ${OpenCV_LIBS})
target_include_directories(conformalResizing PUBLIC
        "${OpenCV_INCLUDE_DIRS}"
        )