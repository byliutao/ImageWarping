cmake_minimum_required(VERSION 3.11-3.18)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} src)
project(ImageWarping)

find_package( OpenCV REQUIRED )
add_executable(ImageWarping ${src})

target_link_libraries(ImageWarping ${OpenCV_LIBS})
target_include_directories(ImageWarping PUBLIC
        "${OpenCV_INCLUDE_DIRS}"
        )