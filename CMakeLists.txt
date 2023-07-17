cmake_minimum_required(VERSION 3.11-3.18)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} src)
project(ImageWarping)

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)


add_executable(ImageWarping ${src})

target_link_libraries(ImageWarping ${OpenCV_LIBS}  ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} GLEW::GLEW)
target_include_directories(ImageWarping PUBLIC
        "${OpenCV_INCLUDE_DIRS}"
        "${EIGEN3_INCLUDE_DIRS}"
        "${GLUT_INCLUDE_DIRS}"
        "${OPENGL_INCLUDE_DIRS}"
        )