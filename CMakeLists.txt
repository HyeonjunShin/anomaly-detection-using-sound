cmake_minimum_required(VERSION 4.2.0)
project(serial_monitoring VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -O0")
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/thirdparty/")

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install") 
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include(CPack)
set(CPACK_PACKAGE_NAME "serial-monitoring")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_GENERATOR "TGZ;ZIP")        
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "hyeonnnjunnn@gmail.com")

include(FetchContent)
FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json
        GIT_PROGRESS TRUE
        GIT_SHALLOW TRUE
        GIT_TAG v3.12.0)

FetchContent_MakeAvailable(json)

find_package(Boost REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
# find_package(CUDA REQUIRED)

set(Torch_DIR "/home/vscode/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories(
  ${TENSORRT_DIR}/include
  /usr/local/cuda-13.0/include
)

link_directories(
  ${TENSORRT_DIR}/lib
  /usr/local/cuda-13.0/lib64
)


add_executable(main 
  src/main.cpp
)
target_link_libraries(main PRIVATE
    nvinfer 
    nvinfer_plugin 
    nvonnxparser 
    cudart
)

add_executable(main_libtorch src/main_libtorch.cpp)
target_link_libraries(main_libtorch PRIVATE 
  "${TORCH_LIBRARIES}"
  nlohmann_json::nlohmann_json
)

if(Boost_FOUND)
  message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
  target_include_directories(main PRIVATE ${Boost_INCLUDE_DIRS})
  # gui 타겟에도 필요하다면 추가
  # target_include_directories(gui PRIVATE ${Boost_INCLUDE_DIRS})
endif()



install(TARGETS main
  RUNTIME DESTINATION bin
)


install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ 
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

install(FILES "${CMAKE_SOURCE_DIR}/README.md"
  DESTINATION share/${PROJECT_NAME}
)
