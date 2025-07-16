# CMake Requirement
cmake_minimum_required(VERSION 3.15)

# C++ requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the build type to Debug if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# C++ flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Setup project
project(AnalyticalNetwork)

# Compile external libraries
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/yaml-cpp yaml-cpp)

# Include src files to compile
file(GLOB srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/event-queue/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/network/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/topology/*.cpp
)

# Compile network backend
add_executable(AnalyticalNetwork ${srcs})
target_sources(AnalyticalNetwork PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

# Properties
set_target_properties(AnalyticalNetwork
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
)
set_target_properties(AnalyticalNetwork PROPERTIES COMPILE_WARNING_AS_ERROR ON)

# Link libraries
target_link_libraries(AnalyticalNetwork PUBLIC yaml-cpp)

# Include directories
target_include_directories(AnalyticalNetwork PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/astra-network-analytical/)
target_include_directories(AnalyticalNetwork PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/)
