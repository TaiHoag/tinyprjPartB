cmake_minimum_required(VERSION 3.10)
project(CPUPerformancePredictor)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Debug and Release configurations
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/DataPoint.cpp
    src/Matrix.cpp
    src/Dataset.cpp
    src/LinearRegression.cpp
    src/Evaluator.cpp
)

# Header files
set(HEADERS
    include/DataPoint.h
    include/Matrix.h
    include/Dataset.h
    include/LinearRegression.h
    include/Evaluator.h
)

# Create executable
add_executable(cpu_performance_predictor main.cpp ${SOURCES})

# Set output directory
set_target_properties(cpu_performance_predictor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create output directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Custom target for running the program
add_custom_target(run
    COMMAND ${CMAKE_BINARY_DIR}/bin/cpu_performance_predictor
    DEPENDS cpu_performance_predictor
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running CPU Performance Predictor"
)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
endif()
