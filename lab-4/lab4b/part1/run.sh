#!/bin/sh
set -e

# find the absolute path to this script
PROJECT_DIR=$(dirname "$(realpath "$0")")
BUILD_DIR="$PROJECT_DIR/build"

# cleanup build directory
if [ -d "${BUILD_DIR:?}" ]; then
        rm -rf "${BUILD_DIR}"
    fi

# run CMake
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug
cmake --build "$BUILD_DIR" --parallel $(nproc)

# run analytical backend simulation
${BUILD_DIR}/bin/AnalyticalNetwork
