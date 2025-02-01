# Find AdaptiveCpp installation
#
# Sets the following variables:
# ACPP_FOUND - True if AdaptiveCpp was found
# ACPP_INCLUDE_DIRS - AdaptiveCpp include directories
# ACPP_LIBRARIES - AdaptiveCpp libraries
# ACPP_VERSION - AdaptiveCpp version

message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")

# Allow user to specify ROCm path, with a default for standard installations
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH $ENV{ROCM_PATH})
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()
message(STATUS "Using ROCM_PATH: ${ROCM_PATH}")

# Get the correct path to the submodule
get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
set(ACPP_SUBMODULE_PATH "${PROJECT_ROOT}/third_party/AdaptiveCpp")
message(STATUS "Project root: ${PROJECT_ROOT}")
message(STATUS "Checking submodule path: ${ACPP_SUBMODULE_PATH}")

if(EXISTS "${ACPP_SUBMODULE_PATH}")
    message(STATUS "Submodule exists")
else()
    message(STATUS "Submodule not found")
endif()

if(EXISTS "${ACPP_SUBMODULE_PATH}/build")
    message(STATUS "Build directory exists")
else()
    message(STATUS "Build directory not found")
endif()

# Function to check if AdaptiveCpp is properly built
function(is_acpp_built RESULT_VAR)
    set(REQUIRED_FILES
        "${ACPP_SUBMODULE_PATH}/build/install/lib/cmake/AdaptiveCpp/adaptivecpp-config.cmake"
        "${ACPP_SUBMODULE_PATH}/build/install/include/AdaptiveCpp/sycl/sycl.hpp"
        "${ACPP_SUBMODULE_PATH}/build/install/lib/libsycl${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )
    
    set(ALL_FILES_EXIST TRUE)
    foreach(FILE ${REQUIRED_FILES})
        if(NOT EXISTS "${FILE}")
            set(ALL_FILES_EXIST FALSE)
            message(STATUS "Missing required file: ${FILE}")
            break()
        endif()
    endforeach()
    
    set(${RESULT_VAR} ${ALL_FILES_EXIST} PARENT_SCOPE)
endfunction()

# Check if AdaptiveCpp is properly built
is_acpp_built(ACPP_BUILT)

# Build if necessary
if(EXISTS "${ACPP_SUBMODULE_PATH}" AND NOT ${ACPP_BUILT})
    # Remove existing build directory if it exists
    if(EXISTS "${ACPP_SUBMODULE_PATH}/build")
        message(STATUS "Removing existing build directory")
        file(REMOVE_RECURSE "${ACPP_SUBMODULE_PATH}/build")
    endif()
    
    message(STATUS "Building AdaptiveCpp from submodule using ROCm at: ${ROCM_PATH}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -B build 
            -DWITH_ROCM_BACKEND=ON 
            -DROCM_PATH=${ROCM_PATH}
            -DCMAKE_INSTALL_PREFIX=${ACPP_SUBMODULE_PATH}/build/install
            -DADAPTIVECPP_INSTALL_CMAKE_DIR=lib/cmake/AdaptiveCpp
            -DBOOST_ROOT=$ENV{BOOST_ROOT}
        WORKING_DIRECTORY ${ACPP_SUBMODULE_PATH}
        RESULT_VARIABLE BUILD_CONFIG_RESULT
        OUTPUT_VARIABLE BUILD_CONFIG_OUTPUT
        ERROR_VARIABLE BUILD_CONFIG_ERROR
    )
    message(STATUS "Build configuration result: ${BUILD_CONFIG_RESULT}")
    message(STATUS "Build configuration output: ${BUILD_CONFIG_OUTPUT}")
    message(STATUS "Build configuration error: ${BUILD_CONFIG_ERROR}")
    
    if(BUILD_CONFIG_RESULT EQUAL 0)
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build build --target install
            WORKING_DIRECTORY ${ACPP_SUBMODULE_PATH}
            RESULT_VARIABLE BUILD_RESULT
            OUTPUT_VARIABLE BUILD_OUTPUT
            ERROR_VARIABLE BUILD_ERROR
        )
        message(STATUS "Build result: ${BUILD_RESULT}")
        message(STATUS "Build output: ${BUILD_OUTPUT}")
        message(STATUS "Build error: ${BUILD_ERROR}")
    endif()
endif()

# Check if config files were generated
set(CONFIG_FILE "${ACPP_SUBMODULE_PATH}/build/install/lib/cmake/AdaptiveCpp/adaptivecpp-config.cmake")
message(STATUS "Looking for config file at: ${CONFIG_FILE}")
if(EXISTS "${CONFIG_FILE}")
    message(STATUS "Config file exists")
else()
    message(STATUS "Config file not found")
endif()

# Now look for the installed AdaptiveCpp
find_path(ACPP_INCLUDE_DIR
    NAMES sycl/sycl.hpp
    PATHS
        ${ACPP_SUBMODULE_PATH}/build/install/include/AdaptiveCpp
        ${ACPP_HOME}/include
    NO_DEFAULT_PATH
)

find_library(ACPP_LIBRARY
    NAMES sycl
    PATHS
        ${ACPP_SUBMODULE_PATH}/build/install/lib
        ${ACPP_HOME}/lib
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AdaptiveCpp
    REQUIRED_VARS 
        ACPP_INCLUDE_DIR
        ACPP_LIBRARY
)

if(AdaptiveCpp_FOUND)
    set(ACPP_INCLUDE_DIRS ${ACPP_INCLUDE_DIR})
    set(ACPP_LIBRARIES ${ACPP_LIBRARY})
    set(AdaptiveCpp_DIR ${ACPP_SUBMODULE_PATH}/build/install/lib/cmake/AdaptiveCpp)
endif() 