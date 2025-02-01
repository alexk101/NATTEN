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

if(EXISTS "${ACPP_SUBMODULE_PATH}" AND NOT EXISTS "${ACPP_SUBMODULE_PATH}/build")
    message(STATUS "Building AdaptiveCpp from submodule using ROCm at: ${ROCM_PATH}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -B build 
            -DWITH_ROCM_BACKEND=ON 
            -DROCM_PATH=${ROCM_PATH}
            -DCMAKE_INSTALL_PREFIX=${ACPP_SUBMODULE_PATH}/build/install
            -DADAPTIVECPP_INSTALL_CMAKE_DIR=lib/cmake/AdaptiveCpp
        WORKING_DIRECTORY ${ACPP_SUBMODULE_PATH}
        RESULT_VARIABLE BUILD_CONFIG_RESULT
    )
    message(STATUS "Build configuration result: ${BUILD_CONFIG_RESULT}")
    
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build build --target install
        WORKING_DIRECTORY ${ACPP_SUBMODULE_PATH}
        RESULT_VARIABLE BUILD_RESULT
    )
    message(STATUS "Build result: ${BUILD_RESULT}")
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