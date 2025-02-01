# Find AdaptiveCpp installation
#
# Sets the following variables:
# ACPP_FOUND - True if AdaptiveCpp was found
# ACPP_INCLUDE_DIRS - AdaptiveCpp include directories
# ACPP_LIBRARIES - AdaptiveCpp libraries
# ACPP_VERSION - AdaptiveCpp version

# First build and install AdaptiveCpp from submodule if not already done
set(ACPP_SUBMODULE_PATH "${CMAKE_SOURCE_DIR}/third_party/AdaptiveCpp")
if(EXISTS "${ACPP_SUBMODULE_PATH}" AND NOT EXISTS "${ACPP_SUBMODULE_PATH}/build")
    message(STATUS "Building AdaptiveCpp from submodule...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -B build -DWITH_ROCM_BACKEND=ON -DROCM_PATH=/opt/rocm
        WORKING_DIRECTORY ${ACPP_SUBMODULE_PATH}
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build build --target install
        WORKING_DIRECTORY ${ACPP_SUBMODULE_PATH}
    )
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