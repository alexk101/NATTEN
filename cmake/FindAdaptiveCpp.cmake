# Find AdaptiveCpp installation
#
# Sets the following variables:
# ACPP_FOUND - True if AdaptiveCpp was found
# ACPP_INCLUDE_DIRS - AdaptiveCpp include directories
# ACPP_LIBRARIES - AdaptiveCpp libraries
# ACPP_VERSION - AdaptiveCpp version

# First check the submodule location
set(ACPP_SUBMODULE_PATH "${CMAKE_SOURCE_DIR}/../third_party/AdaptiveCpp")

find_path(ACPP_INCLUDE_DIR
    NAMES sycl/sycl.hpp
    PATHS
        ${ACPP_SUBMODULE_PATH}/include
        ${ACPP_HOME}/include
        /opt/adaptivecpp/include
        /usr/local/adaptivecpp/include
    NO_DEFAULT_PATH
)

find_library(ACPP_LIBRARY
    NAMES sycl
    PATHS
        ${ACPP_SUBMODULE_PATH}/lib
        ${ACPP_HOME}/lib
        /opt/adaptivecpp/lib
        /usr/local/adaptivecpp/lib
    NO_DEFAULT_PATH
)

# If not found in submodule or specified paths, try system paths
if(NOT ACPP_INCLUDE_DIR OR NOT ACPP_LIBRARY)
    find_path(ACPP_INCLUDE_DIR NAMES sycl/sycl.hpp)
    find_library(ACPP_LIBRARY NAMES sycl)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AdaptiveCpp
    REQUIRED_VARS 
        ACPP_INCLUDE_DIR
        ACPP_LIBRARY
)

if(AdaptiveCpp_FOUND)
    set(ACPP_INCLUDE_DIRS ${ACPP_INCLUDE_DIR})
    set(ACPP_LIBRARIES ${ACPP_LIBRARY})
endif() 