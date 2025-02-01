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

# First try to find an installed version
find_package(AdaptiveCpp QUIET CONFIG)

if(NOT AdaptiveCpp_FOUND)
    # Get the submodule path
    get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
    set(ACPP_SUBMODULE_PATH "${PROJECT_ROOT}/third_party/AdaptiveCpp")
    
    if(EXISTS "${ACPP_SUBMODULE_PATH}/CMakeLists.txt")
        message(STATUS "Building AdaptiveCpp from submodule")
        execute_process(
            COMMAND ${CMAKE_COMMAND} 
                -S "${ACPP_SUBMODULE_PATH}"
                -B "${ACPP_SUBMODULE_PATH}/build"
                -DWITH_ROCM_BACKEND=ON
                -DROCM_PATH=${ROCM_PATH}
                -DCMAKE_INSTALL_PREFIX=${ACPP_SUBMODULE_PATH}/build/install
                -DBOOST_ROOT=$ENV{BOOST_ROOT}
                -DBOOST_INCLUDEDIR=$ENV{BOOST_ROOT}/include
                -DBOOST_LIBRARYDIR=$ENV{BOOST_ROOT}/lib
                -DCMAKE_POLICY_DEFAULT_CMP0144=NEW
                -DBoost_USE_STATIC_LIBS=OFF
                -DBoost_USE_MULTITHREADED=ON
                -DCMAKE_CXX_COMPILER=CC
                -DCMAKE_C_COMPILER=cc
            RESULT_VARIABLE BUILD_RESULT
            OUTPUT_VARIABLE BUILD_OUTPUT
            ERROR_VARIABLE BUILD_ERROR
        )
        
        message(STATUS "Build output: ${BUILD_OUTPUT}")
        message(STATUS "Build error: ${BUILD_ERROR}")
        
        if(BUILD_RESULT EQUAL 0)
            find_package(AdaptiveCpp REQUIRED CONFIG 
                PATHS "${ACPP_SUBMODULE_PATH}/build"
                NO_DEFAULT_PATH
            )
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AdaptiveCpp
    REQUIRED_VARS 
        AdaptiveCpp_DIR
)

if(AdaptiveCpp_FOUND)
    set(ACPP_INCLUDE_DIRS ${AdaptiveCpp_INCLUDE_DIR})
    set(ACPP_LIBRARIES ${AdaptiveCpp_LIBRARY})
    set(AdaptiveCpp_DIR ${AdaptiveCpp_DIR})
endif() 