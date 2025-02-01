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
message(STATUS "Initial AdaptiveCpp find_package result: ${AdaptiveCpp_FOUND}")

if(NOT AdaptiveCpp_FOUND)
    # Get the submodule path
    get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
    set(ACPP_SUBMODULE_PATH "${PROJECT_ROOT}/third_party/AdaptiveCpp")
    message(STATUS "ACPP_SUBMODULE_PATH: ${ACPP_SUBMODULE_PATH}")
    
    if(EXISTS "${ACPP_SUBMODULE_PATH}/CMakeLists.txt")
        message(STATUS "Building AdaptiveCpp from submodule")
        
        # Try to find Boost first
        find_package(Boost COMPONENTS context fiber QUIET)

        if(NOT Boost_FOUND)
            build_boost_from_submodule()
            find_package(Boost COMPONENTS context fiber REQUIRED)
        endif()

        message(STATUS "Using Boost:")
        message(STATUS "  Version: ${Boost_VERSION}")
        message(STATUS "  Include dirs: ${Boost_INCLUDE_DIRS}")
        message(STATUS "  Library dirs: ${Boost_LIBRARY_DIRS}")
        
        execute_process(
            COMMAND ${CMAKE_COMMAND} 
                -S "${ACPP_SUBMODULE_PATH}"
                -B "${ACPP_SUBMODULE_PATH}/build"
                -DWITH_ROCM_BACKEND=ON 
                -DROCM_PATH=${ROCM_PATH}
                -DCMAKE_INSTALL_PREFIX=${ACPP_SUBMODULE_PATH}/build/install
                -DCMAKE_CXX_COMPILER=CC
                -DCMAKE_C_COMPILER=cc
                -DBoost_DIR=${Boost_DIR}
            RESULT_VARIABLE BUILD_RESULT
            OUTPUT_VARIABLE BUILD_OUTPUT
            ERROR_VARIABLE BUILD_ERROR
        )
        
        message(STATUS "Initial build configuration:")
        message(STATUS "  Result: ${BUILD_RESULT}")
        message(STATUS "  Output: ${BUILD_OUTPUT}")
        message(STATUS "  Error: ${BUILD_ERROR}")
        
        if(BUILD_RESULT EQUAL 0)
            execute_process(
                COMMAND ${CMAKE_COMMAND} --build build --target install
                WORKING_DIRECTORY ${ACPP_SUBMODULE_PATH}
                RESULT_VARIABLE BUILD_RESULT
                OUTPUT_VARIABLE BUILD_OUTPUT
                ERROR_VARIABLE BUILD_ERROR
            )
            message(STATUS "Build and install:")
            message(STATUS "  Result: ${BUILD_RESULT}")
            message(STATUS "  Output: ${BUILD_OUTPUT}")
            message(STATUS "  Error: ${BUILD_ERROR}")
            
            if(BUILD_RESULT EQUAL 0)
                set(EXPECTED_CONFIG_PATH "${ACPP_SUBMODULE_PATH}/build/install/lib/cmake/AdaptiveCpp")
                message(STATUS "Looking for AdaptiveCpp config at: ${EXPECTED_CONFIG_PATH}")
                find_package(AdaptiveCpp REQUIRED CONFIG 
                    PATHS "${ACPP_SUBMODULE_PATH}/build/install/lib/cmake/AdaptiveCpp"
                    NO_DEFAULT_PATH
                )
            endif()
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AdaptiveCpp
    REQUIRED_VARS 
        AdaptiveCpp_DIR
)

if(AdaptiveCpp_FOUND)
    message(STATUS "AdaptiveCpp found:")
    message(STATUS "  AdaptiveCpp_DIR: ${AdaptiveCpp_DIR}")
    message(STATUS "  AdaptiveCpp_INCLUDE_DIR: ${AdaptiveCpp_INCLUDE_DIR}")
    message(STATUS "  AdaptiveCpp_LIBRARY: ${AdaptiveCpp_LIBRARY}")
    set(ACPP_INCLUDE_DIRS ${AdaptiveCpp_INCLUDE_DIR})
    set(ACPP_LIBRARIES ${AdaptiveCpp_LIBRARY})
    set(AdaptiveCpp_DIR ${AdaptiveCpp_DIR})
endif()

# Function to build Boost from submodule if needed
function(build_boost_from_submodule)
    set(BOOST_SUBMODULE_PATH "${PROJECT_ROOT}/third_party/boost")
    set(BOOST_BUILD_DIR "${BOOST_SUBMODULE_PATH}/__build__")
    
    message(STATUS "Building Boost from submodule at: ${BOOST_SUBMODULE_PATH}")
    
    execute_process(
        COMMAND ${CMAKE_COMMAND} 
            -S ${BOOST_SUBMODULE_PATH}
            -B ${BOOST_BUILD_DIR}
            -DBUILD_SHARED_LIBS=ON
            -DBOOST_INCLUDE_LIBRARIES=context,fiber
            -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/boost_install
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        RESULT_VARIABLE BOOST_CONFIG_RESULT
        OUTPUT_VARIABLE BOOST_CONFIG_OUTPUT
        ERROR_VARIABLE BOOST_CONFIG_ERROR
    )
    
    if(NOT BOOST_CONFIG_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to configure Boost: ${BOOST_CONFIG_ERROR}")
    endif()
    
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${BOOST_BUILD_DIR} --target install
        RESULT_VARIABLE BOOST_BUILD_RESULT
        OUTPUT_VARIABLE BOOST_BUILD_OUTPUT
        ERROR_VARIABLE BOOST_BUILD_ERROR
    )
    
    if(NOT BOOST_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to build Boost: ${BOOST_BUILD_ERROR}")
    endif()
    
    set(BOOST_ROOT ${PROJECT_BINARY_DIR}/boost_install PARENT_SCOPE)
endfunction() 