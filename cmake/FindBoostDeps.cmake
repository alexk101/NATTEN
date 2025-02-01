# First try to find system Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(BOOST COMPONENTS context QUIET)

if(NOT Boost_FOUND)
    message(STATUS "System Boost not found, building from submodule")
    
    # Get the submodule path
    get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
    set(BOOST_SUBMODULE_PATH "${PROJECT_ROOT}/third_party/boost")
    
    if(NOT EXISTS "${BOOST_SUBMODULE_PATH}/bootstrap.sh")
        message(FATAL_ERROR "Boost submodule not found. Please run 'git submodule update --init --recursive'")
    endif()

    # Build boost
    execute_process(
        COMMAND ./bootstrap.sh --with-libraries=context
        WORKING_DIRECTORY ${BOOST_SUBMODULE_PATH}
        RESULT_VARIABLE BOOST_BOOTSTRAP_RESULT
        OUTPUT_VARIABLE BOOST_BOOTSTRAP_OUTPUT
        ERROR_VARIABLE BOOST_BOOTSTRAP_ERROR
    )

    if(NOT BOOST_BOOTSTRAP_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to bootstrap Boost:\n${BOOST_BOOTSTRAP_ERROR}")
    endif()

    execute_process(
        COMMAND ./b2 install 
            --prefix=${BOOST_SUBMODULE_PATH}/install
            --with-context
            link=shared
            threading=multi
            variant=release
        WORKING_DIRECTORY ${BOOST_SUBMODULE_PATH}
        RESULT_VARIABLE BOOST_BUILD_RESULT
        OUTPUT_VARIABLE BOOST_BUILD_OUTPUT
        ERROR_VARIABLE BOOST_BUILD_ERROR
    )

    if(NOT BOOST_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to build Boost:\n${BOOST_BUILD_ERROR}")
    endif()

    # Set Boost variables to point to our built version
    set(BOOST_ROOT ${BOOST_SUBMODULE_PATH}/install)
    set(Boost_INCLUDE_DIRS ${BOOST_ROOT}/include)
    set(Boost_LIBRARY_DIRS ${BOOST_ROOT}/lib)
    set(Boost_CONTEXT_LIBRARY ${BOOST_ROOT}/lib/libboost_context${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(Boost_VERSION "1.85.0")
    set(Boost_FOUND TRUE)
endif()

message(STATUS "Using Boost:")
message(STATUS "  Version: ${Boost_VERSION}")
message(STATUS "  Boost root: ${BOOST_ROOT}")
message(STATUS "  Include dirs: ${Boost_INCLUDE_DIRS}")
message(STATUS "  Library dirs: ${Boost_LIBRARY_DIRS}")
message(STATUS "  Context library: ${Boost_CONTEXT_LIBRARY}")

# Set variables for parent scope
set(BOOST_CMAKE_ARGS
    -DBoost_DIR=${Boost_DIR}
    -DBoost_INCLUDE_DIR=${Boost_INCLUDE_DIRS}
    -DBoost_LIBRARY_DIR=${Boost_LIBRARY_DIRS}
    -DBOOST_ROOT=${BOOST_ROOT}
)