cmake_minimum_required(VERSION 3.20)

project(SHODHAN VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SHODHAN_BUILD_TESTS "Build the unit tests" ON)
option(SHODHAN_ENABLE_ZLIB "Enable reading .gz input (requires zlib)" OFF)
option(SHODHAN_WERROR "Treat compiler warnings as errors" OFF)
option(SHODHAN_SANITIZE "Enable AddressSanitizer and UBSan (GCC/Clang only)" OFF)

# Applies the project warning level, optional -Werror and optional sanitizers.
function(shodhan_configure_target target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
    if(SHODHAN_WERROR)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(SHODHAN_WERROR)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
    if(SHODHAN_SANITIZE)
      target_compile_options(${target} PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
  endif()
endfunction()

set(SHODHAN_CORE_SOURCES
  src/util/status.cpp
)
file(GLOB SHODHAN_OPTIONAL_CORE_SOURCES CONFIGURE_DEPENDS
  src/util/*.cpp src/model/*.cpp src/io/*.cpp src/scaling/*.cpp src/presolve/*.cpp)
list(APPEND SHODHAN_CORE_SOURCES ${SHODHAN_OPTIONAL_CORE_SOURCES})
list(REMOVE_DUPLICATES SHODHAN_CORE_SOURCES)

add_library(shodhan_core STATIC ${SHODHAN_CORE_SOURCES})
target_include_directories(shodhan_core PUBLIC include)
shodhan_configure_target(shodhan_core)

if(SHODHAN_ENABLE_ZLIB)
  find_package(ZLIB REQUIRED)
  target_link_libraries(shodhan_core PUBLIC ZLIB::ZLIB)
  target_compile_definitions(shodhan_core PUBLIC SHODHAN_HAVE_ZLIB=1)
endif()

file(GLOB SHODHAN_CLI_SOURCES CONFIGURE_DEPENDS src/cli/*.cpp)
if(SHODHAN_CLI_SOURCES)
  add_executable(shodhan ${SHODHAN_CLI_SOURCES})
  target_link_libraries(shodhan PRIVATE shodhan_core)
  shodhan_configure_target(shodhan)
endif()

if(SHODHAN_BUILD_TESTS)
  enable_testing()
  # tests/support holds test-only code (reference solver, generators); it is
  # compiled into the test executable only, never into shodhan_core.
  file(GLOB SHODHAN_TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp tests/support/*.cpp)
  add_executable(shodhan_tests ${SHODHAN_TEST_SOURCES})
  target_link_libraries(shodhan_tests PRIVATE shodhan_core)
  target_include_directories(shodhan_tests PRIVATE tests src)
  shodhan_configure_target(shodhan_tests)
  add_test(NAME shodhan_tests COMMAND shodhan_tests)

  if(SHODHAN_CLI_SOURCES)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      add_test(NAME shodhan_cli
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_check.py
                $<TARGET_FILE:shodhan> ${CMAKE_CURRENT_SOURCE_DIR}/tests/models)
    else()
      message(STATUS "Python3 not found: CLI end-to-end test disabled")
    endif()
  endif()
endif()
