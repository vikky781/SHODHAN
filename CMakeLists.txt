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
option(SHODHAN_BUILD_STRESS "Build the opt-in LP stress harness (tests/stress; not part of ctest)" OFF)
option(SHODHAN_SANITIZE "Enable AddressSanitizer and UBSan (GCC/Clang only)" OFF)

# Applies the project warning level, optional -Werror and optional sanitizers.
function(shodhan_configure_target target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
    # getenv and similar are used by the tests; MSVC calls them unsafe (C4996, an error under /WX).
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    if(SHODHAN_WERROR)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    # No fused multiply-add contraction: arm64 compilers fuse a*b+c by default, which changes the rounding of
    # the simplex arithmetic (and so the path of some numerically delicate LPs) compared with x86-64. With
    # contraction off every platform performs the same IEEE operations.
    target_compile_options(${target} PRIVATE -ffp-contract=off)
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
  src/util/*.cpp src/model/*.cpp src/io/*.cpp src/scaling/*.cpp src/presolve/*.cpp src/linalg/*.cpp src/lp/*.cpp src/cert/*.cpp src/mip/*.cpp src/ipm/*.cpp src/structure/*.cpp src/pooling/*.cpp)
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

  # Corpus generator for the KASAUTI integration test (writes models and certificates into a directory).
  add_executable(shodhan_cert_corpus tests/corpus/cert_corpus.cpp tests/support/lp_families.cpp tests/support/random_lp.cpp
                 tests/support/simplex_lps.cpp tests/support/mip_families.cpp tests/support/qp_families.cpp)
  target_link_libraries(shodhan_cert_corpus PRIVATE shodhan_core)
  target_include_directories(shodhan_cert_corpus PRIVATE tests src)
  shodhan_configure_target(shodhan_cert_corpus)

  if(SHODHAN_CLI_SOURCES)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      add_test(NAME shodhan_cli
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_check.py
                $<TARGET_FILE:shodhan> ${CMAKE_CURRENT_SOURCE_DIR}/tests/models)
    else()
      message(STATUS "Python3 not found: CLI end-to-end test disabled")
    endif()

    # KASAUTI, the independent verifier (verify/, standard-library Python): its unit tests, the differential
    # parser test, and the integration test that solves a seeded corpus, writes certificates and verifies
    # every one in exact AND float mode, followed by the mutation tests on the same corpus.
    if(Python3_Interpreter_FOUND)
      set(KASAUTI_CORPUS_DIR ${CMAKE_CURRENT_BINARY_DIR}/kasauti_corpus)
      set(KASAUTI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/verify)
      add_test(NAME kasauti_corpus_setup
        COMMAND ${CMAKE_COMMAND} -E make_directory ${KASAUTI_CORPUS_DIR})
      add_test(NAME kasauti_corpus_generate COMMAND shodhan_cert_corpus ${KASAUTI_CORPUS_DIR} 60)
      set_tests_properties(kasauti_corpus_setup PROPERTIES FIXTURES_SETUP kasauti_dir)
      set_tests_properties(kasauti_corpus_generate PROPERTIES FIXTURES_REQUIRED kasauti_dir FIXTURES_SETUP kasauti_corpus)
      add_test(NAME kasauti_unittests
        COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -t .
        WORKING_DIRECTORY ${KASAUTI_DIR})
      set_tests_properties(kasauti_unittests PROPERTIES ENVIRONMENT "SHODHAN_EXE=$<TARGET_FILE:shodhan>")
      add_test(NAME kasauti_corpus_check
        COMMAND ${Python3_EXECUTABLE} ${KASAUTI_DIR}/tests/corpus_check.py ${KASAUTI_CORPUS_DIR})
      set_tests_properties(kasauti_corpus_check PROPERTIES FIXTURES_REQUIRED kasauti_corpus)
      add_test(NAME kasauti_mutations
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_mutations tests.test_qp_mutations
        WORKING_DIRECTORY ${KASAUTI_DIR})
      set_tests_properties(kasauti_mutations PROPERTIES FIXTURES_REQUIRED kasauti_corpus
                           ENVIRONMENT "KASAUTI_CORPUS=${KASAUTI_CORPUS_DIR}")
      # Synthetic refinery / process case-study models (bench/refinery, standard-library Python): the generators and their
      # tests, then the C++ reader, round trip and brute-force agreement on the tiny models.
      set(REFINERY_DIR ${CMAKE_CURRENT_BINARY_DIR}/refinery_models)
      add_test(NAME refinery_generate
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bench/refinery/generate.py --suite ${REFINERY_DIR} --sizes tiny,small)
      set_tests_properties(refinery_generate PROPERTIES FIXTURES_SETUP refinery_models)
      add_test(NAME refinery_unittests
        COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -t .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bench/refinery)
      add_test(NAME refinery_cpp COMMAND shodhan_tests refinery)
      set_tests_properties(refinery_cpp PROPERTIES FIXTURES_REQUIRED refinery_models ENVIRONMENT "SHODHAN_REFINERY_DIR=${REFINERY_DIR}")
    else()
      # A visible skip, not a silent pass.
      add_test(NAME kasauti_skipped COMMAND ${CMAKE_COMMAND} -E echo "SKIPPED: Python 3 not found, KASAUTI tests not run")
      set_tests_properties(kasauti_skipped PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED")
    endif()
  endif()
endif()

if(SHODHAN_BUILD_STRESS)
  # Opt-in stress harness: reuses the test support code, never part of the default ctest run.
  file(GLOB SHODHAN_STRESS_SUPPORT CONFIGURE_DEPENDS tests/support/*.cpp)
  add_executable(shodhan_stress tests/stress/lp_stress.cpp ${SHODHAN_STRESS_SUPPORT})
  target_link_libraries(shodhan_stress PRIVATE shodhan_core)
  target_include_directories(shodhan_stress PRIVATE tests src)
  shodhan_configure_target(shodhan_stress)
endif()
