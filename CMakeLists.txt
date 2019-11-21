cmake_minimum_required(VERSION 3.5)

project(POCKET_MCTS)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The general idea is to use the default values and overwrite them only for
# specific, well experimented systems.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU"
    OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")

  set(NO_WARN_FLAGS "-w")

  # `-Wconversion`, `-Wsign-conversion`, `-Weffc++` are other interesting
  # switches to try from time to time (they give many false positives).
  set(WARN_FLAGS "-Wpedantic" "-Wall" "-Wextra" "-Winvalid-pch" "-Wpedantic"
                 "-Wformat=2" "-Wfloat-equal" "-Wshadow" "-Wdouble-promotion"
                 "-Wzero-as-null-pointer-constant")

  set(OTHER_FLAGS "-pipe" "-march=native")

  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DBOOST_DISABLE_ASSERTS")

  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-s")

elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")

  set(WARN_FLAGS "/W4")

endif()

add_compile_options(${OTHER_FLAGS} ${WARN_FLAGS})

# Compiler must support the C++14 standard.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${POCKET_MCTS_SOURCE_DIR})

file(GLOB EXAMPLES_SRC examples/*.cc)
foreach (example_src ${EXAMPLES_SRC})
  # Gets the filename (`example`) without directory / longest extension
  # (`NAME_WE`) from the full filename (${example_src})
  get_filename_component(example ${example_src} NAME_WE)

  add_executable(${example} ${example_src})
endforeach()
