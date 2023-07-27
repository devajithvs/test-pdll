cmake_minimum_required(VERSION 3.22)
project(testPDLL)

set(LLVM_DIR "/home/devajithvs/work/llvm-project/build")
set(MLIR_DIR "/home/devajithvs/work/llvm-project/build")

set(INPUT_PDLL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/testPDLL.pdll")
set(OUTPUT_INC_FILE "testPDLL.h.inc")

add_custom_command(
  OUTPUT ${OUTPUT_INC_FILE}
  COMMAND mlir-pdll ${INPUT_PDLL_FILE} -x cpp -o ${OUTPUT_INC_FILE}
  DEPENDS ${INPUT_PDLL_FILE}
  COMMENT "Generating ${OUTPUT_INC_FILE}"
)

find_package(LLVM 17.0.0 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Add a custom target that depends on the custom command
add_custom_target(generate_testPDLL_h_inc DEPENDS testPDLL.h.inc)

# Now build our tools
add_executable(testPDLL testPDLL.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
# Link against LLVM and MLIR libraries
target_link_libraries(testPDLL PRIVATE MLIRIR MLIRParser MLIRTransforms ${dialect_libs})

add_dependencies(testPDLL generate_testPDLL_h_inc)
target_include_directories(testPDLL PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
