cmake_minimum_required(VERSION 3.22)
project(testPDLL)

set(INPUT_PDLL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/testPDLL.pdll")
set(OUTPUT_INC_FILE "testPDLL.h.inc")

add_custom_command(
  OUTPUT ${OUTPUT_INC_FILE}
  COMMAND mlir-pdll ${INPUT_PDLL_FILE} -x cpp -o ${OUTPUT_INC_FILE}
  DEPENDS ${INPUT_PDLL_FILE}
  COMMENT "Generating ${OUTPUT_INC_FILE}"
)

set(LLVM_DIR "/home/devajithvs/work/llvm-project/build")

find_package(LLVM 17.0.0 REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set the MLIR_DIR to the installed MLIR build directory
set(MLIR_DIR "/home/devajithvs/work/llvm-project/build")

# Find the MLIR package
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MLIR_MAIN_INCLUDE_DIR})

add_definitions(${LLVM_DEFINITIONS})

# Add a custom target that depends on the custom command
add_custom_target(generate_testPDLL_h_inc DEPENDS testPDLL.h.inc)



# Now build our tools
add_executable(testPDLL testPDLL.cpp)

# Link against LLVM and MLIR libraries
target_link_libraries(testPDLL PRIVATE MLIRIR MLIRParser MLIRTransforms)

# Add the build directory to the include directories for the current target (testPDLL)
target_include_directories(testPDLL PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(testPDLL generate_testPDLL_h_inc)