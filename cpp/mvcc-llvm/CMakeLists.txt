# mvcc-ir2msl (device LLVM IR to Metal Shading Language) and the mvcc-passes clang plugin.
cmake_minimum_required(VERSION 3.20)
project(mvcc-llvm C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(LLVM REQUIRED CONFIG)
if(LLVM_PACKAGE_VERSION VERSION_LESS 21)
  message(FATAL_ERROR "LLVM >= 21 required, found ${LLVM_PACKAGE_VERSION}")
endif()
message(STATUS "LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(LLVM_LIBS core support irreader bitreader bitwriter analysis passes transformutils scalaropts ipo instcombine vectorize)

add_executable(mvcc-ir2msl ir2msl.cpp msl_emitter.cpp msl_guarded.cpp ptx_asm.cpp tensor_recovery.cpp staging_elim.cpp symexec.cpp wide_vectors.cpp sig.cpp coords.cpp stir.cpp schedule.cpp legality.cpp engine.cpp emit.cpp dot_recovery.cpp)
if(DEFINED LLVM_TOOLS_BINARY_DIR AND EXISTS "${LLVM_TOOLS_BINARY_DIR}/llc")
  target_compile_definitions(mvcc-ir2msl PRIVATE MVCC_LLC="${LLVM_TOOLS_BINARY_DIR}/llc")
endif()
target_link_libraries(mvcc-ir2msl PRIVATE ${LLVM_LIBS})
# The LLVM imported targets carry their dependencies as interface libraries; the default strategy keeps the archives
# named above first and appends them again in dependency order, and ld64 warns "ignoring duplicate libraries".
# REORDER_FREELY (CMake 3.31+) emits each archive once, in a valid order.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
  set_property(TARGET mvcc-ir2msl PROPERTY LINK_LIBRARIES_STRATEGY REORDER_FREELY)
endif()
target_compile_options(mvcc-ir2msl PRIVATE -fno-rtti -Wall -Wno-unused-parameter -Wno-deprecated-declarations)

# clang pass plugin for the device pass (nvptx_convergence.cpp): loaded into the Homebrew clang with -fpass-plugin,
# so it resolves LLVM against the libLLVM clang already has rather than linking its own copy.
add_library(mvcc-passes MODULE nvptx_convergence.cpp)
set_target_properties(mvcc-passes PROPERTIES PREFIX "lib" SUFFIX ".dylib")
target_compile_options(mvcc-passes PRIVATE -fno-rtti -Wall -Wno-unused-parameter)
target_link_options(mvcc-passes PRIVATE -undefined dynamic_lookup)

install(TARGETS mvcc-ir2msl RUNTIME DESTINATION bin)
install(TARGETS mvcc-passes LIBRARY DESTINATION lib64)
