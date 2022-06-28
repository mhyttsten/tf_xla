/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMISSION_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMISSION_UTILS_H_
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTh {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTh() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};


#include <string>
#include <utility>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// Matrix multiplication before the rewrite.
//
// This function should never return "true" on instructions after
// GemmRewriter pass has finished.
bool IsMatrixMultiplication(const HloInstruction& dot);

inline constexpr int64_t WarpSize() { return 32; }

// Need at least 1024 threads/block for reasonable tree reduction
// performance (assuming all data fits).
inline constexpr int64_t MinThreadsXRowReduction() { return 1024; }

// When doing batched row reduction, how big the batch dimension could be.
inline constexpr int64_t BatchedReductionRaceFreeBound() { return 8; }

// Returns true if `hlo` will be implemented as a call to a cuSolver routine.
//
// This returns true if `hlo` is a CustomCall HLO with a call target equal to
// one of the kCusolver... constants, but returns *false* for HLOs with
// say, a kCholesky opcode.
bool IsCustomCallToCusolver(const HloInstruction& hlo);

// Cholesky decomposition. Takes a (batched) matrix as input, and returns a
// tuple of (result, workspace, info), where result is the result of the
// Cholesky decomposition, workspace is scratch space for cuSolver, and info
// is a success/failure code per batch element.
extern const char* const kCusolverCholeskyCallTarget;

// Returns true if either the dimensions being reduced or the dimensions being
// kept are contiguous in the input of the reduce instruction.
bool IsReductionFromOrToContiguousDimensions(const HloInstruction& reduce);

// MLIR variant.
bool IsReductionFromOrToContiguousDimensions(mlir::Operation* op);

// Returns whether unnested_hlo is an input fusion whose root is either a slice
// or a tuple of slices. If verify_no_strides is true, returns false unless all
// ROOT slices have no strides.
bool IsInputFusibleSlices(mlir::Operation* unnested_hlo,
                          bool verify_no_strides);

struct ReductionDimensions {
  // Indicates whether the reduction is a row reduction or a column reduction.
  bool is_row_reduction;

  // Contains the size of the three contiguous components for
  // the reduction [depth, height, width] (major-to-minor ordering).
  //
  // For row reduction, we do: [D, H, W] -> [D, H].
  // For column reduction, we do: [D, H, W] -> [D, W].
  std::array<int64_t, 3> dimensions;
};

// Given the input shape and dimensions to reduce for a reduction, returns
// ReductionDimensions.
//
// Prerequisite: the reduction instruction passes the check
// IsReductionFromOrToContiguousDimensions, which guarantees either the
// dimensions to reduce or the dimensions to keep are consecutive.
ReductionDimensions GetReductionKindAndContiguousComponents(
    const HloInstruction& reduce);
ReductionDimensions GetReductionKindAndContiguousComponents(
    mlir::Operation* reduce);

// Get tiling per thread for the given reduction in dimensions [D, H, W].
std::array<int64_t, 3> GetReductionTiling(
    const ReductionDimensions& reduction_dimensions,
    se::CudaComputeCapability cuda_compute_capability);

// Emits call to "vprintf" with given format and arguments.
llvm::Value* EmitPrintf(absl::string_view fmt,
                        absl::Span<llvm::Value* const> arguments,
                        llvm::IRBuilder<>* builder);

// Emits code to shuffle data between threads of a warp. This has the same
// semantics as the PTX "shfl.sync.down" instruction but works for values that
// aren't 32 bits in size. The last operand of the emitted "shfl" is
// `WarpSize() - 1`.
//
// This function emits a "full-warp" shuffle, which all threads of a warp
// participate in.  *Do not use this function from a divergent context:* You
// can't correctly do so on both Volta and earlier GPUs.
//
// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync
llvm::Value* EmitFullWarpShuffleDown(llvm::Value* value, llvm::Value* offset,
                                     llvm::IRBuilder<>* builder);

// Emits code that determines whether the current thread is thread 0 within
// block 0 of the kernel.
llvm::Value* IsBlock0Thread0(llvm::IRBuilder<>* b);

// Returns whether the output of a fusion with reduction are consistent with
// `first_reduce`.
bool IsFusedReductionOutputConsistent(const HloInstruction* inst,
                                      const HloInstruction* first_reduce);
inline bool AreFusedReductionOutputsConsistent(
    absl::Span<const HloInstruction* const> output_instructions,
    const HloInstruction* first_reduce) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTh mht_0(mht_0_v, 298, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.h", "AreFusedReductionOutputsConsistent");

  return absl::c_all_of(output_instructions, [=](const HloInstruction* inst) {
    return IsFusedReductionOutputConsistent(inst, first_reduce);
  });
}

inline std::string MlirToString(mlir::Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTh mht_1(mht_1_v, 307, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.h", "MlirToString");

  std::string s;
  {
    llvm::raw_string_ostream os(s);
    op->print(os);
  }
  return s;
}

inline std::string MlirToString(const mlir::Location& loc) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSir_emission_utilsDTh mht_2(mht_2_v, 319, "", "./tensorflow/compiler/xla/service/gpu/ir_emission_utils.h", "MlirToString");

  std::string s;
  {
    llvm::raw_string_ostream os(s);
    loc.print(os);
  }
  return s;
}

int PartitionLmhloOperandsAndOutputs(mlir::Operation* op);
std::vector<mlir::Value> GetHloOperands(mlir::Operation* op);
std::vector<mlir::Value> GetHloOutputs(mlir::Operation* op);

bool WritesMlirBuffer(mlir::Operation* op, mlir::Value operand);

template <typename T>
std::vector<T> ToStdVector(const llvm::SmallVectorImpl<T>& v) {
  return std::vector<T>(v.begin(), v.end());
}

StatusOr<BufferAllocation::Slice> GetAllocationSlice(
    mlir::Value v, absl::Span<const BufferAllocation> allocations,
    std::string* constant_name = nullptr);

bool CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
    mlir::lmhlo::FusionOp fusion,
    absl::Span<const BufferAllocation> allocations);

Shape GetShape(mlir::Value value);

// Returns whether the given reduction can be safely generated without atomics:
// that is, at most one block will write to every output element.
bool ReductionIsRaceFree(const ReductionDimensions& reduction_dimensions,
                         const std::array<int64_t, 3>& reduction_tiling);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMISSION_UTILS_H_
