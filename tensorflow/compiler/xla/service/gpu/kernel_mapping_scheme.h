/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_MAPPING_SCHEME_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_MAPPING_SCHEME_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh() {
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


#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

using Vector3 = std::array<int64_t, 3>;

// Describes tiling used by the kernel.
//
// Used by reductions and 021 transpose algorithm. Both algorithms operate over
// "logical" 3D views over input arrays, hence tiling and number of threads
// information has only 3 dimensions.
//
// In the presence of virtual threadIdx/blockIdx scaling, all accessors are
// "logical", unless otherwise specified.
class TilingScheme {
 public:
  enum { DimZ = 0, DimY, DimX, DimTot };

  enum IndexingOrder {
    // Thread reads consecutive elements.
    LinearIndexingX,
    // Thread reads strided elements while keeping memory coalescing.
    StridedIndexingX,
  };

  TilingScheme(Vector3 dims_in_elems, Vector3 tile_sizes, Vector3 num_threads,
               IndexingOrder indexing_order, int vector_size,
               int scaling_factor)
      : dims_in_elems_(dims_in_elems),
        tile_sizes_(tile_sizes),
        num_threads_(num_threads),
        indexing_order_(indexing_order),
        vector_size_(vector_size),
        thread_id_virtual_scaling_(scaling_factor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_0(mht_0_v, 228, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "TilingScheme");

    CHECK_EQ(tile_sizes[2] % vector_size_, 0);
  }

  static std::string IndexingOrderToString(IndexingOrder order) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_1(mht_1_v, 235, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "IndexingOrderToString");

    switch (order) {
      case LinearIndexingX:
        return "linear";
      case StridedIndexingX:
        return "strided";
    }
  }

  std::string ToString() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_2(mht_2_v, 247, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "ToString");

    return absl::StrJoin(
        {absl::StrFormat("dims_in_elems = {%s}",
                         absl::StrJoin(dims_in_elems_, ", ")),
         absl::StrFormat("tile_sizes = {%s}", absl::StrJoin(tile_sizes_, ", ")),
         absl::StrFormat("num_threads = {%s}",
                         absl::StrJoin(num_threads_, ", ")),
         absl::StrFormat("indexing_order = %s",
                         IndexingOrderToString(indexing_order_)),
         absl::StrFormat("vector_size = %d", vector_size_)},
        ", ");
  }

  // Number of elements in each dimension (Z/Y/X respectively).
  absl::Span<const int64_t> GetDimsInElems() const { return dims_in_elems_; }

  Vector3 GetDimsInBlocks() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_3(mht_3_v, 266, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetDimsInBlocks");

    return {GetDimInBlock(0), GetDimInBlock(1), GetDimInBlock(2)};
  }

  // Number of blocks required to "cover" the given dimension.
  int64_t GetDimInBlock(int d) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_4(mht_4_v, 274, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetDimInBlock");

    return CeilOfRatio(dims_in_elems_[d], GetBlockTileSizeFor(d));
  }

  // Tile size for a given dimensions per thread.
  //
  // Equals to the number of iterations in the loop each tile will make.
  int64_t GetTileSizeFor(int d) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_5(mht_5_v, 284, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetTileSizeFor");
 return tile_sizes_.at(d); }

  // Tile size for a given dimension per entire thread block.
  int64_t GetBlockTileSizeFor(int d) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_6(mht_6_v, 290, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetBlockTileSizeFor");

    return num_threads_.at(d) * tile_sizes_.at(d);
  }

  // Number of threads in given dimension.
  int64_t GetNumThreadsFor(int d) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_7(mht_7_v, 298, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetNumThreadsFor");
 return num_threads_.at(d); }

  // Number of logical threads per block.
  int64_t GetNumThreadsPerBlock() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_8(mht_8_v, 304, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetNumThreadsPerBlock");

    return GetNumThreadsFor(0) * GetNumThreadsFor(1) * GetNumThreadsFor(2);
  }

  // Number of logical blocks.
  int64_t GetNumberOfBlocks() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_9(mht_9_v, 312, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetNumberOfBlocks");

    return GetDimInBlock(0) * GetDimInBlock(1) * GetDimInBlock(2);
  }

  // Number of physical blocks launched (with scaling applied).
  int64_t GetNumberOfBlocksPhysical() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_10(mht_10_v, 320, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetNumberOfBlocksPhysical");

    return CeilOfRatio(GetNumberOfBlocks(), thread_id_virtual_scaling_);
  }

  // Number of physical threads per block launched (with scaling applied).
  int64_t GetNumThreadsPerBlockPhysical() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_11(mht_11_v, 328, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetNumThreadsPerBlockPhysical");

    return GetNumThreadsPerBlock() * thread_id_virtual_scaling_;
  }

  IndexingOrder GetIndexingOrder() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_12(mht_12_v, 335, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetIndexingOrder");
 return indexing_order_; }
  int GetVectorSize() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_13(mht_13_v, 339, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetVectorSize");
 return vector_size_; }

  // Scaling factor for transforming physical threadId to logical.
  int GetThreadIdScalingFactor() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_14(mht_14_v, 345, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetThreadIdScalingFactor");
 return thread_id_virtual_scaling_; }

 private:
  // The number of elements in each dimension.
  const Vector3 dims_in_elems_;

  // The number of elements for each dimension of a tile.
  const Vector3 tile_sizes_;

  // Number of threads implicitly assigned to each dimension.
  const Vector3 num_threads_;

  const IndexingOrder indexing_order_;

  // Vector size for dimension X.
  const int vector_size_;

  // Scaling apply to transform physical threadIdx into logical.
  const int64_t thread_id_virtual_scaling_ = 1;
};

class ReductionCodegenInfo {
 public:
  explicit ReductionCodegenInfo(TilingScheme mapping_scheme,
                                int num_partial_results, bool is_row_reduction,
                                bool is_race_free)
      : tiling_scheme_(mapping_scheme),
        num_partial_results_(num_partial_results),
        is_row_reduction_(is_row_reduction),
        is_race_free_(is_race_free) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_15(mht_15_v, 377, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "ReductionCodegenInfo");

    if (num_partial_results > 1) {
      CHECK_EQ(num_partial_results,
               mapping_scheme.GetTileSizeFor(TilingScheme::DimX));
    }
  }

  const TilingScheme& GetTilingScheme() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_16(mht_16_v, 387, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetTilingScheme");
 return tiling_scheme_; }

  int GetNumPartialResults() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_17(mht_17_v, 392, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetNumPartialResults");
 return num_partial_results_; }
  bool IsRaceFree() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_18(mht_18_v, 396, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "IsRaceFree");
 return is_race_free_; }

 private:
  friend class ReductionCodegenState;

  const TilingScheme tiling_scheme_;
  int num_partial_results_;
  bool is_row_reduction_;
  bool is_race_free_;
};

class ReductionCodegenState {
 public:
  struct ReductionCalculationState {
    llvm::GlobalVariable* shared_cache;
    llvm::Value* initial_value;
    llvm::AllocaInst* partial_result_address;
    llvm::AllocaInst* input_address;
    llvm_ir::ElementGenerator input_gen;
  };

  explicit ReductionCodegenState(
      const ReductionCodegenInfo& reduction_codegen_info)
      : reduction_codegen_info_(reduction_codegen_info) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_19(mht_19_v, 422, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "ReductionCodegenState");
}

  const TilingScheme& GetTilingScheme() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_20(mht_20_v, 427, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetTilingScheme");

    return reduction_codegen_info_.tiling_scheme_;
  }

  int GetNumPartialResults() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_21(mht_21_v, 434, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetNumPartialResults");

    return reduction_codegen_info_.num_partial_results_;
  }

  bool IsRowReduction() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_22(mht_22_v, 441, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "IsRowReduction");

    return reduction_codegen_info_.is_row_reduction_;
  }

  bool IsRaceFree() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_23(mht_23_v, 448, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "IsRaceFree");
 return reduction_codegen_info_.IsRaceFree(); }

  const ReductionCalculationState& GetCalculationStateFor(
      const HloInstruction* instruction, int operand_idx) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_24(mht_24_v, 454, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "GetCalculationStateFor");

    const ReductionOpState& op_state = state_.at(instruction);
    CHECK_LT(operand_idx, op_state.size());
    return op_state[operand_idx];
  }

  void SetCalculationStateFor(
      const ReductionCalculationState& calculation_state,
      const HloInstruction* instruction, int operand_idx) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_mapping_schemeDTh mht_25(mht_25_v, 465, "", "./tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h", "SetCalculationStateFor");

    ReductionOpState& op_state = state_[instruction];
    CHECK_EQ(operand_idx, op_state.size());
    op_state.push_back(calculation_state);
  }

 private:
  ReductionCodegenInfo reduction_codegen_info_;

  // One state per reduction operand.
  using ReductionOpState = absl::InlinedVector<ReductionCalculationState, 2>;

  // HloInstruction -> operand_idx -> cache
  absl::flat_hash_map<const HloInstruction*, ReductionOpState> state_;
};

}  // end namespace gpu
}  // end namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_MAPPING_SCHEME_H_
