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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSparallel_loop_emitterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSparallel_loop_emitterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSparallel_loop_emitterDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"

#include <memory>

#include "tensorflow/core/platform/logging.h"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {

ParallelLoopEmitter::ParallelLoopEmitter(
    llvm_ir::BodyEmitter body_emitter, const Shape& shape,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* b,
    LaunchDimensionsConfig launch_config)
    : launch_dimensions_(launch_dimensions),
      launch_config_(launch_config),
      body_emitter_(body_emitter),
      shape_(shape),
      b_(b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSparallel_loop_emitterDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.cc", "ParallelLoopEmitter::ParallelLoopEmitter");
}

ParallelLoopEmitter::ParallelLoopEmitter(
    const llvm_ir::ElementGenerator& target_element_generator,
    absl::Span<const llvm_ir::IrArray> target_arrays,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* b,

    LaunchDimensionsConfig launch_config)
    : launch_dimensions_(launch_dimensions),
      launch_config_(launch_config),
      body_emitter_(
          llvm_ir::MakeBodyEmitter(target_element_generator, target_arrays, b,
                                   /*is_tuple=*/target_arrays.size() > 1)),
      shape_(target_arrays[0].GetShape()),
      b_(b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSparallel_loop_emitterDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.cc", "ParallelLoopEmitter::ParallelLoopEmitter");
}

std::vector<llvm_ir::IrArray::Index>
ParallelLoopEmitter::EmitIndexAndSetExitBasicBlock(absl::string_view loop_name,
                                                   llvm::Type* index_type,
                                                   llvm::Value* base_index) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("loop_name: \"" + std::string(loop_name.data(), loop_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSparallel_loop_emitterDTcc mht_2(mht_2_v, 236, "", "./tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.cc", "ParallelLoopEmitter::EmitIndexAndSetExitBasicBlock");

  // Emit the following code in LLVM IR:
  //   linear_index = blockIdx.x * blockDim.x * blockDim.y [+ threadIdx.y *
  //   blockDim.x] + threadIdx.x; if (linear_index < num_elements) {
  //     array_index = LinearIndexToMultidimensionalIndex(shape_, linear_index);
  //     ...
  //   }
  // The part between [] are added only if blockDim.y > 1.
  // blockIdx.y and gridDim.y are always 1.

  // Per the PTX documentation:
  //   "It is guaranteed that [...] 0  <=  %ctaid.x <  %nctaid.x"
  //
  // %nctaid.x is currently specified as 2147483647.
  if (launch_dimensions_.thread_counts_per_block().y > 1) {
    // When blockDim.y > 1, then we are in the small row case. Each
    // blockDim.x do exatly to one row and blockDim.y map to some
    // consecutive row. This prevents too small block size that isn't
    // efficient.
    CHECK(launch_config_.row_vectorized);
    CHECK_EQ(shape_.dimensions().back(),
             launch_dimensions_.thread_counts_per_block().x *
                 launch_config_.unroll_factor);
  }
  CHECK_EQ(launch_dimensions_.thread_counts_per_block().z, 1);
  CHECK_EQ(launch_dimensions_.block_counts().y, 1);
  CHECK_EQ(launch_dimensions_.block_counts().z, 1);
  VLOG(3) << "EmitIndexAndSetExitBasicBlock unroll_factor "
          << launch_config_.unroll_factor;
  CHECK_NE(index_type, nullptr);
  std::vector<llvm_ir::IrArray::Index> array_indices;
  llvm::Value* block_id =
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, b_);
  llvm_ir::AddRangeMetadata(0, launch_dimensions_.block_counts().x,
                            static_cast<llvm::Instruction*>(block_id));
  block_id = b_->CreateZExtOrTrunc(block_id, index_type, "block_id");

  // Per the PTX documentation:
  //   "It is guaranteed that [...] 0  <=  %tid.x <  %ntid.x"
  llvm::Value* thread_id_x =
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, b_);
  llvm_ir::AddRangeMetadata(0, launch_dimensions_.thread_counts_per_block().x,
                            static_cast<llvm::Instruction*>(thread_id_x));
  thread_id_x = b_->CreateZExtOrTrunc(thread_id_x, index_type, "thread_id_x");

  llvm::Value* linear_index_base = b_->CreateMul(
      block_id,
      llvm::ConstantInt::get(index_type, launch_dimensions_.total_nb_threads()),
      "",
      /*HasNUW=*/true, /*HasNSW=*/true);
  if (launch_dimensions_.thread_counts_per_block().y > 1) {
    llvm::Value* thread_id_y =
        EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdy, {}, {}, b_);
    llvm_ir::AddRangeMetadata(0, launch_dimensions_.thread_counts_per_block().y,
                              static_cast<llvm::Instruction*>(thread_id_y));
    thread_id_y = b_->CreateZExtOrTrunc(thread_id_y, index_type, "thread_id_y");
    linear_index_base = b_->CreateAdd(
        linear_index_base,
        b_->CreateMul(
            thread_id_y,
            llvm::ConstantInt::get(
                index_type, launch_dimensions_.thread_counts_per_block().x),
            "",
            /*HasNUW=*/true, /*HasNSW=*/true),
        "",
        /*HasNUW=*/true, /*HasNSW=*/true);
  }
  linear_index_base =
      b_->CreateAdd(linear_index_base, thread_id_x, "linear_index",
                    /*HasNUW=*/true, /*HasNSW=*/true);

  // Add an @llvm.assume(linear_index < threads_per_block * num_blocks).
  //
  // This might seem obvious from the computation above, but LLVM does not
  // currently determine the range of linear_index precisely.  InstCombine uses
  // known-bits, which, when applied to the task of determining a value's range,
  // is imprecise for everything other than powers of 2.  And
  // CorrelatedValuePropagation is, as a cost-saving measure, disabled for
  // conditions in the same basic block as their operands.
  llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::assume,
      {b_->CreateICmpULT(
          linear_index_base,
          llvm::ConstantInt::get(index_type,
                                 launch_dimensions_.total_nb_threads() *
                                     launch_dimensions_.block_counts().x),
          "linear_index_in_range")},
      {}, b_);

  if (launch_config_.unroll_factor > 1) {
    linear_index_base = b_->CreateMul(
        linear_index_base,
        llvm::ConstantInt::get(index_type, launch_config_.unroll_factor),
        "linear_index_base", /*HasNUW=*/true, /*HasNSW=*/true);
  }

  if (base_index != nullptr) {
    linear_index_base =
        b_->CreateAdd(linear_index_base, base_index, "linear_index_plus_base",
                      /*HasNUW=*/true, /*HasNSW=*/true);
  }

  // When enable_row_index is true, it means the inner most dimensions
  // match the block sizes.  So we can generate a simpler indexing
  // for that dimensions.  This helps LLVM generate vectorized codes
  // in that cases.
  llvm::Value* row_index = nullptr;
  if (!launch_config_.row_vectorized) {
    array_indices.emplace_back(linear_index_base, shape_, b_);
  } else {
    // Simpler index for row computation.
    // This will allow LLVM to vectorize.
    row_index = b_->CreateMul(
        thread_id_x,
        llvm::ConstantInt::get(index_type, launch_config_.unroll_factor),
        "row_index", /*HasNUW=*/true, /*HasNSW=*/true);
    std::vector<llvm::Value*> multidim(shape_.rank(), nullptr);
    multidim.back() = row_index;
    array_indices.emplace_back(linear_index_base, multidim, shape_, b_);
  }

  for (int i = 1; i < launch_config_.unroll_factor; ++i) {
    llvm::Value* linear_index =
        b_->CreateAdd(linear_index_base, llvm::ConstantInt::get(index_type, i),
                      absl::StrCat("linear_index", i),
                      /*HasNUW=*/true, /*HasNSW=*/true);
    if (!launch_config_.row_vectorized) {
      array_indices.emplace_back(linear_index, shape_, b_);
    } else {
      std::vector<llvm::Value*> multidim(shape_.rank(), nullptr);
      multidim.back() = b_->CreateAdd(
          row_index, llvm::ConstantInt::get(index_type, i),
          absl::StrCat("row_index_plus", i), /*HasNUW=*/true, /*HasNSW=*/true);
      array_indices.emplace_back(linear_index, multidim, shape_, b_);
    }
  }

  auto if_in_bounds = llvm_ir::EmitIfThenElse(
      b_->CreateICmpULT(
          linear_index_base,
          llvm::ConstantInt::get(index_type, ShapeUtil::ElementsIn(shape_))),
      llvm_ir::IrName(loop_name, "in_bounds"), b_, false);

  // Set exit_bb_ to the exit block of the if structure.
  exit_bb_ = if_in_bounds.after_block;
  CHECK_NE(nullptr, exit_bb_);

  // Set IR builder insertion point to the body of the if structure.
  llvm_ir::SetToFirstInsertPoint(if_in_bounds.true_block, b_);

  return array_indices;
}

Status ParallelLoopEmitter::EmitSerialLoop(absl::string_view loop_name,
                                           llvm::Type* index_type,
                                           llvm::Value* base_indvar) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("loop_name: \"" + std::string(loop_name.data(), loop_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSparallel_loop_emitterDTcc mht_3(mht_3_v, 395, "", "./tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.cc", "ParallelLoopEmitter::EmitSerialLoop");

  for (const llvm_ir::IrArray::Index& array_index :
       EmitIndexAndSetExitBasicBlock(loop_name, index_type, base_indvar)) {
    TF_RETURN_IF_ERROR(body_emitter_(array_index));
  }
  return Status::OK();
}

Status ParallelLoopEmitter::EmitLoop(absl::string_view loop_name,
                                     llvm::Type* index_type) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("loop_name: \"" + std::string(loop_name.data(), loop_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSparallel_loop_emitterDTcc mht_4(mht_4_v, 408, "", "./tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.cc", "ParallelLoopEmitter::EmitLoop");

  if (index_type == nullptr) {
    index_type = b_->getInt64Ty();
  }
  int64_t total_threads = launch_dimensions_.launch_bound();
  int64_t num_elements = ShapeUtil::ElementsIn(shape_);
  // If all the elements are handled by the current threads, no need
  // to add a loop inside the kernel.
  if (total_threads * launch_config_.unroll_factor >= num_elements) {
    VLOG(1) << "No loops inside the kernel";
    TF_RETURN_IF_ERROR(EmitSerialLoop(loop_name, index_type));
  } else {
    KernelSupportLibrary ksl(b_, llvm_ir::UnrollMode::kDefaultUnroll);
    auto constant = [&](int64_t val) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSparallel_loop_emitterDTcc mht_5(mht_5_v, 424, "", "./tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.cc", "lambda");

      return llvm::ConstantInt::get(index_type, val);
    };

    TF_RETURN_IF_ERROR(ksl.ForWithStatus(
        "loop", constant(0), constant(num_elements),
        constant(total_threads * launch_config_.unroll_factor),
        [&](llvm::Value* base_indvar) {
          return EmitSerialLoop(loop_name, index_type, base_indvar);
        }));
  }

  // Set the insertion point of b_ to the loop exit, so that
  // code emitted for later instructions will be correctly placed.
  if (exit_bb_ != nullptr) {
    b_->SetInsertPoint(exit_bb_);
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
