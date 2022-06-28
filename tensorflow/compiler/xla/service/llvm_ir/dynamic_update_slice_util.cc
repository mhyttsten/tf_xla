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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc() {
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

#include "tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.h"

#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"

namespace xla {
namespace llvm_ir {

bool MayBeImplementedAsInPlaceDynamicUpdateSlice(const HloInstruction* instr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.cc", "MayBeImplementedAsInPlaceDynamicUpdateSlice");

  // Today we can't emit a dynamic-update-slice if the DUS node is parallelized;
  // the emitter will not emit correct code.  It's possible to change this, but
  // then ParallelTaskAssigner would have to somehow know whether a node *will*
  // be emitted as an in-place DUS, and it can't, because it doesn't have a
  // buffer assignment when it runs.
  if (!instr->outer_dimension_partitions().empty()) {
    return false;
  }

  // Until we know the final buffer assignment, any unfused dynamic-update-slice
  // might be implementable as an in-place DUS.
  if (instr->opcode() == HloOpcode::kDynamicUpdateSlice) {
    return true;
  }

  // A fusion may be implementable as an in-place dynamic update slice if
  //  - it's a loop fusion,
  //  - dynamic-update-slice is the root of the fusion, and
  //  - operand 0 of the dynamic-update-slice is a parameter to the fusion
  //    (ignoring any get-tuple-element operations in the way).
  if (instr->IsLoopFusion()) {
    const HloInstruction* fused_root = instr->fused_expression_root();
    return fused_root->opcode() == HloOpcode::kDynamicUpdateSlice &&
           fused_root->operand(0)->LatestNonGteAncestor()->opcode() ==
               HloOpcode::kParameter;
  }

  return false;
}

bool CanUpdateDynamicSliceInPlace(HloInstruction* dynamic_update_slice,
                                  const BufferAssignment& assignment) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.cc", "CanUpdateDynamicSliceInPlace");

  CHECK_EQ(HloOpcode::kDynamicUpdateSlice, dynamic_update_slice->opcode());
  const HloInstruction* operand = dynamic_update_slice->operand(0);
  return assignment.HasTopLevelAllocation(dynamic_update_slice) &&
         assignment.HasTopLevelAllocation(operand) &&
         assignment.SharesTopLevelSlice(dynamic_update_slice, operand);
}

bool CanEmitFusedDynamicUpdateSliceInPlace(HloInstruction* fusion,
                                           const BufferAssignment& assignment) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc mht_2(mht_2_v, 242, "", "./tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.cc", "CanEmitFusedDynamicUpdateSliceInPlace");

  CHECK_EQ(fusion->opcode(), HloOpcode::kFusion);
  if (!MayBeImplementedAsInPlaceDynamicUpdateSlice(fusion)) {
    return false;
  }

  // Walk DynamicUpdateSlice operand(0) to fused parameter and get its
  // associated operand. See if it shares an allocation with this operand.
  HloInstruction* fused_root = fusion->fused_expression_root();
  HloInstruction* fusion_operand;
  ShapeIndex index;
  std::tie(fusion_operand, index) =
      fused_root->mutable_operand(0)->LatestNonGteAncestorAndIndex();
  // MayBeImplementedAsInPlaceDynamicUpdateSlice should have ensured that
  // fusion_operand is a parameter.
  CHECK_EQ(fusion_operand->opcode(), HloOpcode::kParameter);
  auto* operand = fusion->operand(fusion_operand->parameter_number());
  return assignment.HasAllocationAt(operand, index) &&
         assignment.HasAllocationAt(fusion, {}) &&
         assignment.SharesSliceAtIndex(fusion, {}, operand, index);
}

// Shared implementation of EmitDynamicUpdateSliceInPlace and
// EmitFusedDynamicUpdateSliceInPlace.
//
// Emits a sequential loop if launch_dimensions is null.
using IndexGenerator = std::function<StatusOr<llvm::Value*>(int64_t)>;

static Status EmitDynamicUpdateSliceInPlaceImpl(
    const Shape& update_shape, const IndexGenerator& start_indices_generator,
    bool is_signed, ElementGenerator update_array_generator,
    const IrArray& output_array, const gpu::LaunchDimensions* launch_dimensions,
    absl::string_view name, llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc mht_3(mht_3_v, 278, "", "./tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.cc", "EmitDynamicUpdateSliceInPlaceImpl");

  const Shape& output_shape = output_array.GetShape();

  // Read start indices from start_indices_generator.
  const int64_t rank = output_shape.rank();
  std::vector<llvm::Value*> start_multi_index(rank);
  for (int64_t i = 0; i < rank; ++i) {
    TF_ASSIGN_OR_RETURN(start_multi_index[i], start_indices_generator(i));
    llvm::Value* output_dim_size = llvm::ConstantInt::get(
        start_multi_index[i]->getType(), output_shape.dimensions(i));
    llvm::Value* update_dim_size = llvm::ConstantInt::get(
        start_multi_index[i]->getType(), update_shape.dimensions(i));

    // Clamp the start index so that the update region fits in the operand.
    // start_index = clamp(start_index, 0, output_dim_size - update_dim_size)
    llvm::Value* max_bound = b->CreateSub(output_dim_size, update_dim_size);
    llvm::Value* zero =
        llvm::ConstantInt::get(start_multi_index[i]->getType(), 0);
    start_multi_index[i] =
        b->CreateSelect(b->CreateICmp(is_signed ? llvm::ICmpInst::ICMP_SGE
                                                : llvm::ICmpInst::ICMP_UGE,
                                      zero, start_multi_index[i]),
                        zero, start_multi_index[i]);

    start_multi_index[i] =
        b->CreateSelect(b->CreateICmp(is_signed ? llvm::ICmpInst::ICMP_SLE
                                                : llvm::ICmpInst::ICMP_ULE,
                                      max_bound, start_multi_index[i]),
                        max_bound, start_multi_index[i]);
  }

  auto loop_body_emitter = [&](const IrArray::Index& update_index) -> Status {
    // Calculate output_index, where we'll write the value from update.  For
    // each dimension,
    //
    //   output_index[dim] = start_index[dim] + update_index[dim]
    //
    std::vector<llvm::Value*> output_multi_index(rank);
    for (int64_t i = 0; i < rank; ++i) {
      llvm::Value* start_index0 = b->CreateSExtOrBitCast(
          start_multi_index[i], update_index[i]->getType());
      output_multi_index[i] = b->CreateAdd(start_index0, update_index[i]);
    }

    // Do output[output_index] = update[update_index].
    IrArray::Index output_index(output_multi_index, output_shape,
                                b->getInt64Ty());
    TF_ASSIGN_OR_RETURN(llvm::Value * update_data,
                        update_array_generator(update_index));
    output_array.EmitWriteArrayElement(output_index, update_data, b);
    return Status::OK();
  };

  if (launch_dimensions != nullptr) {
    return gpu::ParallelLoopEmitter(loop_body_emitter, update_shape,
                                    *launch_dimensions, b)
        .EmitLoop(name);
  }
  return LoopEmitter(loop_body_emitter, update_shape, b).EmitLoop(name);
}

Status EmitDynamicUpdateSliceInPlace(absl::Span<const IrArray> operand_arrays,
                                     const IrArray& output_array,
                                     absl::string_view name,
                                     llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc mht_4(mht_4_v, 346, "", "./tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.cc", "EmitDynamicUpdateSliceInPlace");

  VLOG(2) << "EmitDynamicUpdateSliceInPlace for " << name;

  // No need to use operand_arrays[0], the input array of the
  // dynamic-update-slice, because we know it aliases the op's output.
  IrArray update_array = operand_arrays[1];
  IrArray start_indices_array = operand_arrays[2];
  Shape output_shape = output_array.GetShape();
  Shape update_shape = update_array.GetShape();

  IndexGenerator start_indices_generator = [&](int64_t index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc mht_5(mht_5_v, 359, "", "./tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.cc", "lambda");

    return operand_arrays[2 + index].EmitReadArrayElement(
        IrArray::Index(b->getInt64Ty()), b);
  };
  ElementGenerator update_array_generator = [&](const IrArray::Index& index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc mht_6(mht_6_v, 366, "", "./tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.cc", "lambda");

    return update_array.EmitReadArrayElement(index, b);
  };

  bool is_signed = ShapeUtil::ElementIsSigned(start_indices_array.GetShape());
  return EmitDynamicUpdateSliceInPlaceImpl(
      update_shape, start_indices_generator, is_signed, update_array_generator,
      output_array, /*launch_dimensions=*/nullptr, name, b);
}

// Shared implementation for EmitFusedDynamicUpdateSliceInPlace and
// EmitParallelFusedDynamicUpdateSliceInPlace.
//
// Emits a sequential loop if launch_dimensions is null.
static Status EmitFusedDynamicUpdateSliceInPlaceImpl(
    const HloComputation* fusion, const IrArray& fusion_output_array,
    FusedIrEmitter* fused_emitter,
    const gpu::LaunchDimensions* launch_dimensions, llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc mht_7(mht_7_v, 386, "", "./tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.cc", "EmitFusedDynamicUpdateSliceInPlaceImpl");

  VLOG(2) << "EmitFusedDynamicUpdateSliceInPlace for " << fusion->ToString();

  auto* dynamic_update_slice = fusion->root_instruction();

  const auto* update = dynamic_update_slice->operand(1);
  const auto* start_indices = dynamic_update_slice->operand(2);
  Shape update_shape = update->shape();

  // Our in-place dynamic-update-slice implementation emits a loop over
  // update_shape.  To emit a cache-friendly loop, we need to know that shape's
  // layout.
  //
  // update_shape is inside a fusion node -- it's never materialized in memory
  // and thus doesn't have a layout.  In this case we use the layout of the
  // fusion node for iteration, since that corresponds to the order in memory of
  // the buffer we'll be writing to.
  //
  // (This isn't necessarily optimal; in some cases it might be faster to peek
  // through the chain of ops that gives us the update operand and use the
  // layout of its source buffer(s).  But this is no worse than we do with
  // fusion elsewhere.)
  TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
      dynamic_update_slice->shape(), &update_shape));

  // Create element generators for update and start_indices.
  TF_ASSIGN_OR_RETURN(ElementGenerator update_array_generator,
                      fused_emitter->GetGenerator(*update));

  IndexGenerator start_indices_generator =
      [&](int64_t index) -> StatusOr<llvm::Value*> {
    TF_ASSIGN_OR_RETURN(
        ElementGenerator element_generator,
        fused_emitter->GetGenerator(*dynamic_update_slice->operand(2 + index)));
    return element_generator(IrArray::Index(b->getInt64Ty()));
  };
  bool is_signed = ShapeUtil::ElementIsSigned(start_indices->shape());
  return EmitDynamicUpdateSliceInPlaceImpl(
      update_shape, start_indices_generator, is_signed, update_array_generator,
      fusion_output_array, launch_dimensions, IrName(dynamic_update_slice), b);
}

Status EmitFusedDynamicUpdateSliceInPlace(HloInstruction* fusion,
                                          const IrArray& fusion_output_array,
                                          FusedIrEmitter* fused_emitter,
                                          llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc mht_8(mht_8_v, 434, "", "./tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.cc", "EmitFusedDynamicUpdateSliceInPlace");

  return EmitFusedDynamicUpdateSliceInPlaceImpl(
      fusion->called_computations()[0], fusion_output_array, fused_emitter,
      /*launch_dimensions=*/nullptr, b);
}

Status EmitParallelFusedDynamicUpdateSliceInPlace(
    const HloComputation* fusion, const IrArray& fusion_output_array,
    FusedIrEmitter* fused_emitter,
    const gpu::LaunchDimensions& launch_dimensions, llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSdynamic_update_slice_utilDTcc mht_9(mht_9_v, 446, "", "./tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.cc", "EmitParallelFusedDynamicUpdateSliceInPlace");

  return EmitFusedDynamicUpdateSliceInPlaceImpl(
      fusion, fusion_output_array, fused_emitter, &launch_dimensions, b);
}

}  // namespace llvm_ir
}  // namespace xla
