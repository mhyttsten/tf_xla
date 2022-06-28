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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/llvm_ir/sort_util.h"

#include <vector>

// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace llvm_ir {

namespace {

// Adds the inner comparison loop body where we compare elements.
Status EmitCompareLoopBody(
    int64_t iteration_bound, int64_t num_values,
    llvm::Value* element_pair_index, int64_t xor_mask, llvm::Type* index_type,
    std::function<llvm::Value*(int64_t operand, llvm::Value* index)>
        element_address,
    std::function<void(int64_t operand, llvm::Value* index, llvm::Value* value)>
        write_element,
    const EmitCallToNestedComputationCallback& emit_compare_callback,
    llvm::IRBuilder<>* b, bool needs_bounds_checks = true) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc mht_0(mht_0_v, 226, "", "./tensorflow/compiler/xla/service/llvm_ir/sort_util.cc", "EmitCompareLoopBody");

  auto index_typed_constant = [&](int64_t value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/xla/service/llvm_ir/sort_util.cc", "lambda");

    return llvm::ConstantInt::get(index_type, value);
  };
  // The 'xor_mask' determines which elements are compared against each other.
  // Index 'current_keys_index' will be compared with 'current_keys_index' xor
  // 'xor_mask'. This means that we will always compare a block of consecutive
  // elements against elements from the adjacent block of the same size. When
  // 'xor_mask' is a power of 2, it immediately identifies the size of such a
  // block. We can also have 'xor_mask' being 2^k - 1 (for some value of k). In
  // that case, we essentially flip the last 'k' - 1 bits when computing the
  // position of the element to compare to, so the block size is 2^(k - 1).
  int64_t block_size = xor_mask;
  // Check if it is a value 2^k - 1.
  if (xor_mask > 1 && (xor_mask & (xor_mask + 1)) == 0) {
    block_size = (xor_mask + 1) / 2;
  }
  auto current_keys_index = element_pair_index;
  if (block_size == 1) {
    // If the block size is 1, we take every second element and compare it to
    // the next one.
    current_keys_index =
        b->CreateMul(current_keys_index, index_typed_constant(2));
  } else if (block_size * 2 < iteration_bound) {
    // current_keys_index iterates through the 'left' elements of the element
    // pairs to be compared. We first need to compute the comparison block to
    // which the element belongs. The block id of that block is index /
    // block_size.
    auto block_id =
        b->CreateUDiv(current_keys_index, index_typed_constant(block_size));
    // The index of the 'left' element within its block is simply the remainder
    // when dividing by 'block_size'.
    auto index_within_block =
        b->CreateURem(current_keys_index, index_typed_constant(block_size));
    // The first element of the 'left' block of elements that is compared
    // against elements from the adjacent 'right' block of elements is
    // 'block_id' * (2 * 'block_size').
    auto first_element_in_block =
        b->CreateMul(block_id, index_typed_constant(2 * block_size));
    current_keys_index =
        b->CreateAdd(first_element_in_block, index_within_block);
  }
  auto compare_keys_index =
      b->CreateXor(current_keys_index, index_typed_constant(xor_mask));
  // current_keys_index < compare_keys_index
  llvm::Value* is_smaller_index =
      b->CreateICmpSLT(current_keys_index, compare_keys_index);
  // compare_keys_index < iteration_bound
  llvm::Value* index_is_inbounds = b->CreateICmpSLT(
      compare_keys_index, index_typed_constant(iteration_bound));
  llvm::Value* do_comparison =
      needs_bounds_checks ? b->CreateAnd(is_smaller_index, index_is_inbounds)
                          : b->getInt1(true);

  // if (is_smaller_index && index_is_inbounds)
  KernelSupportLibrary ksl(b);
  return ksl.IfWithStatus("smaller_comparison_index", do_comparison, [&]() {
    std::vector<llvm::Value*> values_to_compare;
    for (int i = 0; i < num_values; ++i) {
      values_to_compare.push_back(element_address(i, compare_keys_index));
      values_to_compare.push_back(element_address(i, current_keys_index));
    }
    llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
    llvm::Type* pred_type = llvm_ir::PrimitiveTypeToIrType(PRED, module);
    llvm::Value* compare_return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
        pred_type, "compare_return_buffer", b);
    TF_RETURN_IF_ERROR(
        emit_compare_callback(values_to_compare, compare_return_buffer));
    llvm::Value* result = b->CreateLoad(pred_type, compare_return_buffer);

    // Check if the 'compare' function returns true.
    llvm::Value* is_smaller_than =
        b->CreateICmpNE(result, llvm::ConstantInt::get(result->getType(), 0),
                        "boolean_predicate");
    ksl.If("is_smaller_than", is_smaller_than, [&]() {
      for (int64_t i = 0; i < num_values; ++i) {
        // Swap the values.
        auto value1 = b->CreateLoad(
            values_to_compare[i * 2]->getType()->getPointerElementType(),
            values_to_compare[i * 2]);
        auto value2 = b->CreateLoad(
            values_to_compare[i * 2 + 1]->getType()->getPointerElementType(),
            values_to_compare[i * 2 + 1]);
        write_element(i, current_keys_index, value1);
        write_element(i, compare_keys_index, value2);
      }
    });
    return Status::OK();
  });
}

Status EmitTiledCompareLoop(
    const IrArray::Index& tiled_keys_index, int64_t dimension_to_sort,
    int64_t dimension_to_sort_bound, absl::Span<const int64_t> xor_masks,
    const std::vector<IrArray>& params,
    const std::vector<llvm::Value*>& param_shmem_buffers, int64_t tile_size,
    const EmitCallToNestedComputationCallback& emit_compare_callback,
    llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc mht_2(mht_2_v, 329, "", "./tensorflow/compiler/xla/service/llvm_ir/sort_util.cc", "EmitTiledCompareLoop");

  KernelSupportLibrary ksl(b);
  llvm::Value* thread_id = gpu::EmitCallToTargetIntrinsic(
      gpu::TargetIntrinsicID::kThreadIdx, {}, {}, b);
  llvm_ir::AddRangeMetadata(0, tile_size / 2,
                            llvm::cast<llvm::Instruction>(thread_id));
  thread_id = b->CreateIntCast(thread_id, tiled_keys_index.GetType(),
                               /*isSigned=*/true, "thread.id.x");

  auto copy_loop_body =
      [&](std::function<void(llvm::Value * cache_index, llvm::Value * index)>
              read_or_write) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc mht_3(mht_3_v, 343, "", "./tensorflow/compiler/xla/service/llvm_ir/sort_util.cc", "lambda");

        auto value_one = tiled_keys_index.GetConstantWithIndexType(1);
        auto current_keys_index =
            b->CreateShl(tiled_keys_index[dimension_to_sort], value_one);
        // We want to copy two adjacent elements. We first check whether the
        // first index position is within bounds.
        ksl.If(
            "smaller_keys_index",
            b->CreateICmpSLT(current_keys_index,
                             tiled_keys_index.GetConstantWithIndexType(
                                 dimension_to_sort_bound)),
            [&]() {
              auto cache_index = b->CreateShl(thread_id, value_one);
              read_or_write(cache_index, current_keys_index);
              // Increment to go to the next index position.
              current_keys_index = b->CreateAdd(current_keys_index, value_one);
              // Here we check whether the next index position is within bounds.
              ksl.If("inner_smaller_keys_index",
                     b->CreateICmpSLT(current_keys_index,
                                      tiled_keys_index.GetConstantWithIndexType(
                                          dimension_to_sort_bound)),
                     [&]() {
                       cache_index = b->CreateAdd(cache_index, value_one);
                       read_or_write(cache_index, current_keys_index);
                     });
            });
      };

  // Copy operand tiles from the operand buffers to shared memory.
  std::vector<llvm::Value*> keys_multi_index = tiled_keys_index.multidim();
  for (int64_t i = 0; i < params.size(); ++i) {
    copy_loop_body([&](llvm::Value* cache_index, llvm::Value* index) {
      keys_multi_index[dimension_to_sort] = index;
      IrArray::Index keys_index(keys_multi_index, params[i].GetShape(),
                                tiled_keys_index.GetType());
      auto value = params[i].EmitReadArrayElement(keys_index, b);
      b->CreateStore(
          value,
          b->CreateGEP(
              param_shmem_buffers[i]->getType()->getPointerElementType(),
              param_shmem_buffers[i],
              {tiled_keys_index.GetConstantWithIndexType(0), cache_index}));
    });
  }
  // Wait until all reads have happened.
  gpu::EmitCallToTargetIntrinsic(gpu::TargetIntrinsicID::kBarrierId, {}, {}, b);

  // Now emit the bodies of the comparison loops.
  auto element_address = [&](int64_t operand, llvm::Value* index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc mht_4(mht_4_v, 394, "", "./tensorflow/compiler/xla/service/llvm_ir/sort_util.cc", "lambda");

    auto shared_memory_address = b->CreateGEP(
        param_shmem_buffers[operand]->getType()->getPointerElementType(),
        param_shmem_buffers[operand],
        {tiled_keys_index.GetConstantWithIndexType(0), index});
    auto ptr_type = shared_memory_address->getType();
    // We need a generic pointer with address space 0 instead of a pointer to
    // shared memory (address space 3) so that we can pass it to the comparison
    // computation.
    return b->CreateAddrSpaceCast(
        shared_memory_address,
        llvm::PointerType::get(ptr_type->getPointerElementType(),
                               /*AddressSpace=*/0));
  };
  auto write_element = [&](int64_t operand, llvm::Value* index,
                           llvm::Value* value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc mht_5(mht_5_v, 412, "", "./tensorflow/compiler/xla/service/llvm_ir/sort_util.cc", "lambda");

    b->CreateStore(
        value,
        b->CreateGEP(
            param_shmem_buffers[operand]->getType()->getPointerElementType(),
            param_shmem_buffers[operand],
            {tiled_keys_index.GetConstantWithIndexType(0), index}));
  };
  for (int64_t xor_mask : xor_masks) {
    // The index of the element pair to be compared within the tile stored in
    // shared memory. We order the element pairs by the element with the smaller
    // index.
    auto element_pair_index = thread_id;
    // If 'dimension_to_sort_bound' is evenly divisible by 'tile_size', we don't
    // need any bounds checks.
    if (dimension_to_sort_bound % tile_size) {
      // Otherwise we need a bounds check for the last tile. The last tile has
      // size 'dimension_to_sort_bound' % 'tile_size'.
      TF_RETURN_IF_ERROR(ksl.IfWithStatus(
          "is_last_tile",
          b->CreateICmpUGE(
              b->CreateMul(tiled_keys_index[dimension_to_sort],
                           tiled_keys_index.GetConstantWithIndexType(2)),
              tiled_keys_index.GetConstantWithIndexType(
                  RoundDownTo(dimension_to_sort_bound, tile_size))),
          [&]() {
            return EmitCompareLoopBody(
                dimension_to_sort_bound % tile_size, params.size(),
                element_pair_index, xor_mask, tiled_keys_index.GetType(),
                element_address, write_element, emit_compare_callback, b);
          },
          [&]() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc mht_6(mht_6_v, 446, "", "./tensorflow/compiler/xla/service/llvm_ir/sort_util.cc", "lambda");

            return EmitCompareLoopBody(
                tile_size, params.size(), element_pair_index, xor_mask,
                tiled_keys_index.GetType(), element_address, write_element,
                emit_compare_callback, b,
                /*needs_bounds_checks=*/false);
          }));
    } else {
      TF_RETURN_IF_ERROR(EmitCompareLoopBody(
          tile_size, params.size(), element_pair_index, xor_mask,
          tiled_keys_index.GetType(), element_address, write_element,
          emit_compare_callback, b,
          /*needs_bounds_checks=*/false));
    }
    // Wait until all comparisons have happened.
    gpu::EmitCallToTargetIntrinsic(gpu::TargetIntrinsicID::kBarrierId, {}, {},
                                   b);
  }

  // Copy the operand tiles back from shared memory to the operand buffers.
  for (int64_t i = 0; i < params.size(); ++i) {
    copy_loop_body([&](llvm::Value* cache_index, llvm::Value* index) {
      keys_multi_index[dimension_to_sort] = index;
      IrArray::Index keys_index(keys_multi_index, params[i].GetShape(),
                                tiled_keys_index.GetType());
      auto gep = b->CreateGEP(
          param_shmem_buffers[i]->getType()->getPointerElementType(),
          param_shmem_buffers[i],
          {tiled_keys_index.GetConstantWithIndexType(0), cache_index});
      auto value = b->CreateLoad(gep->getType()->getPointerElementType(), gep);
      params[i].EmitWriteArrayElement(keys_index, value, b);
    });
  }
  // We should normally synchronize here to make sure all writes have happened.
  // However the very next thing each thread does is reading 2 elements from the
  // operand buffer and writing it into the same location in shared memory from
  // which it previously copied it to the operand buffer, and we synchronize
  // after this has happened. We can be sure that a thread always writes to the
  // same location in shared memory because we have exactly tile_size / 2 many
  // threads, and the linear index calculated by ParallelLoopEmitter uses
  // linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  return Status::OK();
}
}  // namespace

Status EmitSortInPlace(
    int64_t dimension_to_sort, const std::vector<IrArray>& values_arrays,
    absl::string_view name, absl::Span<const int64_t> xor_masks,
    llvm::IRBuilder<>* b, const gpu::LaunchDimensions& launch_dimensions,
    int64_t num_iterations_in_sort_dim, const int64_t tile_size,
    const EmitCallToNestedComputationCallback& emit_compare_callback) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc mht_7(mht_7_v, 500, "", "./tensorflow/compiler/xla/service/llvm_ir/sort_util.cc", "EmitSortInPlace");

  // Iterate through the keys shape in physical order, but skip the dimension to
  // sort and make it the innermost loop which is the loop where the comparisons
  // happen. In the dimension to sort, if we use tiling, we iterate through it
  // in tiles of 64 elements each, so we use another loop that happens within
  // one thread to process this tile worth of data (thereby combining several
  // comparison stages of the bitonic sort algorithm because they all happen
  // within those 64 elements and are therefore independent of the other
  // comparisons).

  const Shape& keys_shape = values_arrays[0].GetShape();
  int64_t rank = keys_shape.rank();
  int64_t dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  std::vector<int64_t> dimensions_in_iteration_order(rank);
  std::vector<int64_t> iteration_order_to_logical_order(rank);
  int64_t dim = 0;
  for (int64_t dimension : LayoutUtil::MinorToMajor(keys_shape)) {
    if (dimension != dimension_to_sort) {
      dimensions_in_iteration_order[dim] = keys_shape.dimensions(dimension);
      iteration_order_to_logical_order[dim++] = dimension;
    }
  }
  dimensions_in_iteration_order[dim] = num_iterations_in_sort_dim;
  iteration_order_to_logical_order[dim] = dimension_to_sort;

  Shape iteration_shape = ShapeUtil::MakeShape(keys_shape.element_type(),
                                               dimensions_in_iteration_order);

  // Allocate shared memory for the tiled compare loop.
  std::vector<llvm::Value*> param_shmem_buffers(values_arrays.size(), nullptr);
  if (xor_masks.size() > 1) {
    llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
    for (int64_t i = 0; i < values_arrays.size(); ++i) {
      llvm::Type* tile_type = llvm::ArrayType::get(
          llvm_ir::PrimitiveTypeToIrType(
              values_arrays[i].GetShape().element_type(), module),
          tile_size);
      param_shmem_buffers[i] = llvm_ir::AllocateSharedMemoryTile(
          module, tile_type, absl::StrCat(name, "_tile_param_", i));
    }
  }

  auto compare_loop_body_emitter =
      [&](const IrArray::Index& tiles_index) -> Status {
    // Naive C++ code for the inner compare loop:
    //
    // for (int64_t i = 0; i < dimension_to_sort_bound; ++i) {
    //   int64_t j = i ^ xor_mask;
    //   /* emitted in EmitCompareLoopBody() */
    //   if (i < j && j < dimension_to_sort_bound) {
    //     int64_t min_key = std::min(keys[i], keys[j]);
    //     keys[j] = std::max(keys[i], keys[j]);
    //     keys[i] = min_key;
    //   }
    // }
    //
    // This follows the algorithm described on Wikipedia:
    // https://en.wikipedia.org/wiki/Bitonic_sorter
    std::vector<llvm::Value*> keys_multi_index(rank);
    for (int64_t i = 0; i < rank; ++i) {
      keys_multi_index[iteration_order_to_logical_order[i]] = tiles_index[i];
    }
    if (xor_masks.size() > 1) {
      IrArray::Index keys_index(keys_multi_index, values_arrays[0].GetShape(),
                                tiles_index.GetType());
      TF_RETURN_IF_ERROR(EmitTiledCompareLoop(
          keys_index, dimension_to_sort, dimension_to_sort_bound, xor_masks,
          values_arrays, param_shmem_buffers, tile_size, emit_compare_callback,
          b));
    } else {
      auto element_address = [&](int64_t operand, llvm::Value* index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc mht_8(mht_8_v, 573, "", "./tensorflow/compiler/xla/service/llvm_ir/sort_util.cc", "lambda");

        keys_multi_index[dimension_to_sort] = index;
        IrArray::Index keys_index(keys_multi_index,
                                  values_arrays[operand].GetShape(),
                                  tiles_index.GetType());
        return values_arrays[operand].EmitArrayElementAddress(keys_index, b);
      };
      auto write_element = [&](int64_t operand, llvm::Value* index,
                               llvm::Value* value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSsort_utilDTcc mht_9(mht_9_v, 584, "", "./tensorflow/compiler/xla/service/llvm_ir/sort_util.cc", "lambda");

        keys_multi_index[dimension_to_sort] = index;
        IrArray::Index keys_index(keys_multi_index,
                                  values_arrays[operand].GetShape(),
                                  tiles_index.GetType());
        values_arrays[operand].EmitWriteArrayElement(keys_index, value, b);
      };
      TF_RETURN_IF_ERROR(EmitCompareLoopBody(
          dimension_to_sort_bound, values_arrays.size(), tiles_index[rank - 1],
          xor_masks[0], tiles_index.GetType(), element_address, write_element,
          emit_compare_callback, b));
    }
    return Status::OK();
  };
  return gpu::ParallelLoopEmitter(compare_loop_body_emitter, iteration_shape,
                                  launch_dimensions, b)
      .EmitLoop(name);
}

}  // namespace llvm_ir
}  // namespace xla
