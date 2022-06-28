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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_IR_ARRAY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_IR_ARRAY_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh() {
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


#include <map>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace llvm_ir {

// IrArray represents an XLA array at the LLVM IR level. This class
// encapsulates a base pointer to the buffer holding the array (as an LLVM
// Value) and the shape of the array. The class includes methods for emitting
// LLVM IR sequences which access elements of the array at a multidimensional
// index (eg, [x, y, z] in a 3-dimensional array). Arbitrary shape and layouts
// are supported.
class IrArray {
 public:
  // A multidimensional index into an IrArray. All the runtime indices
  // (multidim) and dimensions (Shape::dimensions(), absl::Span<const int64_t>)
  // are major-first.
  //
  // This may also keep a linear index and the layout and dimensions it was
  // emitted for; if the shape where this `Index` is used matches, the linear
  // index may be used, potentially sparing the cost of computing the
  // multidimensional index, which LLVM DCE can delete.
  class Index {
   public:
    // Constructs an index for a scalar shape.
    explicit Index(llvm::Type* index_ty) : index_type_(index_ty) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_0(mht_0_v, 225, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "Index");

      CHECK(index_ty->isIntegerTy());
    }

    // Constructs an index from linear index "linear" and computes the
    // multi-dimensional index from "linear" and "shape". "b" is the IR
    // builder to emit the index of each dimension in the multi-dimensional
    // index.
    //
    // Precondition: "shape" has a layout.
    Index(llvm::Value* linear, const Shape& shape, llvm::IRBuilder<>* b);

    // As before, but also take a multidim to reuse.  multidim.size()
    // == shape.rank() must be true.  If some of the multidim element
    // are null we will use the value that would be used if
    // deliearized from linear.
    Index(llvm::Value* linear, absl::Span<llvm::Value* const> multidim,
          const Shape& shape, llvm::IRBuilder<>* b);

    // Similar to the above constructor except using "dynamic_dims" instead of
    // shape's static dimension to constructs the index.
    Index(llvm::Value* linear, const Shape& shape,
          absl::Span<llvm::Value*> dynamic_dims, llvm::IRBuilder<>* b);

    // Constructs an index from a multi-dimensional index. 'shape' is the shape
    // for which the multi-dimensional index is used. 'index_type' is the type
    // of the index.
    //
    // Precondition: "shape" has a layout.
    Index(absl::Span<llvm::Value* const> multidim, const Shape& shape,
          llvm::Type* index_type);

    // Same as above, but only the dimensions of the shape without layout is
    // passed. The layout is assumed to be the default (descending
    // minor-to-major) layout.
    Index(absl::Span<llvm::Value* const> multidim,
          absl::Span<int64_t const> dimensions, llvm::Type* index_type);

    // Returns an index that adds `addend` to the given `dim` of the object.
    Index AddOffsetToDim(llvm::Value* addend, int64_t dim,
                         llvm::IRBuilder<>* b) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_1(mht_1_v, 268, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "AddOffsetToDim");

      Index with_offset = *this;
      with_offset.linear_ = nullptr;
      with_offset.multidim_[dim] =
          b->CreateAdd(with_offset.multidim_[dim], addend);
      return with_offset;
    }

    const std::vector<llvm::Value*>& multidim() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_2(mht_2_v, 279, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "multidim");
 return multidim_; }
    const std::vector<int64_t>& dims() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_3(mht_3_v, 283, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "dims");
 return dims_; }
    llvm::Value* linear() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_4(mht_4_v, 287, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "linear");
 return linear_; }

    size_t size() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_5(mht_5_v, 292, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "size");
 return multidim().size(); }

    llvm::Value* operator[](size_t i) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_6(mht_6_v, 297, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "lambda");
 return multidim()[i]; }

    using const_iterator = std::vector<llvm::Value*>::const_iterator;

    const_iterator begin() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_7(mht_7_v, 304, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "begin");
 return multidim().begin(); }
    const_iterator end() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_8(mht_8_v, 308, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "end");
 return multidim().end(); }

    bool LinearValidOnShape(const Shape& a) const;

    static bool ShapeIsCompatible(const Shape& a, const Shape& b);

    bool ShapeIsCompatible(const Shape& a) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_9(mht_9_v, 317, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "ShapeIsCompatible");

      return ShapeIsCompatible(a, AsShapeWithType(a.element_type()));
    }

    Shape AsShapeWithType(PrimitiveType element_type) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_10(mht_10_v, 324, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "AsShapeWithType");

      return ShapeUtil::MakeShapeWithLayout(element_type, dims_,
                                            layout_.minor_to_major());
    }

    // Given that "this" is the target index of a reshape from `input_shape`
    // to `output_shape`, returns the source index.
    Index SourceIndexOfReshape(const Shape& output_shape,
                               const Shape& input_shape,
                               llvm::IRBuilder<>* builder) const;

    // Returns the index into the source operand from which a slice operation
    // selects a value to be placed into index "this". The slice is described
    // by starting indices `starts` and stride values `strides`.
    //
    // Precondition: "this" is an index into a slice whose operand shape is
    // `operand_shape`.
    Index SourceIndexOfSlice(const Shape& operand_shape,
                             absl::Span<const int64_t> starts,
                             absl::Span<const int64_t> strides,
                             llvm::IRBuilder<>* builder) const;

    // Given that "this" is the target index of a transpose from `operand_shape`
    // to `shape` with the given dimension mapping, returns the source index.
    Index SourceIndexOfTranspose(
        const Shape& shape, const Shape& operand_shape,
        absl::Span<const int64_t> dimension_mapping) const;

    // Given that "this" is the target index of a bitcast from `operand_shape`
    // to `shape`, returns the source index.
    Index SourceIndexOfBitcast(const Shape& shape, const Shape& operand_shape,
                               llvm::IRBuilder<>* builder) const;

    // Given that "this" is the target index of a broadcast from `operand_shape`
    // to `shape` with the given dimension mapping, returns the source index.
    Index SourceIndexOfBroadcast(const Shape& shape, const Shape& operand_shape,
                                 absl::Span<const int64_t> dimension_mapping,
                                 llvm::IRBuilder<>* builder) const;

    // Linearizes the index into the given shape, i.e. reshapes it to rank-1 and
    // returns the index into the sole dimension 0 of the new shape.
    llvm::Value* Linearize(absl::Span<const int64_t> dimensions,
                           llvm::IRBuilder<>* builder) const;

    // Linearizes the index into the given dynamic dimensions.
    llvm::Value* Linearize(const std::vector<llvm::Value*>& dynamic_dims,
                           llvm::IRBuilder<>* builder) const;

    llvm::Type* GetType() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_11(mht_11_v, 375, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "GetType");
 return index_type_; }

    llvm::Constant* GetConstantWithIndexType(int64_t c) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_12(mht_12_v, 380, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "GetConstantWithIndexType");

      // The LLVM function makes sure that the value can be represented by the
      // specified type, see ConstantInt::ConstantInt(IntegerType *Ty, const
      // APInt &V).
      return llvm::ConstantInt::get(index_type_, c);
    }

   private:
    // Constructs an index from both a multi-dimensional index and a linear
    // index. 'shape' is the shape on which the index is used. 'index_type' is
    // the type of the index.
    //
    // Precondition: "shape" has a layout.
    Index(absl::Span<llvm::Value* const> multidim, llvm::Value* linear,
          const Shape& shape, llvm::Type* index_type);

    void Delinearize(std::vector<llvm::Value*>* multidim, llvm::Value* linear,
                     const Shape& shape, llvm::IRBuilder<>* b) const;

    // Delinearize the linear index with the dynamic dimensions.
    void Delinearize(std::vector<llvm::Value*>* multidim, llvm::Value* linear,
                     const Shape& shape, absl::Span<llvm::Value*> dynamic_dims,
                     llvm::IRBuilder<>* b) const;

    std::vector<llvm::Value*> multidim_;

    // These values are purely for efficiency; `multidim_` is enough to find the
    // element at a given `Index`, but if a loop is emitted with a linear index
    // space, that linear index can be saved in `linear_`, and the layout and
    // dimensions of the shape the loop was emitted for in `layout_` and
    // `dims_`, and if the `Index` is used in another array, and its layout and
    // dimensions match, the linear index can be used, sparing the cost of
    // computing `multidim_`, which LLVM DCE could potentially so delete.
    // Modifying `multidim_` after construction nullifies `linear_`, lest it
    // be used wrongly, as it would be valid no more.
    // If a loop is emitted with a multidimensional index space, `linear_` would
    // be null and `layout_` and `dims_` would be ignored.
    llvm::Value* linear_ = nullptr;
    Layout layout_;
    std::vector<int64_t> dims_;

    llvm::Type* index_type_;
  };

  // Default constructor. Constructs an IrArray in a null status.
  IrArray() : base_ptr_(nullptr) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_13(mht_13_v, 428, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "IrArray");
}

  // Construct an IrArray with the given base pointer and shape. base_ptr is a
  // pointer type pointing to the first element(lowest address) of the array.
  IrArray(llvm::Value* base_ptr, Shape shape);

  // Default implementations of copying and moving.
  IrArray(IrArray&& other) = default;
  IrArray(const IrArray& other) = default;
  IrArray& operator=(IrArray&& other) = default;
  IrArray& operator=(const IrArray& other) = default;

  llvm::Value* GetBasePointer() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_14(mht_14_v, 443, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "GetBasePointer");
 return base_ptr_; }
  llvm::Type* GetElementLlvmType() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_15(mht_15_v, 447, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "GetElementLlvmType");
 return element_type_; }

  const Shape& GetShape() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_16(mht_16_v, 452, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "GetShape");
 return shape_; }

  // Emit a sequence of instructions to compute the address of the element in
  // the given array at the given index. Returns the address of the element as
  // an LLVM Value.
  //
  // The optional name is useful for debugging when looking at
  // the emitted LLVM IR.
  llvm::Value* EmitArrayElementAddress(const Index& index, llvm::IRBuilder<>* b,
                                       absl::string_view name = "",
                                       bool use_linear_index = true) const;

  // Attach metadata this IrArray instance knows about to "instruction".
  void AnnotateLoadStoreInstructionWithMetadata(
      llvm::Instruction* instruction) const;

  // Emit IR to read an array element at the given index. Returns the read
  // result (effectively, a Value loaded from memory). This method seamlessly
  // handles scalar shapes by broadcasting their value to all indices (index is
  // ignored).
  //
  // The optional name is useful for debugging when looking at
  // the emitted LLVM IR.
  // 'use_linear_index' can be used to specify whether the linear index (if
  // available) or the multi-dimensional index should be used.
  llvm::Value* EmitReadArrayElement(const Index& index, llvm::IRBuilder<>* b,
                                    absl::string_view name = "",
                                    bool use_linear_index = true) const;

  // Emit IR to write the given value to the array element at the given index.
  // 'use_linear_index' can be used to specify whether the linear index (if
  // available) or the multi-dimensional index should be used.
  void EmitWriteArrayElement(const Index& index, llvm::Value* value,
                             llvm::IRBuilder<>* b,
                             bool use_linear_index = true) const;

  // Returns a new IrArray whose shape is "new_shape" and base pointer is a
  // bitcast of the base pointer of "this" IrArray.
  // 'use_linear_index' can be used to specify whether the linear index (if
  // available) or the multi-dimensional index should be used.
  IrArray CastToShape(const Shape& new_shape, llvm::IRBuilder<>* b) const;

  void AddAliasScopeMetadata(llvm::MDNode* alias_scope) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_17(mht_17_v, 497, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "AddAliasScopeMetadata");

    CHECK_NE(alias_scope, nullptr);
    AddMetadata(llvm::LLVMContext::MD_alias_scope, alias_scope);
  }

  void AddNoaliasMetadata(llvm::MDNode* noalias) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_18(mht_18_v, 505, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "AddNoaliasMetadata");

    CHECK_NE(noalias, nullptr);
    AddMetadata(llvm::LLVMContext::MD_noalias, noalias);
  }

  // Promises LLVM that the data pointed to by this IrArray never changes after
  // it's first loaded.
  //
  // The temporal scope of this promise is the "whole program" from LLVM's point
  // of view, but how this translates to HLOs differs between backends.
  //
  // In the single-threaded CPU backend, we emit one function that
  // runs all the HLOs in sequence, so the whole program is the whole HLO
  // module.
  //
  // In the GPU backend, we emit one GPU kernel per top-level HLO (i.e. per HLO
  // in the entry computation).  From LLVM's perspective, launching a new kernel
  // is like launching a new program, and so the whole program is one top-level
  // HLO.  Since the scope of the promise is smaller than in the CPU backend, we
  // can mark more things as invariant in the GPU backend.
  //
  // Marking loads as invariant is particularly helpful on GPUs because
  // invariant loads can be lowered to PTX ld.global.nc (equivalent to CUDA's
  // __ldg intrinsic).  These loads use a special cache, and can be
  // significantly faster than regular loads.
  void MarkInvariantOverWholeProgram(llvm::LLVMContext* context) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_19(mht_19_v, 533, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "MarkInvariantOverWholeProgram");

    if (is_invariant_) {
      return;
    }
    is_invariant_ = true;
    AddMetadata(llvm::LLVMContext::MD_invariant_load,
                llvm::MDNode::get(*context, {}));
  }

  const std::map<int, llvm::MDNode*>& metadata() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_20(mht_20_v, 545, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "metadata");
 return metadata_; }

 private:
  // Add the specified LLVM IR metadata to loads/stores associated with this
  // IrArray.
  void AddMetadata(int kind, llvm::MDNode* md) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTh mht_21(mht_21_v, 553, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.h", "AddMetadata");

    InsertOrDie(&metadata_, kind, md);
  }

  // Address of the base of the array as an LLVM Value.
  llvm::Value* base_ptr_;

  // The LLVM type of the elements in the array.
  llvm::Type* element_type_;

  // Shape of the XLA array.
  Shape shape_;

  // The list of key/value pairs used when attaching metadata to emitted
  // loads/stores for this array.  They keys are the metadata kinds and the
  // values are the metadata nodes.
  std::map<int, llvm::MDNode*> metadata_;

  bool is_invariant_ = false;
};

}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_IR_ARRAY_H_
