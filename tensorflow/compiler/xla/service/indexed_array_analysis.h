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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_INDEXED_ARRAY_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_INDEXED_ARRAY_ANALYSIS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh() {
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


#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/core/util/ptr_util.h"

namespace xla {

// IndexedArrayAnalysis decides if an HLO instruction can be rewritten as a
// gather from another array.  It does this by mapping HLO instructions to
// instances of IndexedArrayAnalysis::Array, which can be inspected to discover
// whether said HLO is equivalent to a gather.
class IndexedArrayAnalysis {
 public:
  // IndexedArrayAnalysis maps each HLO instruction to an instance of a Array.
  // Array really just a sum type of the classes that inherit from it.  The
  // meaning of each of the subtypes is documented on the subtype declaration.
  //
  // Array instances are immutable once created.
  class Array {
   public:
    enum Kind {
      kUnknown,
      kConstant,
      kReshaped,
      kScalarIndexedConstant,
      kScalarIndexed
    };

    virtual Kind kind() const = 0;
    virtual const Shape& shape() const = 0;

    // Does a checked downcast from `Array` to `T` which must be one of its
    // subtypes.
    template <typename T>
    T* as() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_0(mht_0_v, 225, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "as");

      static_assert((std::is_base_of<Array, T>::value),
                    "target type not derived from source type");
      // We skip the CHECK and hence the dynamic_cast if RTTI is disabled.
#if !defined(__GNUC__) || defined(__GXX_RTTI)
      CHECK_NE(dynamic_cast<T*>(this), nullptr);
#endif  // !defined(__GNUC__) || defined(__GXX_RTTI)

      return static_cast<T*>(this);
    }

    virtual ~Array() = default;

    Array& operator=(const Array& other) = delete;
  };

  // Represents an HLO instruction that was not analyzable by this
  // IndexedArrayAnalysis.  Instances of UnknownArray just wrap an existing
  // HloInstruction.
  class UnknownArray : public Array {
   public:
    Kind kind() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_1(mht_1_v, 249, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "kind");
 return kUnknown; }
    const Shape& shape() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_2(mht_2_v, 253, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "shape");
 return instruction().shape(); }
    const HloInstruction& instruction() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_3(mht_3_v, 257, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "instruction");
 return instruction_; }

   private:
    explicit UnknownArray(const HloInstruction* instr) : instruction_(*instr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_4(mht_4_v, 263, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "UnknownArray");
}

    const HloInstruction& instruction_;

    friend class IndexedArrayAnalysis;
  };

  // Represents a constant value.  This constant value may be present in the HLO
  // module being analyzed, or it could have been created on the fly by the
  // analysis.
  class ConstantArray : public Array {
   public:
    Kind kind() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_5(mht_5_v, 278, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "kind");
 return kConstant; }
    const Shape& shape() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_6(mht_6_v, 282, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "shape");
 return literal()->shape(); }
    const Literal* literal() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_7(mht_7_v, 286, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "literal");
 return literal_; }

   private:
    explicit ConstantArray(const Literal* literal) : literal_(literal) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_8(mht_8_v, 292, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "ConstantArray");
}
    const Literal* literal_;

    friend class IndexedArrayAnalysis;
  };

  // Represents an Array that is a reshape of another Array.
  class ReshapedArray : public Array {
   public:
    Kind kind() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_9(mht_9_v, 304, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "kind");
 return kReshaped; }

    // The array to reshape.
    Array* operand() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_10(mht_10_v, 310, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "operand");
 return operand_; }

    // The output shape.
    const Shape& shape() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_11(mht_11_v, 316, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "shape");
 return shape_; }

   private:
    explicit ReshapedArray(Array* operand, Shape shape)
        : operand_(operand), shape_(shape) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_12(mht_12_v, 323, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "ReshapedArray");
}

    Array* operand_;
    const Shape shape_;

    friend class IndexedArrayAnalysis;
  };

  // ---------------------------------------------------------------------------
  // Indexed Array Overview
  // ---------------------------------------------------------------------------
  //
  // ScalarIndexedArray and ScalarIndexedConstantArray form the core of this
  // analysis.  ScalarIndexedConstantArray is just a specialization of
  // ScalarIndexedArray so we will only discuss ScalarIndexedArray in this
  // overview.
  //
  // A ScalarIndexedArray represents an array that can be computed by indexing
  // into a "source" array using an "indices" tensor.  A simple example is a
  // gather operation gathering 12 rows out of a [100,100] matrix -- such an
  // operation will be represented by an instance of a ScalarIndexedArray with
  // the [100,100] matrix as the "source" array and the [12]-shaped indices
  // array as the "indices" tensor.  The ScalarIndexedArray operation itself
  // will be of shape [12,100] (assuming we were gathering with axis=0).
  //
  // Gather operations are not the only operation that maps to
  // ScalarIndexedArray instances (if that were true there would be little point
  // in having a separate analysis).  We can often infer ScalarIndexedArrays for
  // other operations too.  For instance, consider:
  //
  //   %source = f32[100,100] constant
  //   %indices = s32[12] ...
  //   %gather = f32[12,100] ... gather from %source using %indices at axis 0
  //   %dot = dot(%gather, other_constant) [canonical contracting dims]
  //
  // The dot operation itself is also a ScalarIndexedArray with source =
  // dot(constant, other_constant) and indices = %indices.  A reshape of %gather
  // to [12,5,20] too is a ScalarIndexedArray with source = an appropriately
  // reshaped constant and indices = %indices.

  // Represents the result of a gather operation.  This gather operation may
  // explicitly be present in the HLO module being analyzed, or it could have
  // been created on the fly by the analysis.
  //
  // An instance of ScalarIndexedArray represents a array whose I'th element can
  // be mapped to the J'th element of the `source` array (where I and J are
  // multidimensional indices) in this way:
  //
  //   I' = remove components at positions `output_dims` from I
  //   G' = remove components not at positions `output_dims` from I
  //   T  = indices[G']
  //   J  = I' with T inserted at position `source_dim`
  //
  // For example, if source is of shape [11,13,17,19], indices is of shape
  // [23,29], output_dims is [0,2] and source_dim is 2 then the output is of
  // shape [23,11,29,13,19] and the output index [A,B,C,D,E] is mapped to the
  // input index [B,D,indices[A,C],E].
  class ScalarIndexedArray : public Array {
   public:
    Kind kind() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_13(mht_13_v, 385, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "kind");
 return kScalarIndexed; }
    const Shape& shape() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_14(mht_14_v, 389, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "shape");
 return shape_; }

    Array* source() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_15(mht_15_v, 394, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "source");
 return source_; }
    Array* indices() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_16(mht_16_v, 398, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "indices");
 return indices_; }

    // `source_dim` is the dimension in the source array that is being indexed
    // over using indices from the `indices` array.  See the class documentation
    // and the overview for more details.
    int64_t source_dim() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_17(mht_17_v, 406, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "source_dim");
 return source_dim_; }

    // `output_dims` are the dimensions in the output array that are being used
    // to compute an index into the `indices` array.  See the class
    // documentation and the overview for more details.
    absl::Span<const int64_t> output_dims() const { return output_dims_; }

   private:
    explicit ScalarIndexedArray(Array* source, Array* indices,
                                int64_t source_dim,
                                std::vector<int64_t> output_dims, Shape shape)
        : source_(source),
          indices_(indices),
          source_dim_(source_dim),
          output_dims_(std::move(output_dims)),
          shape_(std::move(shape)) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_18(mht_18_v, 424, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "ScalarIndexedArray");
}

    Array* source_;
    Array* indices_;
    int64_t source_dim_;
    std::vector<int64_t> output_dims_;
    Shape shape_;

    friend class IndexedArrayAnalysis;
  };

  // A ScalarIndexedConstantArray is just a ScalarIndexedArray constrained to
  // have a ConstantArray instance as the source.  This is an ergonomic
  // concession -- in theory it is possible to just keep ScalarIndexedArray and
  // check source()->kind().
  class ScalarIndexedConstantArray : public ScalarIndexedArray {
   public:
    Kind kind() const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_19(mht_19_v, 444, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "kind");
 return kScalarIndexedConstant; }

    const Literal& literal() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_20(mht_20_v, 449, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "literal");

      return *source()->as<ConstantArray>()->literal();
    }

   private:
    explicit ScalarIndexedConstantArray(Array* source, Array* indices,
                                        int64_t source_dim,
                                        std::vector<int64_t> output_dims,
                                        Shape shape)
        : ScalarIndexedArray(source, indices, source_dim,
                             std::move(output_dims), std::move(shape)) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_21(mht_21_v, 462, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "ScalarIndexedConstantArray");

      CHECK(dynamic_cast<ConstantArray*>(source));
    }

    friend class IndexedArrayAnalysis;
  };

  // Returns an Array instance for `instr`.  The IndexedArrayAnalysis instance
  // keeps ownership of the returned Array instance.
  //
  // Caching Behavior: IndexedArrayAnalysis has a cache mapping HLO
  // instructions to IndexedArrayAnalysis::Array instances.  This entire cache
  // becomes stale and may cause the analysis to return incorrect results if any
  // transitive operand (stopping at the containing computation) is modified for
  // any HLO instruction on which GetArrayFor has been invoked.
  //
  // NB!  By inspecting the implementation, you may be able to infer a stronger
  // caching guarantee than what is mentioned above.  Nevertheless, what is
  // stated above is the contract.
  StatusOr<Array*> GetArrayFor(const HloInstruction* instr);

  // Pretty-prints the expression rooted at `root`.
  std::string ToString(Array* root, bool print_constants = false);

 private:
  // Helper function that ensures that every HLO instruction that is
  // transitively used by `root` has an entry in `cache_`.
  Status TraverseAndPopulateCache(const HloInstruction* root);

  // Creates an Array instance for `instr` under the assumption that all
  // operations of `instr` are present in `cache_`.
  StatusOr<Array*> ComputeArrayFor(const HloInstruction* instr);

  StatusOr<Array*> ComputeArrayForConstant(const Literal& literal);

  StatusOr<Array*> ComputeArrayForGather(
      const Shape& shape, const GatherDimensionNumbers& dim_numbers,
      absl::Span<const int64_t> slice_sizes, Array* source, Array* indices);

  StatusOr<Array*> ComputeArrayForDotWithIndexedLhs(
      const Shape& shape, const DotDimensionNumbers& dim_numbers,
      const PrecisionConfig& precision_config, ScalarIndexedConstantArray* lhs,
      ConstantArray* rhs);

  StatusOr<Array*> ComputeArrayForDotWithIndexedRhs(
      const Shape& shape, const DotDimensionNumbers& dim_numbers,
      const PrecisionConfig& precision_config, ConstantArray* lhs,
      ScalarIndexedConstantArray* rhs);

  StatusOr<Array*> ComputeArrayForDot(const Shape& shape,
                                      const DotDimensionNumbers& dim_numbers,
                                      const PrecisionConfig& precision_config,
                                      Array* lhs, Array* rhs);

  // This tries to fold a ScalarIndexedArray which has another
  // ScalarIndexedArray as a source into a ScalarIndexedArray that instead has a
  // ScalarIndexedArray as indices.  If `source` happened to be a
  // ScalarIndexedConstantArray this can result in an expression that is more
  // canonical.
  //
  // As an example, consider a gather operation, G0, gathering 7 elements from
  // an array "Arr" of shape [100] resulting in an array of shape [7], and a
  // second gather operation, G1, which gathers 3 elements out of the result of
  // G0 resulting in an array of shape [3].  Let the indices uses by G0 be I0
  // (of shape [7]) and the indices used by G1 be I1 (of shape [3]).  We can
  // instead rewrite G1 to gather directly from "Arr" with the three indices
  // from I0 as per I1.  In other words, we can rewrite:
  //
  //    G0 = [Arr[i] for i in I0]
  //    G1 = [G0[i]  for i in I1]
  //
  // into
  //
  //    I2 = [I0[i]  for i in I1]
  //    G1 = [Arr[i] for i in I2]
  StatusOr<ScalarIndexedArray*> FoldGatherOfGather(
      ScalarIndexedArray* source, Array* indices, int64_t source_dim,
      absl::Span<const int64_t> output_dims, Shape shape);

  // Reshapes a scalar-indexed node to remove the degenerate dimensions in its
  // output.  The result is always a scalar-indexed node.
  StatusOr<ScalarIndexedArray*> ReshapeToRemoveDegenerateDims(
      ScalarIndexedArray* operand);

  // Reshapes a scalar-indexed node such that the result has the degenerate
  // dimensions `degenerate_dims`.  The result is always a scalar-indexed node.
  StatusOr<ScalarIndexedArray*> ReshapeToAddDegenerateDims(
      ScalarIndexedArray* operand, absl::Span<const int64_t> degenerate_dims);

  StatusOr<ScalarIndexedArray*> FoldReshapeOfGather(
      const Shape& shape, ScalarIndexedConstantArray* operand);
  StatusOr<ScalarIndexedArray*> FoldReshapeOfGatherNoDegenerateDims(
      const Shape& shape, ScalarIndexedConstantArray* scalar_indexed);
  StatusOr<Array*> ComputeArrayForReshape(const Shape& shape, Array* operand);

  StatusOr<Array*> ComputeArrayForElementwiseBinaryOp(HloOpcode opcode,
                                                      Array* lhs, Array* rhs);
  StatusOr<Array*> ComputeArrayForElementwiseUnaryOp(HloOpcode opcode,
                                                     Array* operand);

  template <typename T, typename... Args>
  T* Construct(Args&&... args) {
    T* new_tensor = new T(std::forward<Args>(args)...);
    owned_tensors_.push_back(std::unique_ptr<T>(new_tensor));
    return new_tensor;
  }

  ScalarIndexedArray* ConstructScalarIndexedArray(
      Array* source, Array* indices, int64_t source_dim,
      std::vector<int64_t> output_dims, Shape shape) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_22(mht_22_v, 574, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "ConstructScalarIndexedArray");

    if (source->kind() == Array::kConstant) {
      return Construct<ScalarIndexedConstantArray>(source, indices, source_dim,
                                                   std::move(output_dims),
                                                   std::move(shape));
    } else {
      return Construct<ScalarIndexedArray>(source, indices, source_dim,
                                           std::move(output_dims),
                                           std::move(shape));
    }
  }

  Literal* TakeOwnership(Literal literal) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSindexed_array_analysisDTh mht_23(mht_23_v, 589, "", "./tensorflow/compiler/xla/service/indexed_array_analysis.h", "TakeOwnership");

    owned_literals_.push_back(std::move(literal));
    return &owned_literals_.back();
  }

  StatusOr<Literal*> TakeOwnership(StatusOr<Literal> literal_or_error) {
    TF_ASSIGN_OR_RETURN(Literal literal, std::move(literal_or_error));
    owned_literals_.push_back(std::move(literal));
    return &owned_literals_.back();
  }

  std::vector<std::unique_ptr<Array>> owned_tensors_;
  std::vector<Literal> owned_literals_;
  absl::flat_hash_map<const HloInstruction*, Array*> cache_;
};

// A pass that prints all non-trivial results returned by IndexedArrayAnalysis.
// This pass is a no-op if !VLOG_IS_ON(2) so it should be fine to
// unconditionally add to the regular HLO pass pipeline.
class IndexedArrayAnalysisPrinterPass : public HloModulePass {
 public:
  absl::string_view name() const override;
  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_INDEXED_ARRAY_ANALYSIS_H_
