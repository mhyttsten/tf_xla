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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file contains the analysis and transformation to rewrite kernel
// functions such that they use a single set of arguments for the strides and
// sizes of operands with equal shapes.

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

#define DEBUG_TYPE "kernel-gen-shapes"

namespace {

using mlir::ArrayRef;
using mlir::SmallVector;
using mlir::Value;

/// Represents a value or constant. Used to unify operands for operations that
/// take both ssa values and attributes.
struct ValueOrConst {
  explicit ValueOrConst(Value v) : value_or_constant(v), is_constant(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_0(mht_0_v, 220, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "ValueOrConst");
}
  explicit ValueOrConst(int64_t c) : value_or_constant(c), is_constant(true) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "ValueOrConst");
}

  Value value() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "value");

    assert(!is_constant);
    return value_or_constant.value;
  }

  int64_t constant() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_3(mht_3_v, 237, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "constant");

    assert(is_constant);
    return value_or_constant.constant;
  }

  bool isConstant() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_4(mht_4_v, 245, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "isConstant");
 return is_constant; }

 private:
  union ValueOrConstStorage {
    explicit ValueOrConstStorage(Value v) : value(v) {}
    explicit ValueOrConstStorage(size_t c) : constant(c) {}

    Value value;
    int64_t constant;
  } value_or_constant;

  bool is_constant;
};

llvm::hash_code hash_value(ValueOrConst value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_5(mht_5_v, 262, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "hash_value");

  return value.isConstant() ? static_cast<llvm::hash_code>(value.constant())
                            : mlir::hash_value(value.value());
}

bool operator==(ValueOrConst lhs, ValueOrConst rhs) {
  if (lhs.isConstant()) {
    return rhs.isConstant() && lhs.constant() == rhs.constant();
  } else {
    return !rhs.isConstant() && lhs.value() == rhs.value();
  }
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ValueOrConst &value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_6(mht_6_v, 279, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "operator<<");

  if (value.isConstant()) {
    os << value.constant();
  } else {
    Value val = value.value();
    mlir::AsmState asm_state(
        val.getParentRegion()->getParentOfType<mlir::func::FuncOp>());
    val.printAsOperand(os, asm_state);
  }
  return os;
}

/// Represents a shape, as either a single SSA value that represents the entire
/// shape vector or as a vector of SSA values representing scalars.
struct ShapeValue {
  explicit ShapeValue(Value vector)
      : shape({ValueOrConst{vector}}), is_vector(true) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_7(mht_7_v, 298, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "ShapeValue");
}
  explicit ShapeValue(ValueOrConst vector) : shape({vector}), is_vector(true) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_8(mht_8_v, 302, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "ShapeValue");

    assert(!vector.isConstant());
  }
  template <typename T>
  explicit ShapeValue(T values)
      : shape(values.begin(), values.end()), is_vector(false) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_9(mht_9_v, 310, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "ShapeValue");
}

  ValueOrConst vector() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_10(mht_10_v, 315, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "vector");

    assert(is_vector);
    return shape.front();
  }

  ArrayRef<ValueOrConst> scalars() const {
    assert(!is_vector);
    return llvm::makeArrayRef(shape);
  }

  bool isVector() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_11(mht_11_v, 328, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "isVector");
 return is_vector; }

 private:
  SmallVector<ValueOrConst, 4> shape;
  bool is_vector;
};

llvm::hash_code hash_value(ShapeValue shape) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_12(mht_12_v, 338, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "hash_value");

  return shape.isVector() ? hash_value(shape.vector())
                          : hash_value(shape.scalars());
}

bool operator==(ShapeValue lhs, ShapeValue rhs) {
  if (lhs.isVector()) {
    return rhs.isVector() && lhs.vector() == rhs.vector();
  } else {
    return !rhs.isVector() && lhs.scalars() == rhs.scalars();
  }
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ShapeValue &shape) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_13(mht_13_v, 355, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "operator<<");

  if (shape.isVector()) {
    os << shape.vector();
    return os;
  }
  os << "[";
  bool first = true;
  for (auto scalar : shape.scalars()) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << scalar;
  }
  os << "]";
  return os;
}

}  // namespace

namespace llvm {

template <>
struct DenseMapInfo<ShapeValue> {
  static ShapeValue getEmptyKey() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_14(mht_14_v, 382, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "getEmptyKey");

    return ShapeValue(DenseMapInfo<mlir::Value>::getEmptyKey());
  }
  static ShapeValue getTombstoneKey() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_15(mht_15_v, 388, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "getTombstoneKey");

    return ShapeValue(DenseMapInfo<mlir::Value>::getTombstoneKey());
  }
  static unsigned getHashValue(ShapeValue shape) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_16(mht_16_v, 394, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "getHashValue");
 return hash_value(shape); }
  static bool isEqual(ShapeValue LHS, ShapeValue RHS) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_17(mht_17_v, 398, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "isEqual");
 return LHS == RHS; }
};

}  // namespace llvm

namespace mlir {
namespace kernel_gen {
namespace transforms {

namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

// A basic shape equality inference. This should be superceeded by a proper
// inference once available. Until then, we just build this out to the needs of
// the kernel generator project.
class ShapeEqualityKnowledge {
 public:
  /// Checks all operations for potential shape equality of their respective
  /// results.
  void build(FuncOp function) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_18(mht_18_v, 422, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "build");

    function.walk([&](Operation *op) {
      if (auto reshape = dyn_cast<memref::ReshapeOp>(op)) {
        registerAssociation(ShapeValue{reshape.shape()}, reshape.result());
        return;
      }
      if (auto cast = dyn_cast<memref::ReinterpretCastOp>(op)) {
        // Only support fully dynamic sizes for now.
        // TODO(herhut): Fix once the op has canonicalizers that break this.
        for (unsigned int p = 0, e = cast.getResultRank(); p < e; ++p) {
          if (!cast.isDynamicSize(p)) {
            return;
          }
        }
        registerAssociation(ShapeValue{cast.sizes()}, cast.result());
        return;
      }
      if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
        SmallVector<ValueOrConst, 4> shape;
        ShapedType type = alloc.getResult().getType().cast<ShapedType>();
        fillShapeFromAllocLike(alloc.getDynamicSizes(), type, shape);
        registerAssociation(ShapeValue{shape}, alloc.getResult());
        return;
      }
      if (auto alloc = dyn_cast<tf_framework::TFAllocOp>(op)) {
        // Construct a symbol representing the allocated shape.
        SmallVector<ValueOrConst, 4> shape;
        ShapedType type = alloc.getResult().getType().cast<ShapedType>();
        fillShapeFromAllocLike(alloc.dyn_sizes(), type, shape);
        registerAssociation(ShapeValue{shape}, alloc.getResult());
        return;
      }
    });
  }

  /// Checks whether `one` and `other` are known to have the same shape and
  /// strides.
  bool haveSameShape(Value one, Value other) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_19(mht_19_v, 462, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "haveSameShape");

    return equal_shapes_.isEquivalent(one.getAsOpaquePointer(),
                                      other.getAsOpaquePointer());
  }

 private:
  static void fillShapeFromAllocLike(mlir::OperandRange operands,
                                     ShapedType type,
                                     SmallVectorImpl<ValueOrConst> &shape) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_20(mht_20_v, 473, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "fillShapeFromAllocLike");

    assert(type.hasRank());
    auto dynamic_sizes = operands.begin();
    for (auto extent : type.getShape()) {
      shape.push_back(ShapedType::isDynamic(extent)
                          ? ValueOrConst{*(dynamic_sizes++)}
                          : ValueOrConst{extent});
    }
  }

  /// Registers the value `value` to have the shape represented by `shape`. If
  /// `shape` has been registered before, place `value` into the same
  /// equivalence class. Otherwise register `value` as an equivalence class of
  /// its own.
  void registerAssociation(ShapeValue shape, Value value) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_21(mht_21_v, 490, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "registerAssociation");

    LLVM_DEBUG({ llvm::dbgs() << "Processing " << value << "\n"; });
    auto insert_symbolic = symbolic_shapes_.insert({shape, value});
    if (insert_symbolic.second) {
      LLVM_DEBUG({ llvm::dbgs() << "New symbolic shape " << shape << "\n"; });
      equal_shapes_.insert(value.getAsOpaquePointer());
      // We have seen this symbolic shape for the first time. Try to match it
      // with a vector or shape we already know and alias classes if possible.
      // This could be based on shape dialect if we weren't late in the
      // lowering.
      tryEvaluateShapeToRoot(shape, value);
    } else {
      auto rep = insert_symbolic.first->second;
      LLVM_DEBUG({ llvm::dbgs() << "Aliasing with rep " << rep << "\n"; });
      equal_shapes_.unionSets(rep.getAsOpaquePointer(),
                              value.getAsOpaquePointer());
    }
  }

  /// Follows the definition chains of the ShapeValue `shape` to identify cases
  /// where `shape` is derived from some other value's shape. In such case, the
  /// equivalence classes of that other value and `value` are unioned.
  /// This is based on pattern matching and not complete.
  void tryEvaluateShapeToRoot(ShapeValue shape, Value value) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_22(mht_22_v, 516, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "tryEvaluateShapeToRoot");

    // Just some pattern matching for common cases here.
    if (!shape.isVector()) {
      // Patterns that revolve around scalars.
      // Check whether the scalars are all dim operations for some other memref.
      Value candidate;
      bool all_are_dimops =
          llvm::all_of(llvm::enumerate(shape.scalars()), [&candidate](auto p) {
            ValueOrConst val = p.value();
            if (val.isConstant()) return false;
            auto dimOp = val.value().getDefiningOp<memref::DimOp>();
            if (!dimOp) return false;
            if (!candidate) candidate = dimOp.source();
            auto index = dimOp.getConstantIndex();
            if (!index.hasValue()) return false;
            return candidate == dimOp.source() && p.index() == index.getValue();
          });
      if (all_are_dimops && candidate) {
        equal_shapes_.unionSets(candidate.getAsOpaquePointer(),
                                value.getAsOpaquePointer());
      }
    }
  }

  // These are values with identical shapes (or rather their opaque pointers).
  llvm::EquivalenceClasses<void *> equal_shapes_;
  // A map from a value that encodes a shape to a value that has this shape.
  llvm::DenseMap<ShapeValue, Value> symbolic_shapes_;
};

/// For arguments to kernels that have the same shape, use the stride and
/// shape information of the left-most argument inside of the kernel function.
/// That way, llvm can CSE index computations on same-shaped inputs.
struct PropagateShapeKnowledgeToKernels
    : public PropagateShapeKnowledgeToKernelsBase<
          PropagateShapeKnowledgeToKernels> {
  void runOnOperation() override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSsame_shape_propagationDTcc mht_23(mht_23_v, 555, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/same_shape_propagation.cc", "runOnOperation");

    ShapeEqualityKnowledge knowledge;

    knowledge.build(getOperation());

    getOperation().walk([&](gpu::LaunchFuncOp launch) {
      auto module = launch->getParentOfType<ModuleOp>();
      auto kernel = module.lookupSymbol<LLVM::LLVMFuncOp>(launch.kernel());

      if (!kernel || kernel.isExternal()) return;

      llvm::SmallVector<std::pair<Value, int>, 4> seen_memrefs;
      // Position of the kernel argument we are currently at.
      int kernel_p = 0;
      for (auto operand : launch.operands()) {
        auto memref = operand.getType().dyn_cast<MemRefType>();
        if (!memref) {
          // Scalar argument, advance kernel position by one.
          kernel_p++;
          continue;
        }
        for (auto previous : seen_memrefs) {
          if (!knowledge.haveSameShape(operand, previous.first)) {
            continue;
          }
          auto previous_type = previous.first.getType().cast<MemRefType>();
          // We use the first equality found and replace uses of corresponding
          // size and (potentially) stride information here.
          auto args_to_replace = memref.getRank();
          // If both memrefs have identity layouts, we can also reuse the
          // strides here, as they are the identity strides and hence fully
          // determinded by the shape.
          if (previous_type.getLayout().isIdentity() &&
              memref.getLayout().isIdentity()) {
            args_to_replace *= 2;
          }
          int previous_args_pos = previous.second;
          auto previous_args = kernel.getArguments()
                                   .drop_front(previous_args_pos + 3)
                                   .take_front(args_to_replace);
          auto current_args = kernel.getArguments()
                                  .drop_front(kernel_p + 3)
                                  .take_front(args_to_replace);
          for (auto pair : llvm::zip(previous_args, current_args)) {
            mlir::BlockArgument prev, curr;
            std::tie(prev, curr) = pair;
            curr.replaceAllUsesWith(prev);
          }
          break;
        }
        seen_memrefs.push_back({operand, kernel_p});
        // Advance base, aligned, offset, strides and sizes many arguments.
        kernel_p += memref.getRank() * 2 + 3;
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreatePropagateShapeKnowledgeToKernels() {
  return std::make_unique<PropagateShapeKnowledgeToKernels>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
