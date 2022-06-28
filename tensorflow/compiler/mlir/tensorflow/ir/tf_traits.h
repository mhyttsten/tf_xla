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

// This file defines the op traits used in the MLIR TensorFlow dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TRAITS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TRAITS_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_traitsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_traitsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_traitsDTh() {
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


#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace OpTrait {
namespace TF {

// Verifies if 'ref_type' is a REF type corresponding to 'type'.
static inline LogicalResult VerifyRefTypeMatch(mlir::Type type,
                                               mlir::Type maybe_ref_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_traitsDTh mht_0(mht_0_v, 204, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h", "VerifyRefTypeMatch");

  if (auto ref_type =
          maybe_ref_type.dyn_cast<mlir::tf_type::TensorFlowRefType>())
    return success(ref_type.RemoveRef().getTypeID() == type.getTypeID());
  return failure();
}

// This class provides verification for ops that are known to have the same
// result types and all operands are either of the same type as result or a REF
// type corresponding to the result type.
// TODO(jpienaar): Update the name and the description.
template <typename ConcreteType>
class OperandsSameAsResultsTypeOrRef
    : public TraitBase<ConcreteType, OperandsSameAsResultsTypeOrRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_traitsDTh mht_1(mht_1_v, 222, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h", "verifyTrait");

    LogicalResult shapeMatch = impl::verifySameOperandsAndResultShape(op);
    if (failed(shapeMatch)) return shapeMatch;
    Type type = op->getResult(0).getType();
    // Verify that the first result type is same as the rest of the results.
    // We skip the comparison against itself.
    for (auto result_type : llvm::drop_begin(op->getResultTypes(), 1)) {
      if (!mlir::tf_type::HasCompatibleElementTypes(type, result_type))
        return op->emitOpError()
               << "requires all return types to have compatible element types";
    }
    for (auto operand_type : op->getOperandTypes()) {
      if (!mlir::tf_type::HasCompatibleElementTypes(
              operand_type, type, /*may_ignore_ref_type_lhs=*/true))
        return op->emitError() << "requires all operands and results to have "
                                  "compatible element types";
    }
    return success();
  }
};

namespace detail {
inline LogicalResult verifySameOperandsAndResultElementTypeResolveRef(
    Operation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_traitsDTh mht_2(mht_2_v, 248, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h", "verifySameOperandsAndResultElementTypeResolveRef");

  Type element_type;
  if (op->getNumResults() > 0) {
    element_type = mlir::tf_type::GetElementTypeOrSelfResolveRef(
        op->getResult(0).getType());
  } else if (op->getNumOperands() > 0) {
    element_type = mlir::tf_type::GetElementTypeOrSelfResolveRef(
        op->getOperand(0).getType());
  } else {
    // Nothing to check.
    return success();
  }
  // Verify that all result element types are compatible to `element_type`.
  for (const auto& result_type : op->getResultTypes()) {
    if (mlir::tf_type::GetElementTypeOrSelfResolveRef(result_type) !=
        element_type) {
      return op->emitOpError(
          "requires compatible element types for all operands and results");
    }
  }
  // Verify that all operand element types are compatible to `element_type`.
  for (const auto& operand_type : op->getOperandTypes()) {
    if (mlir::tf_type::GetElementTypeOrSelfResolveRef(operand_type) !=
        element_type) {
      return op->emitOpError(
          "requires compatible element types for all operands and results");
    }
  }
  return success();
}
}  // namespace detail

// Verifies that op has the same operand and result element types (or type
// itself, if scalar) after resolving reference types (i.e., after converting
// reference types to their corresponding TensorFlow or standard types).
template <typename ConcreteType>
class SameOperandsAndResultElementTypeResolveRef
    : public TraitBase<ConcreteType,
                       SameOperandsAndResultElementTypeResolveRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_traitsDTh mht_3(mht_3_v, 291, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h", "verifyTrait");

    return detail::verifySameOperandsAndResultElementTypeResolveRef(op);
  }
};

// Verifies that op has the same operand and result types after resolving
// reference types (i.e., after converting reference types to their
// corresponding TensorFlow or standard types).
template <typename ConcreteType>
class SameOperandsAndResultTypeResolveRef
    : public TraitBase<ConcreteType, SameOperandsAndResultTypeResolveRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_traitsDTh mht_4(mht_4_v, 306, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h", "verifyTrait");

    if (failed(impl::verifySameOperandsAndResultShape(op))) return failure();
    return detail::verifySameOperandsAndResultElementTypeResolveRef(op);
  }
};

// Layout agnostic operations do not depend on the operands data layout (data
// format), as and example all element wise operations are layout agnostic.
template <typename ConcreteType>
class LayoutAgnostic : public TraitBase<ConcreteType, LayoutAgnostic> {};

// Trait to indicate operations that cannot be duplicated as they might carry
// certain state around within their implementations.
template <typename ConcreteType>
class CannotDuplicate : public TraitBase<ConcreteType, CannotDuplicate> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_traitsDTh mht_5(mht_5_v, 325, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h", "verifyTrait");

    if (MemoryEffectOpInterface::hasNoEffect(op))
      return op->emitError(
          "operations with no side effects cannot have CannotDuplicate trait");
    return success();
  }
};

// Trait to indicate an operation cannot be constant folded.
template <typename ConcreteType>
class NoConstantFold : public TraitBase<ConcreteType, NoConstantFold> {};

// Coefficient-wise binary operation with implicit broadcasting support, for
// example tf.Sub operation.
template <typename ConcreteType>
class CwiseBinary : public TraitBase<ConcreteType, CwiseBinary> {};

// Coefficient-wise unary operation, for example tf.Sqrt operation.
template <typename ConcreteType>
class CwiseUnary : public TraitBase<ConcreteType, CwiseUnary> {};

// Indicates that any returned resource is unique.
template <typename ConcreteType>
class UniqueResourceAllocation
    : public TraitBase<ConcreteType, UniqueResourceAllocation> {
 public:
  // Implements method required for `ResourceHandleAllocatorInterface`.
  llvm::SmallVector<mlir::TF::ResourceHandleValueAndId>
  GetResourceHandleValueAndIdList(
      llvm::SmallDenseMap<mlir::TF::ResourceHandle, int64_t>&
          resource_handle_id_map,
      int64_t& next_id) {
    llvm::SmallVector<mlir::TF::ResourceHandleValueAndId> resource_vec;
    for (Value resource :
         mlir::tf_type::filter_resources(this->getOperation()->getResults())) {
      resource_vec.push_back({resource, next_id++});
    }
    return resource_vec;
  }
};

}  // namespace TF
}  // namespace OpTrait
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TRAITS_H_
