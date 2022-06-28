/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_LAYOUT_HELPER_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_LAYOUT_HELPER_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_ops_layout_helperDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_ops_layout_helperDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_ops_layout_helperDTh() {
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


#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"

namespace mlir {

class MLIRContext;

namespace TF {

SmallVector<int64_t, 4> ReversePermutation(ArrayRef<int64_t> permutation);

SmallVector<int64_t, 4> GetDataFormatPermutation(StringRef from, StringRef to);

// Shuffle elements in the `attr` according to the permutation. Optional
// `inner_size` allows to shuffle array attributes created from rank 2 tensors
// on outer dimension only.
ArrayAttr ShuffleArrayAttr(ArrayAttr attr, ArrayRef<int64_t> permutation,
                           int inner_size = 1);

// Shuffle ranked tensor dimensions according to the permutation.
Type ShuffleRankedTensorType(Type type, ArrayRef<int64_t> permutation);

bool AreCancellablePermutations(DenseIntElementsAttr perm0,
                                DenseIntElementsAttr perm1);

// Default implementation of `LayoutSensitiveInterface::UpdateDataFormat` for
// layout sensitive operations that do not have any additional layout dependent
// attributes besides `data_format` string.
template <typename Op>
LogicalResult UpdateDataFormat(StringRef data_format, Op *op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_ops_layout_helperDTh mht_0(mht_0_v, 223, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.h", "UpdateDataFormat");

  auto perm = GetDataFormatPermutation(op->data_format(), data_format);
  if (perm.empty()) return failure();

  // Update data format attribute.
  (*op)->setAttr("data_format", StringAttr::get(op->getContext(), data_format));

  // Update types for all layout sensitive results.
  auto layout_sensitive = cast<LayoutSensitiveInterface>(op->getOperation());
  for (unsigned idx : layout_sensitive.GetLayoutDependentResults()) {
    OpResult result = op->getOperation()->getResult(idx);
    result.setType(ShuffleRankedTensorType(result.getType(), perm));
  }

  return success();
}

// Default implementation for folding operand transpose into the operation.
// See `FoldOperandsTransposeInterface::FoldOperandsPermutation`.
template <typename Op>
LogicalResult FoldOperandsPermutation(
    ArrayRef<int64_t> permutation, Op *op,
    ArrayRef<std::pair<StringRef, ArrayAttr>> shuffle_attrs = {}) {
  MLIRContext *context =
      (*op)->template getParentOfType<ModuleOp>().getContext();

  // We only support NHWC <-> NCHW permutations.
  static constexpr std::array<int64_t, 4> kNchwToNhwc = {0, 2, 3, 1};
  static constexpr std::array<int64_t, 4> kNhwcToNchw = {0, 3, 1, 2};

  // Operation data format after folding `permutation`.
  StringRef target_data_format = [&]() -> StringRef {
    if (op->data_format() == "NHWC" && permutation.equals(kNchwToNhwc)) {
      return "NCHW";  // cancel NCHW->NHWC operand permutation
    } else if (op->data_format() == "NCHW" && permutation.equals(kNhwcToNchw)) {
      return "NHWC";  // cancel NHWC->NCHW operand permutation
    } else {
      return "";
    }
  }();
  if (target_data_format.empty()) return failure();

  // To fold operand `permutation` into the `op` we need shuffle all layout
  // dependent attributes and types with a reverse permutation, and change
  // operation data format to `target_data_format`.
  //
  // Example:
  //   %1 = SomeOp(...)   {data_format = NHWC}
  //   %2 = Transpose(%1) {permutation = NHWC->NCHW}
  //   %3 = Op(%2)        {data_format = NCHW}
  //
  // To bypass %2 we have to change data format to shuffle data format from NCHW
  // to NHWC, which is the reverse of operand permutation (function argument).
  auto reverse_permutation =
      GetDataFormatPermutation(op->data_format(), target_data_format);
  if (reverse_permutation.empty()) return failure();

  (*op)->setAttr("data_format", StringAttr::get(context, target_data_format));

  for (auto pair : shuffle_attrs) {
    StringRef attr_name = pair.first;
    ArrayAttr attr_value = pair.second;
    (*op)->setAttr(attr_name,
                   ShuffleArrayAttr(attr_value, reverse_permutation));
  }

  auto fold = cast<FoldOperandsTransposeInterface>(op->getOperation());
  for (unsigned idx : fold.GetLayoutDependentResults()) {
    OpResult result = op->getOperation()->getResult(idx);
    result.setType(
        ShuffleRankedTensorType(result.getType(), reverse_permutation));
  }

  return success();
}

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_LAYOUT_HELPER_H_
