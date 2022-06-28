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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_COMMON_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_COMMON_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTh() {
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


#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tfrt {
namespace fallback_common {

template <typename OpTy>
mlir::LogicalResult VerifyExecuteOpCommon(OpTy op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTh mht_0(mht_0_v, 197, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.h", "VerifyExecuteOpCommon");

  auto op_attr_array = op.op_attrs().getValue();
  for (auto op_attr : op_attr_array) {
    auto key_value = op_attr.template dyn_cast<mlir::ArrayAttr>();
    if (!key_value || key_value.getValue().size() != 2 ||
        !key_value.getValue()[0].template isa<mlir::StringAttr>())
      return op.emitOpError() << "each op_attr should be a key-value pair, "
                                 "where the key is a string";
  }
  return mlir::success();
}

template <typename OpTy>
mlir::LogicalResult VerifyFallbackExecuteOp(OpTy op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTh mht_1(mht_1_v, 213, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.h", "VerifyFallbackExecuteOp");

  auto result = VerifyExecuteOpCommon(op);
  if (failed(result)) return result;

  // Verify function attributes.
  auto op_func_attr_array = op.op_func_attrs().getValue();
  for (auto op_attr : op_func_attr_array) {
    auto key_value = op_attr.template dyn_cast<mlir::ArrayAttr>();
    if (!key_value || key_value.getValue().size() != 2 ||
        !key_value.getValue()[0].template isa<mlir::StringAttr>() ||
        !key_value.getValue()[1].template isa<mlir::StringAttr>())
      return op.emitOpError() << "each op_func_attr should be a key-value "
                                 "pair, where both the key and the value are "
                                 "strings";
  }
  return mlir::success();
}

template <typename OpTy>
void PrintExecuteOpFuncAttribute(mlir::OpAsmPrinter &p, OpTy op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTh mht_2(mht_2_v, 235, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.h", "PrintExecuteOpFuncAttribute");

  auto op_func_attrs = op.op_func_attrs();
  if (!op_func_attrs.empty()) {
    auto print_key_value = [&](mlir::Attribute attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTh mht_3(mht_3_v, 241, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.h", "lambda");

      auto key_value = attr.cast<mlir::ArrayAttr>().getValue();
      auto key = key_value[0];
      auto value = key_value[1];

      p << key.cast<mlir::StringAttr>().getValue();
      p << " = ";
      p << value;
    };

    auto op_func_attr_array = op_func_attrs.getValue();
    p << " {";
    llvm::interleaveComma(op_func_attr_array, p, print_key_value);
    p << '}';
  }
}

template <typename OpTy>
void PrintExecuteOpCommon(mlir::OpAsmPrinter &p, OpTy op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTh mht_4(mht_4_v, 262, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.h", "PrintExecuteOpCommon");

  auto op_attrs = op.op_attrs();
  if (!op_attrs.empty()) {
    auto print_key_value = [&](mlir::Attribute attr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTh mht_5(mht_5_v, 268, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.h", "lambda");

      auto key_value = attr.cast<mlir::ArrayAttr>().getValue();
      auto key = key_value[0];
      auto value = key_value[1];

      p << key.cast<mlir::StringAttr>().getValue();
      p << " = ";
      p << value;
    };

    auto op_attr_array = op_attrs.getValue();
    p << " {";
    llvm::interleaveComma(op_attr_array, p, print_key_value);
    p << '}';
  }
}

void GetExecuteOpAttrsCommon(
    mlir::MLIRContext *context, llvm::ArrayRef<mlir::Attribute> op_attr_array,
    llvm::SmallVectorImpl<std::pair<llvm::StringRef, mlir::Attribute>>
        *op_attrs);

struct ParseExecuteOpOptions {
  bool has_chain = false;
  bool has_key = false;
  bool has_device = false;
  bool has_func_attr = false;
  bool has_cost = false;
};

mlir::ParseResult ParseExecuteOpCommon(mlir::OpAsmParser &parser,
                                       mlir::Builder &builder,
                                       mlir::OperationState &result,
                                       mlir::Type tensor_type,
                                       const ParseExecuteOpOptions &options);
}  // namespace fallback_common
}  // namespace tfrt

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_COMMON_H_
