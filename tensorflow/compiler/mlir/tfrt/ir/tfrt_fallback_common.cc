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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTcc() {
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
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.h"

#include "mlir/IR/Builders.h"  // from @llvm-project

namespace tfrt {
namespace fallback_common {

void GetExecuteOpAttrsCommon(
    mlir::MLIRContext *context, llvm::ArrayRef<mlir::Attribute> op_attr_array,
    llvm::SmallVectorImpl<std::pair<llvm::StringRef, mlir::Attribute>>
        *op_attrs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTcc mht_0(mht_0_v, 194, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.cc", "GetExecuteOpAttrsCommon");

  assert(op_attrs);
  op_attrs->clear();

  mlir::Builder builder(context);
  for (auto iter : op_attr_array) {
    auto key_value = iter.cast<mlir::ArrayAttr>().getValue();
    llvm::StringRef key = key_value[0].cast<mlir::StringAttr>().getValue();
    mlir::Attribute value = key_value[1];
    op_attrs->push_back({key, value});
  }
}

mlir::ParseResult ParseExecuteOpCommon(mlir::OpAsmParser &parser,
                                       mlir::Builder &builder,
                                       mlir::OperationState &result,
                                       mlir::Type tensor_type,
                                       const ParseExecuteOpOptions &options) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_commonDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.cc", "ParseExecuteOpCommon");

  auto chain_type = builder.getType<compiler::ChainType>();

  mlir::IntegerAttr op_key;
  mlir::IntegerAttr cost;
  mlir::StringAttr device;
  mlir::StringAttr op_name;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> in_chains;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
  mlir::NamedAttrList op_attrs;
  mlir::NamedAttrList op_func_attrs;
  auto loc = parser.getNameLoc();

  if (options.has_chain &&
      parser.parseOperandList(in_chains,
                              /*requiredOperandCount=*/1,
                              mlir::OpAsmParser::Delimiter::Paren))
    return mlir::failure();

  if (options.has_key &&
      (parser.parseKeyword("key") || parser.parseLParen() ||
       parser.parseAttribute(op_key, "op_key", result.attributes) ||
       parser.parseRParen()))
    return mlir::failure();

  if (options.has_cost &&
      (parser.parseKeyword("cost") || parser.parseLParen() ||
       parser.parseAttribute(cost, "_tfrt_cost", result.attributes) ||
       parser.parseRParen()))
    return mlir::failure();

  if (options.has_device &&
      (parser.parseKeyword("device") || parser.parseLParen() ||
       parser.parseAttribute(device, "device", result.attributes) ||
       parser.parseRParen()))
    return mlir::failure();

  if (parser.parseAttribute(op_name, "op_name", result.attributes) ||
      parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(op_attrs) ||
      parser.parseOptionalAttrDict(op_func_attrs))
    return mlir::failure();

  int64_t num_results = 0;
  if (succeeded(parser.parseOptionalColon())) {
    mlir::IntegerAttr attr;
    mlir::NamedAttrList attrs;
    if (failed(parser.parseAttribute(attr, "num_results", attrs)))
      return mlir::failure();
    num_results = attr.getValue().getSExtValue();
  }

  llvm::SmallVector<mlir::Type, 4> operand_types;
  if (options.has_chain) operand_types.push_back(chain_type);
  if (parser.resolveOperands(in_chains, operand_types, loc, result.operands) ||
      parser.resolveOperands(operands, tensor_type, result.operands))
    return mlir::failure();

  if (options.has_chain) result.types.push_back(chain_type);
  result.types.append(num_results, tensor_type);

  llvm::SmallVector<mlir::Attribute, 4> op_attr_array;
  for (const auto &key_value : op_attrs) {
    auto key = key_value.getName();
    auto value = key_value.getValue();
    op_attr_array.push_back(builder.getArrayAttr({key, value}));
  }

  result.attributes.push_back(
      builder.getNamedAttr("op_attrs", builder.getArrayAttr(op_attr_array)));

  // TODO(tfrt-devs): support func attributes in tfrt_fallback_sync.
  if (options.has_func_attr) {
    llvm::SmallVector<mlir::Attribute, 4> op_func_attr_array;
    for (const auto &key_value : op_func_attrs) {
      auto key = key_value.getName();
      auto value = key_value.getValue();
      op_func_attr_array.push_back(builder.getArrayAttr({key, value}));
    }

    result.attributes.push_back(builder.getNamedAttr(
        "op_func_attrs", builder.getArrayAttr(op_func_attr_array)));
  }

  return mlir::success();
}

}  // namespace fallback_common
}  // namespace tfrt
