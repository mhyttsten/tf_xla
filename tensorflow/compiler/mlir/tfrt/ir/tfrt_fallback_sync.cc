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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_syncDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_syncDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_syncDTcc() {
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
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/attributes.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime

namespace tfrt {
namespace fallback_sync {

FallbackSyncDialect::FallbackSyncDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfrt_fallback_sync", context,
              TypeID::get<FallbackSyncDialect>()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_syncDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cc", "FallbackSyncDialect::FallbackSyncDialect");

  context->getOrLoadDialect<tfrt::fallback::FallbackDialect>();
  context->getOrLoadDialect<compiler::TFRTDialect>();
  context->getOrLoadDialect<corert::CoreRTDialect>();

  allowUnknownTypes();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cpp.inc"
      >();
}

static Type GetTensorType(Builder *builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_syncDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cc", "GetTensorType");

  return tfrt::t::TensorType::get(builder->getContext());
}

LogicalResult ExecuteOp::verify() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_syncDTcc mht_2(mht_2_v, 231, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cc", "ExecuteOp::verify");

  return fallback_common::VerifyExecuteOpCommon(*this);
}

ParseResult ExecuteOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_syncDTcc mht_3(mht_3_v, 238, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cc", "ExecuteOp::parse");

  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = false;
  parse_options.has_key = false;
  parse_options.has_device = false;
  parse_options.has_func_attr = false;
  parse_options.has_cost = false;

  auto &builder = parser.getBuilder();
  return fallback_common::ParseExecuteOpCommon(
      parser, builder, result, GetTensorType(&builder), parse_options);
}

void ExecuteOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_syncDTcc mht_4(mht_4_v, 254, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cc", "ExecuteOp::print");

  p << " " << (*this)->getAttr("op_name") << '(' << operands() << ')';

  fallback_common::PrintExecuteOpCommon(p, *this);
  if (!results().empty()) p << " : " << results().size();
}

void ExecuteOp::getOpAttrs(
    SmallVectorImpl<std::pair<StringRef, Attribute>> *op_attrs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_syncDTcc mht_5(mht_5_v, 265, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cc", "ExecuteOp::getOpAttrs");

  fallback_common::GetExecuteOpAttrsCommon(
      this->getContext(), this->op_attrs().getValue(), op_attrs);
}

}  // namespace fallback_sync
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cpp.inc"
