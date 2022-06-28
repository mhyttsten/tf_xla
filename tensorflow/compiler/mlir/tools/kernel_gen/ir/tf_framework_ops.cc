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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPSirPStf_framework_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPSirPStf_framework_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPSirPStf_framework_opsDTcc() {
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

// This file defines the operations used in the tf_framework dialect.

#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_status.cc.inc"

// Generated dialect definitions.
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_dialect.cc.inc"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

void TFFrameworkDialect::initialize() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPSirPStf_framework_opsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.cc", "TFFrameworkDialect::initialize");

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.cc.inc"
      >();
  addTypes<JITCallableType, OpKernelContextType>();
}

/// Parse a type registered to this dialect.
Type TFFrameworkDialect::parseType(DialectAsmParser &parser) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPSirPStf_framework_opsDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.cc", "TFFrameworkDialect::parseType");

  StringRef keyword;
  if (parser.parseKeyword(&keyword)) return Type();

  if (keyword == "op_kernel_context") {
    return OpKernelContextType::get(getContext());
  }
  if (keyword == "jit_callable") {
    return JITCallableType::get(getContext());
  }

  parser.emitError(parser.getNameLoc(), "unknown TF Framework type: ")
      << keyword;
  return Type();
}

/// Print a type registered to this dialect.
void TFFrameworkDialect::printType(Type type, DialectAsmPrinter &os) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPSirPStf_framework_opsDTcc mht_2(mht_2_v, 235, "", "./tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.cc", "TFFrameworkDialect::printType");

  if (type.isa<OpKernelContextType>()) {
    os << "op_kernel_context";
    return;
  }
  if (type.isa<JITCallableType>()) {
    os << "jit_callable";
    return;
  }
  llvm_unreachable("unexpected TF Framework type kind");
}

//===----------------------------------------------------------------------===//
// TFAllocOp
//===----------------------------------------------------------------------===//
LogicalResult TFAllocOp::verify() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPSirPStf_framework_opsDTcc mht_3(mht_3_v, 253, "", "./tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.cc", "TFAllocOp::verify");

  TFAllocOp op = *this;
  // Check that the total number of operands matches the number of dynamic
  // dimensions specified in the memref type.
  unsigned result_dyn_dims = op.getType().getNumDynamicDims();
  unsigned dyn_sizes_count = op.dyn_sizes().size();
  if (dyn_sizes_count != result_dyn_dims)
    return op.emitOpError()
           << "`dyn_sizes` count " << dyn_sizes_count
           << " does not match dynamic dimensions count in the result type"
           << op.getType();
  return success();
}

Optional<Operation *> TFAllocOp::buildDealloc(OpBuilder &builder, Value alloc) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPSirPStf_framework_opsDTcc mht_4(mht_4_v, 270, "", "./tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.cc", "TFAllocOp::buildDealloc");

  auto funcop = alloc.getParentRegion()->getParentOfType<func::FuncOp>();
  return builder
      .create<TFDeallocOp>(alloc.getLoc(), funcop.getArgument(0), alloc)
      .getOperation();
}

Optional<Value> TFAllocOp::buildClone(OpBuilder &builder, Value alloc) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPSirPStf_framework_opsDTcc mht_5(mht_5_v, 280, "", "./tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.cc", "TFAllocOp::buildClone");

  // TODO(herhut): We should have our own clone op if one of these survives.
  return builder.create<mlir::bufferization::CloneOp>(alloc.getLoc(), alloc)
      .getResult();
}

::tensorflow::error::Code ConvertAttrToEnumValue(ErrorCode error_code) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPSirPStf_framework_opsDTcc mht_6(mht_6_v, 289, "", "./tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.cc", "ConvertAttrToEnumValue");

  using ::tensorflow::error::Code;
  switch (error_code) {
    case ErrorCode::OK:
      return Code::OK;
    case ErrorCode::CANCELLED:
      return Code::CANCELLED;
    case ErrorCode::UNKNOWN:
      return Code::UNKNOWN;
    case ErrorCode::INVALID_ARGUMENT:
      return Code::INVALID_ARGUMENT;
    case ErrorCode::DEADLINE_EXCEEDED:
      return Code::DEADLINE_EXCEEDED;
    case ErrorCode::NOT_FOUND:
      return Code::NOT_FOUND;
    case ErrorCode::ALREADY_EXISTS:
      return Code::ALREADY_EXISTS;
    case ErrorCode::PERMISSION_DENIED:
      return Code::PERMISSION_DENIED;
    case ErrorCode::UNAUTHENTICATED:
      return Code::UNAUTHENTICATED;
    case ErrorCode::RESOURCE_EXHAUSTED:
      return Code::RESOURCE_EXHAUSTED;
    case ErrorCode::FAILED_PRECONDITION:
      return Code::FAILED_PRECONDITION;
    case ErrorCode::ABORTED:
      return Code::ABORTED;
    case ErrorCode::OUT_OF_RANGE:
      return Code::OUT_OF_RANGE;
    case ErrorCode::UNIMPLEMENTED:
      return Code::UNIMPLEMENTED;
    case ErrorCode::INTERNAL:
      return Code::INTERNAL;
    case ErrorCode::UNAVAILABLE:
      return Code::UNAVAILABLE;
    case ErrorCode::DATA_LOSS:
      return Code::DATA_LOSS;
  }
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.cc.inc"
