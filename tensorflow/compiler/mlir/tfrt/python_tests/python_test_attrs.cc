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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSpython_testsPSpython_test_attrsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSpython_testsPSpython_test_attrsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSpython_testsPSpython_test_attrsDTcc() {
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

#include "tensorflow/compiler/mlir/tfrt/python_tests/python_test_attrs.h"

#include <algorithm>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
// Include the auto-generated dialect defs.
#include "tensorflow/compiler/mlir/tfrt/python_tests/python_test_attrs.cc.inc"

namespace mlir {
namespace tfrt {

void PythonTestAttrsDialect::initialize() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSpython_testsPSpython_test_attrsDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/mlir/tfrt/python_tests/python_test_attrs.cc", "PythonTestAttrsDialect::initialize");
}

::mlir::LogicalResult PythonTestAttrsDialect::verifyRegionArgAttribute(
    ::mlir::Operation* op, unsigned regionIndex, unsigned argIndex,
    ::mlir::NamedAttribute attribute) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSpython_testsPSpython_test_attrsDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/mlir/tfrt/python_tests/python_test_attrs.cc", "PythonTestAttrsDialect::verifyRegionArgAttribute");

  const auto& arg = op->getRegion(regionIndex).getArguments()[argIndex];

  // Only verify at the tensor level. We are interested in the correct attribute
  // values when processing the Tensorflow dialect IR.
  auto arg_type = arg.getType().dyn_cast<RankedTensorType>();
  if (!arg_type) return success();

  if (attribute.getName() == GetStaticTypeAttrName()) {
    auto type_attr = attribute.getValue().dyn_cast<TypeAttr>();
    if (!type_attr) {
      return op->emitError()
             << GetStaticTypeAttrName()
             << " argument attribute of other type than TypeAttr";
    }

    auto attr_type = type_attr.getValue().dyn_cast<RankedTensorType>();
    if (!attr_type) {
      return op->emitError()
             << GetStaticTypeAttrName()
             << " argument type attribute is not a ranked tensor type";
    }
    if (attr_type.getNumDynamicDims() > 0) {
      return op->emitError() << GetStaticTypeAttrName()
                             << " argument type attribute is a ranked tensor "
                                "type with dynamic dimensions";
    }
    if (attr_type.getRank() != arg_type.getRank()) {
      return op->emitError()
             << GetStaticTypeAttrName()
             << " argument type attribute is a ranked tensor type with a "
                "different rank than the rank of the argument tensor";
    }
    auto compatible = [&](Type a, Type b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSpython_testsPSpython_test_attrsDTcc mht_2(mht_2_v, 244, "", "./tensorflow/compiler/mlir/tfrt/python_tests/python_test_attrs.cc", "lambda");

      if (a == b) {
        return true;
      }
      if (!a.isa<IntegerType>() || !b.isa<IntegerType>()) {
        return false;
      }
      auto width_a = a.dyn_cast<IntegerType>().getWidth();
      auto width_b = b.dyn_cast<IntegerType>().getWidth();
      return width_a == width_b || std::max(width_a, width_b) == 8;
    };
    if (!compatible(attr_type.getElementType(), arg_type.getElementType())) {
      return op->emitError()
             << GetStaticTypeAttrName()
             << " argument type attribute is a ranked tensor type with a "
                "different element type than the element type of the argument "
                "tensor";
    }
    const auto& attr_shape = attr_type.getShape();
    const auto& arg_shape = arg_type.getShape();
    for (int64_t i = 0; i < attr_shape.size(); ++i) {
      if (!arg_type.isDynamicDim(i) && arg_shape[i] != attr_shape[i]) {
        return op->emitError()
               << GetStaticTypeAttrName()
               << " argument type attribute is a ranked tensor type with a "
                  "shape that doesn't match the static dimensions of the "
                  "argument tensor";
      }
    }
  } else if (attribute.getName() == GetShapeValueAttrName()) {
    auto dense_attr = attribute.getValue().dyn_cast<DenseIntElementsAttr>();
    if (!dense_attr) {
      return op->emitError()
             << GetShapeValueAttrName()
             << " argument attribute is not a dense int elements attribute";
    }

    if (dense_attr.getType() != arg_type) {
      return op->emitError() << GetShapeValueAttrName()
                             << " argument elements attribute has a different "
                                "type than the argument type";
    }

    // We expect a valid shape value, therefore check that the dimension values
    // are not negative.
    for (auto&& dim : dense_attr) {
      if (dim.isNegative()) {
        return op->emitError()
               << GetShapeValueAttrName()
               << " argument elements attribute has a negative dimension value";
      }
    }
  }
  return success();
}

}  // namespace tfrt
}  // namespace mlir
