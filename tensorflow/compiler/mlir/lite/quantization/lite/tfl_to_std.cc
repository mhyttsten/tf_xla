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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePStfl_to_stdDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePStfl_to_stdDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePStfl_to_stdDTcc() {
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
#include "tensorflow/compiler/mlir/lite/quantization/lite/tfl_to_std.h"

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"

namespace mlir {
namespace TFL {

void ConvertTFLQuantOpsToMlirQuantOps(FuncOp func) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePStfl_to_stdDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/mlir/lite/quantization/lite/tfl_to_std.cc", "ConvertTFLQuantOpsToMlirQuantOps");

  OpBuilder b(func);
  func.walk([&](Operation* op) {
    b.setInsertionPoint(op);
    if (auto dq = llvm::dyn_cast<DequantizeOp>(op)) {
      auto dcast = b.create<quant::DequantizeCastOp>(
          dq.getLoc(), dq.output().getType(), dq.input());
      dq.output().replaceAllUsesWith(dcast);
      dq.erase();
    } else if (auto q = llvm::dyn_cast<QuantizeOp>(op)) {
      auto qcast = b.create<quant::QuantizeCastOp>(
          q.getLoc(), q.output().getType(), q.input());
      q.output().replaceAllUsesWith(qcast);
      q.erase();
    } else if (auto q = llvm::dyn_cast<ConstOp>(op)) {
      auto value = q.value();
      auto type = q.getResult().getType();
      if (arith::ConstantOp::isBuildableWith(value, type)) {
        auto c = b.create<arith::ConstantOp>(q.getLoc(), q.value());
        q.output().replaceAllUsesWith(c);
        q.erase();
      } else if (TFL::NoValueOp::isBuildableWith(value, type)) {
        auto c = b.create<TFL::NoValueOp>(q.getLoc(), type, mlir::UnitAttr());
        q.output().replaceAllUsesWith(c);
        q.erase();
      }
    }
  });
}

void ConvertMlirQuantOpsToTFLQuantOps(FuncOp func) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePStfl_to_stdDTcc mht_1(mht_1_v, 229, "", "./tensorflow/compiler/mlir/lite/quantization/lite/tfl_to_std.cc", "ConvertMlirQuantOpsToTFLQuantOps");

  OpBuilder b(func);
  func.walk([&](Operation* op) {
    b.setInsertionPoint(op);
    if (auto dq = llvm::dyn_cast<quant::DequantizeCastOp>(op)) {
      auto dcast = b.create<DequantizeOp>(dq.getLoc(), dq.getResult().getType(),
                                          dq.arg());
      dq.getResult().replaceAllUsesWith(dcast);
      if (auto extra_attr = op->getAttr(mlir::quant::kVolatileOpAttrName)) {
        dcast->setAttr(mlir::quant::kVolatileOpAttrName, extra_attr);
      }
      dq.erase();
    } else if (auto q = llvm::dyn_cast<quant::QuantizeCastOp>(op)) {
      auto out_type = q.getResult().getType();
      auto qcast = b.create<QuantizeOp>(q.getLoc(), out_type, q.arg(),
                                        TypeAttr::get(out_type));
      q.getResult().replaceAllUsesWith(qcast);
      if (auto extra_attr = op->getAttr(mlir::quant::kVolatileOpAttrName)) {
        qcast->setAttr(mlir::quant::kVolatileOpAttrName, extra_attr);
      }
      q.erase();
    }
  });
}

}  // namespace TFL
}  // namespace mlir
