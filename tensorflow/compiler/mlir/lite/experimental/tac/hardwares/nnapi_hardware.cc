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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSnnapi_hardwareDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSnnapi_hardwareDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSnnapi_hardwareDTcc() {
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

#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/nnapi_hardware.h"

#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform_patterns.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/arithmetic_count_util.h"

namespace mlir {
namespace TFL {
namespace tac {

// The copy can be non-consectutive copy. This is just fake data.
constexpr float kNNAPICopyUnitCost = 0.2;

// Default values.
constexpr float kNNAPIDefaultFixedValuedCost = 10000.0;

constexpr char NNAPIHardware::kId[];  // Define kId.

mlir::RewritePatternSet NNAPIHardware::GetTransformations(
    MLIRContext* context) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSnnapi_hardwareDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/nnapi_hardware.cc", "NNAPIHardware::GetTransformations");

  mlir::RewritePatternSet patterns(context);

  patterns.add<SquaredDifference, LowerPackIntoConcatReshape,
               ReduceMeanToAvgPool, InsertRequantForReduceMean>(context);
  return patterns;
}

std::unique_ptr<TargetHardware> CreateNNAPIHardware() {
  return std::make_unique<NNAPIHardware>();
}

TargetHardwareRegistration<NNAPIHardware> nnapi_hardware(
    "Target device for NNAPI", CreateNNAPIHardware);

// Currently used for these ops:
// tfl.squared_difference
class NNAPIBasicSupportedOpNoCost : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSnnapi_hardwareDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/nnapi_hardware.cc", "GetOpCost");
 return 0; }

  bool IsOpSupported(mlir::Operation* op) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSnnapi_hardwareDTcc mht_2(mht_2_v, 232, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/nnapi_hardware.cc", "IsOpSupported");

    return true;
  }
};

std::unique_ptr<TargetHardwareOperation> CreateBasicOpNoCost() {
  return std::make_unique<NNAPIBasicSupportedOpNoCost>();
}

// Currently used for these ops:
// tfl.concatenation / tfl.reshape / tfl.pack
class NNAPIConcatOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSnnapi_hardwareDTcc mht_3(mht_3_v, 247, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/nnapi_hardware.cc", "GetOpCost");

    int64_t count;
    if (ArithmeticCountUtilHelper::GetInputTensorTotalSize(op, &count))
      return kNNAPICopyUnitCost * count;
    return kNNAPIDefaultFixedValuedCost;
  }

  bool IsOpSupported(mlir::Operation* op) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPSnnapi_hardwareDTcc mht_4(mht_4_v, 257, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/nnapi_hardware.cc", "IsOpSupported");
 return true; }
};
std::unique_ptr<TargetHardwareOperation> CreateConcatOp() {
  return std::make_unique<NNAPIConcatOp>();
}

#define TAC_REGISTER_NNAPI_OP(Op, Create)                                      \
  TargetHardwareOpRegistration<NNAPIHardware, Op> Op##_NNAPIHardware_hardware( \
      Create);

// Op registeration
TAC_REGISTER_NNAPI_OP(SquaredDifferenceOp, CreateBasicOpNoCost);
TAC_REGISTER_NNAPI_OP(ConcatenationOp, CreateConcatOp);
TAC_REGISTER_NNAPI_OP(ReshapeOp, CreateConcatOp);
TAC_REGISTER_NNAPI_OP(PackOp, CreateConcatOp);

#undef TAC_REGISTER_NNAPI_OP
}  // namespace tac
}  // namespace TFL
}  // namespace mlir
