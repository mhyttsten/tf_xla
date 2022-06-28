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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSfully_connected_test_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSfully_connected_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSfully_connected_test_utilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/fully_connected_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/fully_connected.h"

namespace tflite {
namespace gpu {

absl::Status FullyConnectedTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSfully_connected_test_utilDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/gpu/common/tasks/fully_connected_test_util.cc", "FullyConnectedTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 4);
  attr.weights.data = {0.0f, 1.0f, 2.0f, 3.0f,  //
                       4.0f, 5.0f, 6.0f, 7.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      FullyConnected operation =
          CreateFullyConnected(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<FullyConnected>(std::move(operation)),
          BHWC(1, 1, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({14.5f, 37.5f}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedLargeTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSfully_connected_test_utilDTcc mht_1(mht_1_v, 233, "", "./tensorflow/lite/delegates/gpu/common/tasks/fully_connected_test_util.cc", "FullyConnectedLargeTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 8);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(12, 1, 1, 8);
  attr.weights.data = {
      0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,   //
      8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,  //
      16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,  //
      24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,  //
      32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,  //
      40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,  //
      48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f,  //
      56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f,  //
      64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f,  //
      72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,  //
      80.0f, 81.0f, 82.0f, 83.0f, 84.0f, 85.0f, 86.0f, 87.0f,  //
      88.0f, 89.0f, 90.0f, 91.0f, 92.0f, 93.0f, 94.0f, 95.0f,  //
  };
  attr.bias.shape = Linear(12);
  attr.bias.data = {-0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f,
                    0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  0.6f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 0.0f : 1.0f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      FullyConnected operation =
          CreateFullyConnected(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<FullyConnected>(std::move(operation)),
          BHWC(1, 1, 1, 12), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({139.4f, 363.5f, 587.6f, 811.7f, 1035.8f, 1259.9f,
                         1484.1f, 1708.2f, 1932.3f, 2156.4f, 2380.5f, 2604.6f},
                        dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedExtraLargeTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSfully_connected_test_utilDTcc mht_2(mht_2_v, 285, "", "./tensorflow/lite/delegates/gpu/common/tasks/fully_connected_test_util.cc", "FullyConnectedExtraLargeTest");

  static const int kInputSize = 1024;
  static const int kOutputSize = 1024;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, kInputSize);
  src_tensor.data.assign(kInputSize, 1.1f);

  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(1024, 1, 1, kInputSize);
  attr.weights.data.assign(kOutputSize * kInputSize, 2.2f);
  attr.bias.shape = Linear(kOutputSize);
  attr.bias.data.assign(kOutputSize, 3.3f);

  std::vector<float> expected(kOutputSize, 2481.38f);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      float eps;
      switch (precision) {
        case CalculationsPrecision::F32:
          eps = 2.45e-3f;
          break;
        case CalculationsPrecision::F32_F16:
          eps = 1.38f;
          break;
        case CalculationsPrecision::F16:
          eps = 39.0f;
          break;
      }
      if (precision == CalculationsPrecision::F32_F16 &&
          env->GetGpuInfo().IsApiMetal() && env->GetGpuInfo().IsIntel()) {
        eps = 3.5f;
      }
      if (precision == CalculationsPrecision::F32_F16 &&
          env->GetGpuInfo().IsGlsl()) {
        eps = 3.5f;
      }
      if (!env->GetGpuInfo().IsRoundToNearestSupported()) {
        eps *= 4.0f;
      }
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      FullyConnected operation =
          CreateFullyConnected(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<FullyConnected>(std::move(operation)),
          BHWC(1, 1, 1, kOutputSize), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(expected, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
