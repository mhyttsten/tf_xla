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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmean_stddev_normalization_test_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmean_stddev_normalization_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmean_stddev_normalization_test_utilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization.h"

namespace tflite {
namespace gpu {

// Parameterized test: mean, difference, tolerance.
// Input is constructed as [mean-2*diff, mean-diff, mean+diff, mean+2*diff]
absl::Status MeanStddevNormSeparateBatchesTest(float mean, float diff,
                                               float tolerance,
                                               TestExecutionEnvironment* env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmean_stddev_normalization_test_utilDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization_test_util.cc", "MeanStddevNormSeparateBatchesTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {mean - 2 * diff, mean - diff, mean + diff,
                     mean + 2 * diff};
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorFloat32 dst_tensor;
      auto operation =
          CreateMeanStdDevNormalization(op_def, env->GetGpuInfo(), 1);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor},
          absl::make_unique<MeanStdDevNormalization>(std::move(operation)),
          BHWC(1, 1, 1, 4), &dst_tensor));

      std::vector<float> expected_output;
      if (diff == 0.0f) {
        expected_output.assign({0.0f, 0.0f, 0.0f, 0.0f});
      } else {
        const float ksqrt16 = std::sqrt(1.6f);
        const float ksqrt04 = std::sqrt(0.4f);
        expected_output.assign({-ksqrt16, -ksqrt04, ksqrt04, ksqrt16});
      }
      RETURN_IF_ERROR(
          PointWiseNear(expected_output, dst_tensor.data, tolerance));
    }
  }
  return absl::OkStatus();
}

absl::Status MeanStddevNormalizationAllBatchesTest(
    TestExecutionEnvironment* env) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmean_stddev_normalization_test_utilDTcc mht_1(mht_1_v, 240, "", "./tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization_test_util.cc", "MeanStddevNormalizationAllBatchesTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(9, 1, 1, 4);
  src_tensor.data = {
      0.0f,    0.0f,    0.0f,   0.0f,    // zero mean, zero variance
      -0.02f,  -0.01f,  0.01f,  0.02f,   // zero mean, small variance
      -200.0f, -100.0f, 100.0f, 200.0f,  // zero mean, large variance
      0.01f,   0.01f,   0.01f,  0.01f,   // small mean, zero variance
      -0.01f,  0.0f,    0.02f,  0.03f,   // small mean, small variance
      -199.0f, -99.0f,  101.0f, 201.0f,  // small mean, large variance
      100.0f,  100.0f,  100.0f, 100.0f,  // large mean, zero variance
      98.0f,   99.0f,   101.0f, 102.0f,  // large mean, small variance
      -100.0f, 0.0f,    200.0f, 300.0f,  // large mean, large variance
  };
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps =
          precision == CalculationsPrecision::F32 ? 2.53e-05f : 3.57e-4f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorFloat32 dst_tensor;
      auto operation =
          CreateMeanStdDevNormalization(op_def, env->GetGpuInfo(), 1);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor},
          absl::make_unique<MeanStdDevNormalization>(std::move(operation)),
          BHWC(9, 1, 1, 4), &dst_tensor));

      const float ksqrt16 = std::sqrt(1.6f);
      const float ksqrt04 = std::sqrt(0.4f);
      const std::vector<float> expected_output = {
          0.0f,     0.0f,     0.0f,    0.0f,     // zero mean, zero variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // zero mean, small variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // zero mean, large variance
          0.0f,     0.0f,     0.0f,    0.0f,     // small mean, zero variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // small mean, small variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // small mean, large variance
          0.0f,     0.0f,     0.0f,    0.0f,     // large mean, zero variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // large mean, small variance
          -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // large mean, large variance
      };
      RETURN_IF_ERROR(PointWiseNear(expected_output, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status MeanStddevNormalizationLargeVectorTest(
    TestExecutionEnvironment* env) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSmean_stddev_normalization_test_utilDTcc mht_2(mht_2_v, 295, "", "./tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization_test_util.cc", "MeanStddevNormalizationLargeVectorTest");

  const float mean = 100.0f;
  const float diff = 1.0f;
  // Some large vector that is not a round multiple of any SIMD vector sizes.
  constexpr int kVectorSize = 16 * 16 + 16 + 1;

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, kVectorSize);
  src_tensor.data.resize(kVectorSize);
  // First input is mean.
  src_tensor.data[0] = mean;
  // Rest is alternating between mean + diff and mean - diff.
  for (int i = 1; i < kVectorSize - 1; i += 2) {
    src_tensor.data[i + 0] = mean + diff;
    src_tensor.data[i + 1] = mean - diff;
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps =
          precision == CalculationsPrecision::F32 ? 0.0f : 8.60e-4f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorFloat32 dst_tensor;
      auto operation = CreateMeanStdDevNormalization(op_def, env->GetGpuInfo(),
                                                     (kVectorSize + 3) / 4);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor},
          absl::make_unique<MeanStdDevNormalization>(std::move(operation)),
          BHWC(1, 1, 1, kVectorSize), &dst_tensor));

      std::vector<float> expected_output(kVectorSize);
      // First output should be 0.
      expected_output[0] = 0.0;
      // Rest should be alternating between ±√(N/(N-1)).
      const float expected_elem =
          std::sqrt(static_cast<double>(kVectorSize) /
                    static_cast<double>(kVectorSize - 1));
      for (int i = 1; i < kVectorSize - 1; i += 2) {
        expected_output[i + 0] = +expected_elem;
        expected_output[i + 1] = -expected_elem;
      }
      RETURN_IF_ERROR(PointWiseNear(expected_output, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
