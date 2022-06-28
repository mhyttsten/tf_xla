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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/elementwise.h"

namespace tflite {
namespace gpu {

absl::Status AbsTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "AbsTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {half(0.0f), half(-1.0f), half(-0.05f), half(0.045f)};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::ABS);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(0.0f), half(1.0f), half(0.05f), half(0.045f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status CosTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "CosTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, -1.0f, -0.05f, 0.045f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 5e-5f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COS);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {std::cos(0.0f), std::cos(-1.0f), std::cos(-0.05f), std::cos(0.045f)},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status CopyTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_2(mht_2_v, 256, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "CopyTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {half(0.0f), half(-1.0f), half(-0.05f), half(0.045f)};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::COPY);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(src_tensor.data, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status EluTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_3(mht_3_v, 283, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "EluTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {0.0f, 1.0f, -1.0f, 100.0f, -100.0f, 0.01f, -0.01f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::ELU);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 1, 1, 7), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {0.0f, 1.0f, std::exp(-1.0f) - 1.0f, 100.0f, std::exp(-100.0f) - 1.0f,
           0.01f, std::exp(-0.01f) - 1.0f},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ExpTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_4(mht_4_v, 314, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "ExpTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {0.0f, 1.0f, -1.0f, 2.5f, -1.7f, 0.01f, -0.01f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 2e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::EXP);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 1, 1, 7), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {std::exp(0.0f), std::exp(1.0f), std::exp(-1.0f), std::exp(2.5f),
           std::exp(-1.7f), std::exp(0.01f), std::exp(-0.01f)},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status FloorTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_5(mht_5_v, 345, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "FloorTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::FLOOR);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          src_tensor.shape, &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {-5.0, -3.0f, -2.0f, 0.0f, 1.0f, 3.0f, 4.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status FloorDivTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_6(mht_6_v, 374, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "FloorDivTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  float scalar = 2.7f;
  ElementwiseAttributes attr;
  attr.param = scalar;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(
          env->GetGpuInfo(), op_def, OperationType::FLOOR_DIV, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          src_tensor.shape, &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({std::floor(-4.5f / scalar), std::floor(-3.0f / scalar),
                         std::floor(-1.5f / scalar), std::floor(0.0f / scalar),
                         std::floor(1.5f / scalar), std::floor(3.0f / scalar),
                         std::floor(4.5f / scalar)},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status FloorModTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_7(mht_7_v, 411, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "FloorModTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  float scalar = 2.7f;
  ElementwiseAttributes attr;
  attr.param = scalar;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(
          env->GetGpuInfo(), op_def, OperationType::FLOOR_MOD, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          src_tensor.shape, &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({-4.5f - std::floor(-4.5f / scalar) * scalar,
                         -3.0f - std::floor(-3.0f / scalar) * scalar,
                         -1.5f - std::floor(-1.5f / scalar) * scalar,
                         0.0f - std::floor(0.0f / scalar) * scalar,
                         1.5f - std::floor(1.5f / scalar) * scalar,
                         3.0f - std::floor(3.0f / scalar) * scalar,
                         4.5f - std::floor(4.5f / scalar) * scalar},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status HardSwishTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_8(mht_8_v, 451, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "HardSwishTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::HARD_SWISH);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          src_tensor.shape, &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.0f, 0.0f, -0.375f, 0.0f, 1.125f, 3.f, 4.5f},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status LogTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_9(mht_9_v, 481, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "LogTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::LOG);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {std::log(1.0f), std::log(2.0f), std::log(3.0f), std::log(4.0f)},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status NegTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_10(mht_10_v, 511, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "NegTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, -2.0f, 0.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::NEG);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({-1.0f, 2.0f, 0.0f, -4.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status RsqrtTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_11(mht_11_v, 540, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "RsqrtTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::RSQRT);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f / std::sqrt(1.0f), 1.0f / std::sqrt(2.0f),
                         1.0f / std::sqrt(3.0f), 1.0f / std::sqrt(4.0f)},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SigmoidTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_12(mht_12_v, 571, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "SigmoidTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-std::log(1.0f), -std::log(2.0f), -std::log(3.0f),
                     -std::log(4.0f)};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SIGMOID);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({0.5f, 1.0f / 3.0f, 0.25f, 0.2f},
                                    dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SinTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_13(mht_13_v, 601, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "SinTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, -1.0f, -0.05f, 0.045f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 5e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SIN);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {std::sin(0.0f), std::sin(-1.0f), std::sin(-0.05f), std::sin(0.045f)},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SqrtTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_14(mht_14_v, 631, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "SqrtTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SQRT);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {std::sqrt(1.0f), std::sqrt(2.0f), std::sqrt(3.0f), std::sqrt(4.0f)},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SquareTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_15(mht_15_v, 661, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "SquareTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, -2.0f, 3.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::SQUARE);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 4.0f, 9.0f, 16.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status TanhTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_16(mht_16_v, 690, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "TanhTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-4.0f, -0.1f, 0.1f, 2.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseOneInput(
          env->GetGpuInfo(), op_def, OperationType::TANH);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({std::tanh(-4.0f), std::tanh(-0.1f),
                                     std::tanh(0.1f), std::tanh(2.0f)},
                                    dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SubTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_17(mht_17_v, 720, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "SubTest");

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.0f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 3.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::SUB, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.5f, 1.0f, 0.0f, 0.5f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SquaredDiffTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_18(mht_18_v, 753, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "SquaredDiffTest");

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.0f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 3.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::SQUARED_DIFF, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.25f, 1.0f, 0.0f, 0.25f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status DivTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_19(mht_19_v, 786, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "DivTest");

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 1.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::DIV, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({2.0f, 2.0f, 1.0f, 3.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status PowTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_20(mht_20_v, 819, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "PowTest");

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {6.0f, 7.0f, 4.0f, 2.0f};
  src_tensor_1.data = {0.0f, 1.0f, 2.0f, 3.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::POW, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 7.0f, 16.0f, 8.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status AddTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_21(mht_21_v, 852, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "AddTest");

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 1.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::ADD, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.5f, 3.0f, 6.0f, 6.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MaximumTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_22(mht_22_v, 885, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "MaximumTest");

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};
  src_tensor_1.data = {1.0f, 2.0f, 3.0f, -2.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::MAXIMUM, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 2.0f, 3.0f, -2.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MaximumWithScalarTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_23(mht_23_v, 918, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "MaximumWithScalarTest");

  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 4, 1, 1);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};

  ElementwiseAttributes attr;
  attr.param = -1.0f;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::MAXIMUM, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 4, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.0f, -1.0f, 2.0f, -1.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MaximumWithConstantLinearTensorTest(
    TestExecutionEnvironment* env) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_24(mht_24_v, 951, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "MaximumWithConstantLinearTensorTest");

  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, -6.2f, -2.0f, 3.0f};

  ::tflite::gpu::Tensor<Linear, DataType::FLOAT32> linear_tensor;
  linear_tensor.shape = Linear(2);
  linear_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes attr;
  attr.param = linear_tensor;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::MAXIMUM, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 2.0f, 0.5f, 3.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MaximumWithConstantHWCTensorTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_25(mht_25_v, 986, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "MaximumWithConstantHWCTensorTest");

  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, -6.2f, -2.0f, 3.0f};

  ::tflite::gpu::Tensor<HWC, DataType::FLOAT32> hwc_tensor;
  hwc_tensor.shape = HWC(2, 1, 2);
  hwc_tensor.data = {0.5f, 2.0f, 0.7f, 4.7f};
  ElementwiseAttributes attr;
  attr.param = hwc_tensor;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::MAXIMUM, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 2.0f, 0.7f, 4.7f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}
absl::Status MaximumWithConstantHWCTensorBroadcastChannelsTest(
    TestExecutionEnvironment* env) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_26(mht_26_v, 1021, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "MaximumWithConstantHWCTensorBroadcastChannelsTest");

  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, -6.2f, -2.0f, 3.0f};

  ::tflite::gpu::Tensor<HWC, DataType::FLOAT32> hwc_tensor;
  hwc_tensor.shape = HWC(2, 1, 1);
  hwc_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes attr;
  attr.param = hwc_tensor;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::MAXIMUM, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 0.5f, 2.0f, 3.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MinimumTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_27(mht_27_v, 1056, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "MinimumTest");

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};
  src_tensor_1.data = {1.0f, 2.0f, 3.0f, -2.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::MINIMUM, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.0f, -6.2f, 2.0f, -3.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MinimumWithScalarTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_28(mht_28_v, 1089, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "MinimumWithScalarTest");

  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 4, 1, 1);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};

  ElementwiseAttributes attr;
  attr.param = -1.0f;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::MINIMUM, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 4, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({-1.0f, -6.2f, -1.0f, -3.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MulTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_29(mht_29_v, 1121, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "MulTest");

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 1.5f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::MUL, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.5f, 2.0f, 9.0f, 6.75f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MulBroadcastHWTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_30(mht_30_v, 1154, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "MulBroadcastHWTest");

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 1, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 3.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::MUL, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.5f, 6.0f, 1.5f, 13.5f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status MulBroadcastChannelsTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_31(mht_31_v, 1187, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "MulBroadcastChannelsTest");

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 1);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 3.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::MUL, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.5f, 1.0f, 9.0f, 13.5f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status SubWithScalarAtFirstPositionTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_32(mht_32_v, 1220, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "SubWithScalarAtFirstPositionTest");

  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 4, 1, 1);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};

  ElementwiseAttributes attr;
  attr.param = 4.0f;
  attr.runtime_tensor_is_second = true;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::SUB, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 4, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({4.0f, 10.2f, 2.0f, 7.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status LessTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_33(mht_33_v, 1253, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "LessTest");

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};
  src_tensor_1.data = {1.0f, 0.0f, 2.0f, -4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwiseTwoInput(
          op_def, OperationType::LESS, src_tensor_1.shape);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor_0, src_tensor_1},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 0.0f, 0.0f, 0.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status LessEqualTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_34(mht_34_v, 1286, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "LessEqualTest");

  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(
          env->GetGpuInfo(), op_def, OperationType::LESS_EQUAL, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 1.0f, 1.0f, 0.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status GreaterTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_35(mht_35_v, 1318, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "GreaterTest");

  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::GREATER, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.0f, 0.0f, 0.0f, 1.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status GreaterEqualTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_36(mht_36_v, 1350, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "GreaterEqualTest");

  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(
          env->GetGpuInfo(), op_def, OperationType::GREATER_EQUAL, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.0f, 0.0f, 1.0f, 1.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status EqualTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_37(mht_37_v, 1382, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "EqualTest");

  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(env->GetGpuInfo(), op_def,
                                                 OperationType::EQUAL, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.0f, 0.0f, 1.0f, 0.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status NotEqualTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSelementwise_test_utilDTcc mht_38(mht_38_v, 1414, "", "./tensorflow/lite/delegates/gpu/common/tasks/elementwise_test_util.cc", "NotEqualTest");

  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateElementwise(
          env->GetGpuInfo(), op_def, OperationType::NOT_EQUAL, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor_0, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 1.0f, 0.0f, 1.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
