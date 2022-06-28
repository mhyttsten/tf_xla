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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinograd_test_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinograd_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinograd_test_utilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/winograd_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/winograd.h"
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

namespace tflite {
namespace gpu {

absl::Status Winograd4x4To36TileX6Test(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinograd_test_utilDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd_test_util.cc", "Winograd4x4To36TileX6Test");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 4, 4, 1);
  src_tensor.data.resize(16);
  for (int i = 0; i < 16; ++i) {
    src_tensor.data[i] = sin(i);
  }

  TensorFloat32 dst_ref;
  dst_ref.shape = BHWC(1, 36, 1, 1);
  dst_ref.data.resize(36, 0.0f);
  auto b_t = BtMatrixForWinograd4x4To6x6();

  // Bt * Src * B
  // 1: temp = Src * B
  std::vector<float> temp(36, 0.0f);
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        if (y < 1 || y > 4 || i < 1 || i > 4) continue;
        const int index = src_tensor.shape.LinearIndex({0, y - 1, i - 1, 0});
        sum += src_tensor.data[index] * b_t[x * 6 + i];
      }
      temp[y * 6 + x] = sum;
    }
  }
  // 2: ref = Bt * temp
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        sum += b_t[y * 6 + i] * temp[i * 6 + x];
      }
      const int index = dst_ref.shape.LinearIndex({0, y * 6 + x, 0, 0});
      dst_ref.data[index] = sum;
    }
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      if (!env->GetGpuInfo().IsRoundToNearestSupported()) {
        eps *= 4.0f;
      }
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Padding2D padding;
      padding.prepended = HW(1, 1);
      padding.appended = HW(1, 1);
      Winograd4x4To36TileX6 operation =
          CreateWinograd4x4To36TileX6(env->GetGpuInfo(), op_def, padding);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor,
          absl::make_unique<Winograd4x4To36TileX6>(std::move(operation)),
          BHWC(1, 36, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(dst_ref.data, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status Winograd36To4x4Tile4x1Test(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinograd_test_utilDTcc mht_1(mht_1_v, 267, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd_test_util.cc", "Winograd36To4x4Tile4x1Test");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 36, 1, 1);
  src_tensor.data.resize(36);
  for (int i = 0; i < 36; ++i) {
    src_tensor.data[i] = sin(i);
  }

  ::tflite::gpu::Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(1);
  biases.data.resize(biases.shape.DimensionsProduct());
  for (int i = 0; i < biases.data.size(); ++i) {
    biases.data[i] = 0.0f;
  }

  TensorFloat32 dst_ref;
  dst_ref.shape = BHWC(1, 4, 4, 1);
  dst_ref.data.resize(16, 0.0f);
  auto a_t = AtMatrixForWinograd4x4To6x6();

  // At * Src * A
  // 1: temp = Src * A
  std::vector<float> temp(24, 0.0f);
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 4; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        const int index = src_tensor.shape.LinearIndex({0, y * 6 + i, 0, 0});
        sum += src_tensor.data[index] * a_t[x * 6 + i];
      }
      temp[y * 4 + x] = sum;
    }
  }
  // 2: ref = At * temp
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        sum += a_t[y * 6 + i] * temp[i * 4 + x];
      }
      const int index = dst_ref.shape.LinearIndex({0, y, x, 0});
      dst_ref.data[index] = sum;
    }
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      if (!env->GetGpuInfo().IsRoundToNearestSupported()) {
        eps *= 4.0f;
      }
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Winograd36To4x4Tile4x1 operation =
          CreateWinograd36To4x4Tile4x1(env->GetGpuInfo(), op_def, biases);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor,
          absl::make_unique<Winograd36To4x4Tile4x1>(std::move(operation)),
          BHWC(1, 4, 4, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(dst_ref.data, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status Winograd4x4To36Test(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinograd_test_utilDTcc mht_2(mht_2_v, 339, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd_test_util.cc", "Winograd4x4To36Test");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 4, 4, 1);
  src_tensor.data.resize(16);
  for (int i = 0; i < 16; ++i) {
    src_tensor.data[i] = sin(i);
  }

  TensorFloat32 dst_ref;
  dst_ref.shape = BHWC(1, 36, 1, 1);
  dst_ref.data.resize(36, 0.0f);
  auto b_t = BtMatrixForWinograd4x4To6x6();

  // Bt * Src * B
  // 1: temp = Src * B
  std::vector<float> temp(36, 0.0f);
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        if (y < 1 || y > 4 || i < 1 || i > 4) continue;
        const int index = src_tensor.shape.LinearIndex({0, y - 1, i - 1, 0});
        sum += src_tensor.data[index] * b_t[x * 6 + i];
      }
      temp[y * 6 + x] = sum;
    }
  }
  // 2: ref = Bt * temp
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        sum += b_t[y * 6 + i] * temp[i * 6 + x];
      }
      const int index = dst_ref.shape.LinearIndex({0, y * 6 + x, 0, 0});
      dst_ref.data[index] = sum;
    }
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      if (!env->GetGpuInfo().IsRoundToNearestSupported()) {
        eps *= 4.0f;
      }
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Padding2D padding;
      padding.prepended = HW(1, 1);
      padding.appended = HW(1, 1);
      Winograd4x4To36 operation = CreateWinograd4x4To36(op_def, padding);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Winograd4x4To36>(std::move(operation)),
          BHWC(1, 36, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(dst_ref.data, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status Winograd36To4x4Test(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinograd_test_utilDTcc mht_3(mht_3_v, 406, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd_test_util.cc", "Winograd36To4x4Test");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 36, 1, 1);
  src_tensor.data.resize(36);
  for (int i = 0; i < 36; ++i) {
    src_tensor.data[i] = sin(i);
  }

  ::tflite::gpu::Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(1);
  biases.data.resize(biases.shape.DimensionsProduct());
  for (int i = 0; i < biases.data.size(); ++i) {
    biases.data[i] = 0.0f;
  }

  TensorFloat32 dst_ref;
  dst_ref.shape = BHWC(1, 4, 4, 1);
  dst_ref.data.resize(16, 0.0f);
  auto a_t = AtMatrixForWinograd4x4To6x6();

  // At * Src * A
  // 1: temp = Src * A
  std::vector<float> temp(24, 0.0f);
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 4; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        const int index = src_tensor.shape.LinearIndex({0, y * 6 + i, 0, 0});
        sum += src_tensor.data[index] * a_t[x * 6 + i];
      }
      temp[y * 4 + x] = sum;
    }
  }
  // 2: ref = At * temp
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        sum += a_t[y * 6 + i] * temp[i * 4 + x];
      }
      const int index = dst_ref.shape.LinearIndex({0, y, x, 0});
      dst_ref.data[index] = sum;
    }
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      if (!env->GetGpuInfo().IsRoundToNearestSupported()) {
        eps *= 4.0f;
      }
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Winograd36To4x4 operation = CreateWinograd36To4x4(op_def, biases);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Winograd36To4x4>(std::move(operation)),
          BHWC(1, 4, 4, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(dst_ref.data, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
