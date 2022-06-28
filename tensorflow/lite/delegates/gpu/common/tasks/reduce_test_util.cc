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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduce_test_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduce_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduce_test_utilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/reduce_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/reduce.h"

namespace tflite {
namespace gpu {
namespace {
template <DataType T>
absl::Status ReduceSumChannelsIntTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduce_test_utilDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce_test_util.cc", "ReduceSumChannelsIntTest");

  tflite::gpu::Tensor<BHWC, T> src;
  src.shape = BHWC(1, 2, 1, 5);
  src.data = {1, 2, -5, -2, 1, 3, 4, -2, 1, 4};

  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  tflite::gpu::Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 1);
  ref_tensor.data = {-3, 10};

  for (auto storage : env->GetSupportedStorages(T)) {
    OperationDef op_def;
    op_def.precision = CalculationsPrecision::F32;
    op_def.src_tensors.push_back({T, storage, Layout::HWC});
    op_def.dst_tensors.push_back({T, storage, Layout::HWC});
    TensorDescriptor src_0, dst;
    src_0 = op_def.src_tensors[0];
    src_0.UploadData(src);
    dst.SetBHWCShape(BHWC(1, 2, 1, 1));
    Reduce operation = CreateReduce(axis, src.shape, OperationType::REDUCE_SUM,
                                    op_def, env->GetGpuInfo());
    RETURN_IF_ERROR(env->ExecuteGPUOperation(
        {&src_0}, {&dst}, absl::make_unique<Reduce>(std::move(operation))));
    tflite::gpu::Tensor<BHWC, T> dst_tensor;
    dst.DownloadData(&dst_tensor);
    if (dst_tensor.data != ref_tensor.data) {
      return absl::InternalError("not equal");
    }
  }
  return absl::OkStatus();
}

template absl::Status ReduceSumChannelsIntTest<DataType::INT32>(
    TestExecutionEnvironment* env);
template absl::Status ReduceSumChannelsIntTest<DataType::INT16>(
    TestExecutionEnvironment* env);
template absl::Status ReduceSumChannelsIntTest<DataType::INT8>(
    TestExecutionEnvironment* env);

template <DataType T>
absl::Status ReduceProductChannelsUIntTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduce_test_utilDTcc mht_1(mht_1_v, 242, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce_test_util.cc", "ReduceProductChannelsUIntTest");

  tflite::gpu::Tensor<BHWC, T> src;
  src.shape = BHWC(1, 3, 1, 2);
  src.data = {1, 2, 3, 4, 0, 7};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  tflite::gpu::Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 3, 1, 1);
  ref_tensor.data = {2, 12, 0};

  for (auto storage : env->GetSupportedStorages(T)) {
    OperationDef op_def;
    op_def.precision = CalculationsPrecision::F32;
    op_def.src_tensors.push_back({T, storage, Layout::HWC});
    op_def.dst_tensors.push_back({T, storage, Layout::HWC});
    TensorDescriptor src_0, dst;
    src_0 = op_def.src_tensors[0];
    src_0.UploadData(src);
    dst.SetBHWCShape(BHWC(1, 3, 1, 1));
    Reduce operation =
        CreateReduce(axis, src.shape, OperationType::REDUCE_PRODUCT, op_def,
                     env->GetGpuInfo());
    RETURN_IF_ERROR(env->ExecuteGPUOperation(
        {&src_0}, {&dst}, absl::make_unique<Reduce>(std::move(operation))));
    tflite::gpu::Tensor<BHWC, T> dst_tensor;
    dst.DownloadData(&dst_tensor);
    if (dst_tensor.data != ref_tensor.data) {
      return absl::InternalError("not equal");
    }
  }
  return absl::OkStatus();
}

template absl::Status ReduceProductChannelsUIntTest<DataType::INT32>(
    TestExecutionEnvironment* env);
template absl::Status ReduceProductChannelsUIntTest<DataType::INT16>(
    TestExecutionEnvironment* env);
template absl::Status ReduceProductChannelsUIntTest<DataType::INT8>(
    TestExecutionEnvironment* env);
}  // namespace

absl::Status MeanHWTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduce_test_utilDTcc mht_2(mht_2_v, 286, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce_test_util.cc", "MeanHWTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::set<tflite::gpu::Axis> axis{Axis::HEIGHT, Axis::WIDTH};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::MEAN, op_def,
                       env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 1, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({2.5f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ReduceSumChannelsTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduce_test_utilDTcc mht_3(mht_3_v, 316, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce_test_util.cc", "ReduceSumChannelsTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 5);
  src_tensor.data = {1.1, 2.1, 0.7, 0.3, 1.2, 3.1, 4.1, 0.0, 1.0, 4.4};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_SUM,
                       op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 2, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({5.4f, 12.6f}, dst_tensor.data, eps));
    }
  }

  RETURN_IF_ERROR(ReduceSumChannelsIntTest<DataType::INT32>(env));
  RETURN_IF_ERROR(ReduceSumChannelsIntTest<DataType::INT16>(env));
  RETURN_IF_ERROR(ReduceSumChannelsIntTest<DataType::INT8>(env));
  return absl::OkStatus();
}

absl::Status ReduceProductChannelsTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduce_test_utilDTcc mht_4(mht_4_v, 350, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce_test_util.cc", "ReduceProductChannelsTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.1, 2.0, 3.1, 4.0};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_PRODUCT,
                       op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 2, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({2.2f, 12.4f}, dst_tensor.data, eps));
    }
  }

  RETURN_IF_ERROR(ReduceProductChannelsUIntTest<DataType::UINT32>(env));
  RETURN_IF_ERROR(ReduceProductChannelsUIntTest<DataType::UINT16>(env));
  RETURN_IF_ERROR(ReduceProductChannelsUIntTest<DataType::UINT8>(env));
  return absl::OkStatus();
}

absl::Status ReduceMaxChannelsTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduce_test_utilDTcc mht_5(mht_5_v, 384, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce_test_util.cc", "ReduceMaxChannelsTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 6);
  src_tensor.data = {1.1,  2.0,  -0.3, -100.0, 32.6, 1.1,
                     -3.1, -4.0, -5.0, -7.0,   -2.0, -100.0};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_MAXIMUM,
                       op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 2, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({32.6f, -2.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ReduceMinChannelsTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSreduce_test_utilDTcc mht_6(mht_6_v, 415, "", "./tensorflow/lite/delegates/gpu/common/tasks/reduce_test_util.cc", "ReduceMinChannelsTest");

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 6);
  src_tensor.data = {1.1,  2.0,  -0.3, -100.0, 32.6, 1.1,
                     -3.1, -4.0, -5.0, -7.0,   -2.0, 100.0};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_MINIMUM,
                       op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 2, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({-100.0f, -7.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
