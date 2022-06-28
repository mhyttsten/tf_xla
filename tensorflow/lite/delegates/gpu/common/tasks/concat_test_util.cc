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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_test_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_test_utilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/concat_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/concat_xy.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/concat_z.h"

namespace tflite {
namespace gpu {

absl::Status ConcatWidthTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_test_utilDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/gpu/common/tasks/concat_test_util.cc", "ConcatWidthTest");

  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {half(0.0f), half(-1.0f), half(-0.05f), half(0.045f)};
  src1.shape = BHWC(1, 2, 2, 2);
  src1.data = {half(1.0f), half(-1.2f), half(-0.45f), half(1.045f),
               half(1.1f), half(-1.3f), half(-0.55f), half(2.045f)};

  ConcatAttributes attr;
  attr.axis = Axis::WIDTH;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateConcatXY(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src0, src1}, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 3, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(0.0f), half(-1.0f), half(1.0f), half(-1.2f),
                         half(-0.45f), half(1.045f), half(-0.05f), half(0.045f),
                         half(1.1f), half(-1.3f), half(-0.55f), half(2.045f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status ConcatHeightTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_test_utilDTcc mht_1(mht_1_v, 235, "", "./tensorflow/lite/delegates/gpu/common/tasks/concat_test_util.cc", "ConcatHeightTest");

  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {half(0.0f), half(-1.0f), half(-0.05f), half(0.045f)};
  src1.shape = BHWC(1, 1, 1, 2);
  src1.data = {half(1.0f), half(-1.2f)};

  ConcatAttributes attr;
  attr.axis = Axis::HEIGHT;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateConcatXY(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src0, src1}, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 3, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({half(0.0f), half(-1.0f), half(-0.05f),
                                     half(0.045f), half(1.0f), half(-1.2f)},
                                    dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status ConcatChannelsTest(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_test_utilDTcc mht_2(mht_2_v, 269, "", "./tensorflow/lite/delegates/gpu/common/tasks/concat_test_util.cc", "ConcatChannelsTest");

  TensorFloat32 src0, src1, src2;
  src0.shape = BHWC(1, 2, 1, 1);
  src0.data = {half(0.0f), half(-1.0f)};
  src1.shape = BHWC(1, 2, 1, 2);
  src1.data = {half(1.0f), half(2.0f), half(3.0f), half(4.0f)};
  src2.shape = BHWC(1, 2, 1, 3);
  src2.data = {half(5.0f), half(6.0f), half(7.0f),
               half(8.0f), half(9.0),  half(10.0f)};

  ConcatAttributes attr;
  attr.axis = Axis::CHANNELS;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation =
          CreateConcatZ(op_def, {1, 2, 3}, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src0, src1, src2},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 6), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(0.0f), half(1.0f), half(2.0f), half(5.0f),
                         half(6.0f), half(7.0f), half(-1.0f), half(3.0f),
                         half(4.0f), half(8.0f), half(9.0), half(10.0f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status ConcatChannelsAlignedx4Test(TestExecutionEnvironment* env) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconcat_test_utilDTcc mht_3(mht_3_v, 311, "", "./tensorflow/lite/delegates/gpu/common/tasks/concat_test_util.cc", "ConcatChannelsAlignedx4Test");

  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 4);
  src0.data = {half(-1.0f), half(-2.0f), half(-3.0f), half(-4.0f),
               half(1.0f),  half(2.0f),  half(3.0f),  half(4.0f)};
  src1.shape = BHWC(1, 2, 1, 4);
  src1.data = {half(5.0f),  half(6.0f),  half(7.0f),  half(8.0f),
               half(-5.0f), half(-6.0f), half(-7.0f), half(-8.0f)};

  ConcatAttributes attr;
  attr.axis = Axis::CHANNELS;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateConcatZ(op_def, {4, 4}, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src0, src1}, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 8), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(-1.0f), half(-2.0f), half(-3.0f), half(-4.0f),
                         half(5.0f), half(6.0f), half(7.0f), half(8.0f),
                         half(1.0f), half(2.0f), half(3.0f), half(4.0f),
                         half(-5.0f), half(-6.0f), half(-7.0f), half(-8.0f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
