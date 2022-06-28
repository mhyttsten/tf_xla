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
class MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStest_utilsDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStest_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStest_utilsDTcc() {
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

#include "tensorflow/c/experimental/saved_model/core/test_utils.h"

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace testing {

std::unique_ptr<StaticDeviceMgr> CreateTestingDeviceMgr() {
  return std::make_unique<StaticDeviceMgr>(
      DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:0"));
}

EagerContextPtr CreateTestingEagerContext(DeviceMgr* device_mgr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStest_utilsDTcc mht_0(mht_0_v, 213, "", "./tensorflow/c/experimental/saved_model/core/test_utils.cc", "CreateTestingEagerContext");

  return EagerContextPtr(new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /* async= */ false, device_mgr,
      /* device_mgr_owned= */ false, /* rendezvous= */ nullptr,
      /* cluster_flr= */ nullptr));
}

std::vector<DataType> DataTypeSetToVector(DataTypeSet set) {
  std::vector<DataType> result;
  result.reserve(set.size());
  for (DataType dt : set) {
    result.push_back(dt);
  }
  return result;
}

std::vector<std::vector<int64_t>> InterestingShapes() {
  std::vector<std::vector<int64_t>> interesting_shapes;
  interesting_shapes.push_back({});             // Scalar
  interesting_shapes.push_back({10});           // 1D Vector
  interesting_shapes.push_back({3, 3});         // 2D Matrix
  interesting_shapes.push_back({1, 4, 6, 10});  // Higher Dimension Tensor
  return interesting_shapes;
}

ImmediateTensorHandlePtr CreateTensorHandle(ImmediateExecutionContext* ctx,
                                            DataType dtype,
                                            absl::Span<const int64_t> shape,
                                            int8_t value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStest_utilsDTcc mht_1(mht_1_v, 246, "", "./tensorflow/c/experimental/saved_model/core/test_utils.cc", "CreateTensorHandle");

  AbstractTensorPtr tensor(ctx->CreateTensor(dtype, shape));
  CHECK_NE(tensor.get(), nullptr)
      << "Tensor creation failed for tensor of dtype: "
      << DataTypeString(dtype);
  CHECK_EQ(tensor->Type(), dtype);
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_EQ(tensor->Dim(i), shape[i]);
  }
  FillNumericTensorBuffer(tensor->Type(), tensor->NumElements(), tensor->Data(),
                          value);
  ImmediateTensorHandlePtr handle(ctx->CreateLocalHandle(tensor.get()));
  CHECK_NE(handle.get(), nullptr);
  return handle;
}

void FillNumericTensorBuffer(DataType dtype, size_t num_elements, void* buffer,
                             int8_t value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStest_utilsDTcc mht_2(mht_2_v, 266, "", "./tensorflow/c/experimental/saved_model/core/test_utils.cc", "FillNumericTensorBuffer");

  switch (dtype) {
#define CASE(type)                                   \
  case DataTypeToEnum<type>::value: {                \
    type* typed_buffer = static_cast<type*>(buffer); \
    for (size_t i = 0; i < num_elements; ++i) {      \
      typed_buffer[i] = value;                       \
    }                                                \
    break;                                           \
  }
    TF_CALL_INTEGRAL_TYPES(CASE);
    TF_CALL_double(CASE);
    TF_CALL_float(CASE);
#undef CASE
    default:
      CHECK(false) << "Unsupported data type: " << DataTypeString(dtype);
      break;
  }
}

// Checks the underlying data is equal for the buffers for two numeric tensors.
// Note: The caller must ensure to check that the dtypes and sizes of the
// underlying buffers are the same before calling this.
void CheckBufferDataIsEqual(DataType dtype, int64_t num_elements, void* a,
                            void* b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStest_utilsDTcc mht_3(mht_3_v, 293, "", "./tensorflow/c/experimental/saved_model/core/test_utils.cc", "CheckBufferDataIsEqual");

  switch (dtype) {
#define CASE(type)                               \
  case DataTypeToEnum<type>::value: {            \
    type* typed_a = static_cast<type*>(a);       \
    type* typed_b = static_cast<type*>(b);       \
    for (int64_t i = 0; i < num_elements; ++i) { \
      if (DataTypeIsFloating(dtype)) {           \
        EXPECT_FLOAT_EQ(typed_a[i], typed_b[i]); \
      } else {                                   \
        EXPECT_EQ(typed_a[i], typed_b[i]);       \
      }                                          \
    }                                            \
    break;                                       \
  }
    TF_CALL_INTEGRAL_TYPES(CASE);
    TF_CALL_double(CASE);
    TF_CALL_float(CASE);
#undef CASE
    default:
      CHECK(false) << "Unsupported data type: " << DataTypeString(dtype);
  }
}

AbstractTensorPtr TensorHandleToTensor(ImmediateExecutionTensorHandle* handle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePStest_utilsDTcc mht_4(mht_4_v, 320, "", "./tensorflow/c/experimental/saved_model/core/test_utils.cc", "TensorHandleToTensor");

  Status status;
  AbstractTensorPtr tensor(handle->Resolve(&status));
  CHECK(status.ok()) << status.error_message();
  CHECK_NE(tensor.get(), nullptr);
  return tensor;
}

}  // namespace testing
}  // namespace tensorflow
