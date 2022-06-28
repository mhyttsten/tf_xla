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
class MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc {
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
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc() {
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

#include <sstream>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels/tensor_shape_utils.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

namespace {

// Struct that stores the status and TF_Tensor inputs to the opkernel.
// Used to delete tensor and status in its destructor upon kernel return.
struct Params {
  TF_Tensor* tags;
  TF_Tensor* values;
  TF_Status* status;
  explicit Params(TF_OpKernelContext* ctx)
      : tags(nullptr), values(nullptr), status(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc mht_0(mht_0_v, 214, "", "./tensorflow/c/kernels/summary_op.cc", "Params");

    status = TF_NewStatus();
    TF_GetInput(ctx, 0, &tags, status);
    if (TF_GetCode(status) == TF_OK) {
      TF_GetInput(ctx, 1, &values, status);
    }
  }
  ~Params() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc mht_1(mht_1_v, 224, "", "./tensorflow/c/kernels/summary_op.cc", "~Params");

    TF_DeleteStatus(status);
    TF_DeleteTensor(tags);
    TF_DeleteTensor(values);
  }
};

// dummy functions used for kernel registration
void* ScalarSummaryOp_Create(TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc mht_2(mht_2_v, 235, "", "./tensorflow/c/kernels/summary_op.cc", "ScalarSummaryOp_Create");
 return nullptr; }

void ScalarSummaryOp_Delete(void* kernel) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc mht_3(mht_3_v, 240, "", "./tensorflow/c/kernels/summary_op.cc", "ScalarSummaryOp_Delete");
}

// Helper functions for compute method
bool IsSameSize(TF_Tensor* tensor1, TF_Tensor* tensor2);
// Returns a string representation of a single tag or empty string if there
// are multiple tags
std::string SingleTag(TF_Tensor* tags);

template <typename T>
void ScalarSummaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc mht_4(mht_4_v, 252, "", "./tensorflow/c/kernels/summary_op.cc", "ScalarSummaryOp_Compute");

  Params params(ctx);
  if (TF_GetCode(params.status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, params.status);
    return;
  }
  if (!IsSameSize(params.tags, params.values)) {
    std::ostringstream err;
    err << "tags and values are not the same shape: "
        << tensorflow::ShapeDebugString(params.tags)
        << " != " << tensorflow::ShapeDebugString(params.values)
        << SingleTag(params.tags);
    TF_SetStatus(params.status, TF_INVALID_ARGUMENT, err.str().c_str());
    TF_OpKernelContext_Failure(ctx, params.status);
    return;
  }
  // Convert tags and values tensor to array to access elements by index
  tensorflow::Summary s;
  auto tags_array =
      static_cast<tensorflow::tstring*>(TF_TensorData(params.tags));
  auto values_array = static_cast<T*>(TF_TensorData(params.values));
  // Copy tags and values into summary protobuf
  for (int i = 0; i < TF_TensorElementCount(params.tags); ++i) {
    tensorflow::Summary::Value* v = s.add_value();
    const tensorflow::tstring& Ttags_i = tags_array[i];
    v->set_tag(Ttags_i.data(), Ttags_i.size());
    v->set_simple_value(static_cast<float>(values_array[i]));
  }
  TF_Tensor* summary_tensor =
      TF_AllocateOutput(ctx, 0, TF_ExpectedOutputDataType(ctx, 0), nullptr, 0,
                        sizeof(tensorflow::tstring), params.status);
  if (TF_GetCode(params.status) != TF_OK) {
    TF_DeleteTensor(summary_tensor);
    TF_OpKernelContext_Failure(ctx, params.status);
    return;
  }
  tensorflow::tstring* output_tstring =
      reinterpret_cast<tensorflow::tstring*>(TF_TensorData(summary_tensor));
  CHECK(SerializeToTString(s, output_tstring));
  TF_DeleteTensor(summary_tensor);
}

bool IsSameSize(TF_Tensor* tensor1, TF_Tensor* tensor2) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc mht_5(mht_5_v, 297, "", "./tensorflow/c/kernels/summary_op.cc", "IsSameSize");

  if (TF_NumDims(tensor1) != TF_NumDims(tensor2)) {
    return false;
  }
  for (int d = 0; d < TF_NumDims(tensor1); d++) {
    if (TF_Dim(tensor1, d) != TF_Dim(tensor2, d)) {
      return false;
    }
  }
  return true;
}

std::string SingleTag(TF_Tensor* tags) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc mht_6(mht_6_v, 312, "", "./tensorflow/c/kernels/summary_op.cc", "SingleTag");

  if (TF_TensorElementCount(tags) == 1) {
    const char* single_tag =
        static_cast<tensorflow::tstring*>(TF_TensorData(tags))->c_str();
    return tensorflow::strings::StrCat(" (tag '", single_tag, "')");
  } else {
    return "";
  }
}

template <typename T>
void RegisterScalarSummaryOpKernel() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc mht_7(mht_7_v, 326, "", "./tensorflow/c/kernels/summary_op.cc", "RegisterScalarSummaryOpKernel");

  TF_Status* status = TF_NewStatus();
  {
    auto* builder = TF_NewKernelBuilder(
        "ScalarSummary", tensorflow::DEVICE_CPU, &ScalarSummaryOp_Create,
        &ScalarSummaryOp_Compute<T>, &ScalarSummaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(tensorflow::DataTypeToEnum<T>::v()), status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << "Error while adding type constraint";
    TF_RegisterKernelBuilder("ScalarSummary", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering Scalar Summmary kernel";
  }
  TF_DeleteStatus(status);
}

// A dummy static variable initialized by a lambda whose side-effect is to
// register the ScalarSummary kernel.
TF_ATTRIBUTE_UNUSED bool IsScalarSummaryOpKernelRegistered = []() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSkernelsPSsummary_opDTcc mht_8(mht_8_v, 348, "", "./tensorflow/c/kernels/summary_op.cc", "lambda");

  if (SHOULD_REGISTER_OP_KERNEL("ScalarSummary")) {
    RegisterScalarSummaryOpKernel<int64_t>();
    RegisterScalarSummaryOpKernel<tensorflow::uint64>();
    RegisterScalarSummaryOpKernel<tensorflow::int32>();
    RegisterScalarSummaryOpKernel<tensorflow::uint32>();
    RegisterScalarSummaryOpKernel<tensorflow::uint16>();
    RegisterScalarSummaryOpKernel<tensorflow::int16>();
    RegisterScalarSummaryOpKernel<tensorflow::int8>();
    RegisterScalarSummaryOpKernel<tensorflow::uint8>();
    RegisterScalarSummaryOpKernel<Eigen::half>();
    RegisterScalarSummaryOpKernel<tensorflow::bfloat16>();
    RegisterScalarSummaryOpKernel<float>();
    RegisterScalarSummaryOpKernel<double>();
  }
  return true;
}();
}  // namespace
