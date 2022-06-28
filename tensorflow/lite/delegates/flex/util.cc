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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutilDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/flex/util.h"

#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace flex {

static constexpr char kResourceVariablePrefix[] = "tflite_resource_variable";

TfLiteStatus ConvertStatus(TfLiteContext* context,
                           const tensorflow::Status& status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutilDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/flex/util.cc", "ConvertStatus");

  if (!status.ok()) {
    context->ReportError(context, "%s", status.error_message().c_str());
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus CopyShapeAndType(TfLiteContext* context,
                              const tensorflow::Tensor& src,
                              TfLiteTensor* tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutilDTcc mht_1(mht_1_v, 213, "", "./tensorflow/lite/delegates/flex/util.cc", "CopyShapeAndType");

  tensor->type = GetTensorFlowLiteType(static_cast<TF_DataType>(src.dtype()));
  if (tensor->type == kTfLiteNoType) {
    context->ReportError(context,
                         "TF Lite does not support TensorFlow data type: %s",
                         DataTypeString(src.dtype()).c_str());
    return kTfLiteError;
  }

  int num_dims = src.dims();
  TfLiteIntArray* shape = TfLiteIntArrayCreate(num_dims);
  for (int j = 0; j < num_dims; ++j) {
    // We need to cast from TensorFlow's int64 to TF Lite's int32. Let's
    // make sure there's no overflow.
    if (src.dim_size(j) >= std::numeric_limits<int>::max()) {
      context->ReportError(context,
                           "Dimension value in TensorFlow shape is larger than "
                           "supported by TF Lite");
      TfLiteIntArrayFree(shape);
      return kTfLiteError;
    }
    shape->data[j] = static_cast<int>(src.dim_size(j));
  }
  return context->ResizeTensor(context, tensor, shape);
}

TF_DataType GetTensorFlowDataType(TfLiteType type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutilDTcc mht_2(mht_2_v, 242, "", "./tensorflow/lite/delegates/flex/util.cc", "GetTensorFlowDataType");

  switch (type) {
    case kTfLiteNoType:
      return TF_FLOAT;
    case kTfLiteFloat32:
      return TF_FLOAT;
    case kTfLiteFloat16:
      return TF_HALF;
    case kTfLiteFloat64:
      return TF_DOUBLE;
    case kTfLiteInt16:
      return TF_INT16;
    case kTfLiteUInt16:
      return TF_UINT16;
    case kTfLiteInt32:
      return TF_INT32;
    case kTfLiteUInt32:
      return TF_UINT32;
    case kTfLiteUInt8:
      return TF_UINT8;
    case kTfLiteInt8:
      return TF_INT8;
    case kTfLiteInt64:
      return TF_INT64;
    case kTfLiteUInt64:
      return TF_UINT64;
    case kTfLiteComplex64:
      return TF_COMPLEX64;
    case kTfLiteComplex128:
      return TF_COMPLEX128;
    case kTfLiteString:
      return TF_STRING;
    case kTfLiteBool:
      return TF_BOOL;
    case kTfLiteResource:
      return TF_RESOURCE;
    case kTfLiteVariant:
      return TF_VARIANT;
  }
}

TfLiteType GetTensorFlowLiteType(TF_DataType type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutilDTcc mht_3(mht_3_v, 286, "", "./tensorflow/lite/delegates/flex/util.cc", "GetTensorFlowLiteType");

  switch (type) {
    case TF_FLOAT:
      return kTfLiteFloat32;
    case TF_HALF:
      return kTfLiteFloat16;
    case TF_DOUBLE:
      return kTfLiteFloat64;
    case TF_INT16:
      return kTfLiteInt16;
    case TF_INT32:
      return kTfLiteInt32;
    case TF_UINT8:
      return kTfLiteUInt8;
    case TF_INT8:
      return kTfLiteInt8;
    case TF_INT64:
      return kTfLiteInt64;
    case TF_UINT64:
      return kTfLiteUInt64;
    case TF_COMPLEX64:
      return kTfLiteComplex64;
    case TF_COMPLEX128:
      return kTfLiteComplex128;
    case TF_STRING:
      return kTfLiteString;
    case TF_BOOL:
      return kTfLiteBool;
    case TF_RESOURCE:
      return kTfLiteResource;
    case TF_VARIANT:
      return kTfLiteVariant;
    default:
      return kTfLiteNoType;
  }
}

// Returns the TF data type name to be stored in the FunctionDef.
const char* TfLiteTypeToTfTypeName(TfLiteType type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutilDTcc mht_4(mht_4_v, 327, "", "./tensorflow/lite/delegates/flex/util.cc", "TfLiteTypeToTfTypeName");

  switch (type) {
    case kTfLiteNoType:
      return "invalid";
    case kTfLiteFloat32:
      return "float";
    case kTfLiteInt16:
      return "int16";
    case kTfLiteUInt16:
      return "uint16";
    case kTfLiteInt32:
      return "int32";
    case kTfLiteUInt32:
      return "uint32";
    case kTfLiteUInt8:
      return "uint8";
    case kTfLiteInt8:
      return "int8";
    case kTfLiteInt64:
      return "int64";
    case kTfLiteUInt64:
      return "uint64";
    case kTfLiteBool:
      return "bool";
    case kTfLiteComplex64:
      return "complex64";
    case kTfLiteComplex128:
      return "complex128";
    case kTfLiteString:
      return "string";
    case kTfLiteFloat16:
      return "float16";
    case kTfLiteFloat64:
      return "float64";
    case kTfLiteResource:
      return "resource";
    case kTfLiteVariant:
      return "variant";
  }
  return "invalid";
}

std::string TfLiteResourceIdentifier(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutilDTcc mht_5(mht_5_v, 372, "", "./tensorflow/lite/delegates/flex/util.cc", "TfLiteResourceIdentifier");

  // TODO(b/199782192): Create a util function to get Resource ID from a TF Lite
  // resource tensor.
  const int resource_id = tensor->data.i32[0];
  return absl::StrFormat("%s:%d", kResourceVariablePrefix, resource_id);
}

bool GetTfLiteResourceTensorFromResourceHandle(
    const tensorflow::ResourceHandle& resource_handle, TfLiteTensor* tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSutilDTcc mht_6(mht_6_v, 383, "", "./tensorflow/lite/delegates/flex/util.cc", "GetTfLiteResourceTensorFromResourceHandle");

  std::vector<std::string> parts = absl::StrSplit(resource_handle.name(), ':');
  if (parts.size() != 2) {
    return false;
  }
  const int kBytesRequired = sizeof(int32_t);
  TfLiteTensorRealloc(kBytesRequired, tensor);
  int resource_id;
  if (parts[0] == kResourceVariablePrefix &&
      absl::SimpleAtoi<int32_t>(parts[1], &resource_id)) {
    // TODO(b/199782192): Create a util function to set the Resource ID of
    // a TF Lite resource tensor.
    GetTensorData<int32_t>(tensor)[0] = resource_id;
    return true;
  }
  return false;
}

tensorflow::StatusOr<tensorflow::Tensor> CreateTfTensorFromTfLiteTensor(
    const TfLiteTensor* tflite_tensor) {
  if (IsResourceOrVariant(tflite_tensor)) {
    // Returns error if the input tflite tensor has variant or resource type.
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Input tensor has resource or variant type.");
  }

  tensorflow::TensorShape shape;
  int num_dims = tflite_tensor->dims->size;
  for (int i = 0; i < num_dims; ++i) {
    shape.AddDim(tflite_tensor->dims->data[i]);
  }

  tensorflow::Tensor tf_tensor(
      tensorflow::DataType(GetTensorFlowDataType(tflite_tensor->type)), shape);
  if (tf_tensor.dtype() == tensorflow::DataType::DT_STRING &&
      tf_tensor.data()) {
    tensorflow::tstring* buf =
        static_cast<tensorflow::tstring*>(tf_tensor.data());
    for (int i = 0; i < tflite::GetStringCount(tflite_tensor); ++buf, ++i) {
      auto ref = GetString(tflite_tensor, i);
      buf->assign(ref.str, ref.len);
    }
  } else {
    if (tf_tensor.tensor_data().size() != tflite_tensor->bytes) {
      return tensorflow::Status(
          tensorflow::error::INTERNAL,
          "TfLiteTensor's size doesn't match the TF tensor's size.");
    }
    if (!tflite_tensor->data.raw) {
      return tensorflow::Status(tensorflow::error::INTERNAL,
                                "TfLiteTensor's data field is null.");
    }
    std::memcpy(tf_tensor.data(), tflite_tensor->data.raw,
                tflite_tensor->bytes);
  }

  return tf_tensor;
}

}  // namespace flex
}  // namespace tflite
