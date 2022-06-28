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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_tensorDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_tensorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_tensorDTcc() {
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

// This file implements TF runtime fallback tensor.

#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/runtime_fallback/util/tensor_util.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

using tfrt::DenseHostTensor;
using tfrt::DType;
using tfrt::Expected;
using tfrt::HostBuffer;
using tfrt::HostContext;
using tfrt::RCReference;
using tfrt::StringHostTensor;
using tfrt::Tensor;
using tfrt::TensorMetadata;
using tfrt::TensorShape;

using OwnedTFStatus = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;

// If dtype is unsupported, only crash when converting this object to
// HostTensor.
RuntimeFallbackTensor::RuntimeFallbackTensor(const TensorShape& shape,
                                             DType dtype, OwnedTensorHandle th)
    : Tensor(TensorMetadata(dtype, shape)), tensor_handle_{std::move(th)} {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_tensorDTcc mht_0(mht_0_v, 231, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.cc", "RuntimeFallbackTensor::RuntimeFallbackTensor");

  assert(IsValid(dtype) && "Invalid dtype");
}

llvm::SmallVector<tfrt::Index, 4> GetShape(
    AbstractTensorInterface* tensor_interface) {
  llvm::SmallVector<tfrt::Index, 4> dims;
  int64_t num_dims = tensor_interface->NumDims();
  dims.reserve(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dims.push_back(tensor_interface->Dim(i));
  }
  return dims;
}

Expected<StringHostTensor> CopyTfStringTensorToStringHostTensor(
    AbstractTensorInterface* tensor_interface, HostContext* host) {
  auto sht = StringHostTensor::CreateUninitialized(
      TensorMetadata(DType(DType::String), GetShape(tensor_interface)), host);
  if (!sht)
    return tfrt::MakeStringError(
        "failed to create uninitialized string tensor");

  assert(tensor_interface->Type() == DT_STRING);
  const int64_t num_elems = tensor_interface->NumElements();
  const tensorflow::tstring* tstrings =
      reinterpret_cast<const tensorflow::tstring*>(tensor_interface->Data());

  auto strings = sht->strings();
  for (int i = 0; i < num_elems; ++i) {
    strings[i] = tstrings[i];
  }

  return std::move(*sht);
}

// TODO(jingdong): Format the tensor in more user-friendly format, especially
// for large tensors. See tensorflow::Tensor::DebugString().
void RuntimeFallbackTensor::Print(tfrt::raw_ostream& os) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_tensorDTcc mht_1(mht_1_v, 272, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.cc", "RuntimeFallbackTensor::Print");

  tensorflow::Status status;
  OwnedAbstractTensorInterface tensor_interface{
      tensor_handle_->Resolve(&status)};
  assert(status.ok());

  int rank = tensor_interface->NumDims();

  llvm::SmallVector<tfrt::Index, 4> dims;
  for (auto i = 0; i < rank; ++i) {
    dims.push_back(tensor_interface->Dim(i));
  }

  DataType dtype = tensor_interface->Type();
  os << "RuntimeFallbackTensor dtype = " << DataTypeString(dtype)
     << ", shape = [";
  llvm::interleaveComma(dims, os);
  os << "], values = [";

  int64_t num_elements = tensor_interface->NumElements();
  void* tensor_data = tensor_interface->Data();

  switch (dtype) {
    case TF_DataType::TF_FLOAT:
      PrintTensorValues<float>(tensor_data, num_elements, os);
      break;
    case TF_DataType::TF_DOUBLE:
      PrintTensorValues<double>(tensor_data, num_elements, os);
      break;
    case TF_DataType::TF_INT32:
      PrintTensorValues<int32_t>(tensor_data, num_elements, os);
      break;
    case TF_DataType::TF_INT64:
      PrintTensorValues<int64_t>(tensor_data, num_elements, os);
      break;
    case TF_DataType::TF_INT8:
      PrintTensorValues<int8_t>(tensor_data, num_elements, os);
      break;
    default:
      os << "Unsupported tensor dtype " << dtype;
      break;
  }

  os << "]\n";
}

tfrt::Expected<RuntimeFallbackTensor>
CreateRuntimeFallbackTensorFromTfTensorHandle(OwnedTensorHandle owned_th,
                                              HostContext* host) {
  int rank;
  tensorflow::Status status = owned_th->NumDims(&rank);
  if (!status.ok())
    return tfrt::MakeStringError(tfrt::StrCat(
        "error getting rank from TF tensor handle: ", status.error_message()));

  llvm::SmallVector<tfrt::Index, 4> dims;
  for (auto i = 0; i < rank; ++i) {
    int64_t dim;
    status = owned_th->Dim(i, &dim);
    if (!status.ok())
      return tfrt::MakeStringError(
          tfrt::StrCat("error getting dimension from TFE tensor handle: ",
                       status.error_message()));
    dims.push_back(dim);
  }

  TensorShape shape{dims};
  DataType dtype = owned_th->DataType();
  return RuntimeFallbackTensor(shape, GetTfrtDtype(dtype), std::move(owned_th));
}

RuntimeFallbackTensor MoveDHTToRuntimeFallbackTensor(DenseHostTensor&& dht,
                                                     HostContext* host) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_tensorDTcc mht_2(mht_2_v, 347, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.cc", "MoveDHTToRuntimeFallbackTensor");

  // TF_NewTensor takes the ownership of host_buffer.
  RCReference<HostBuffer> host_buffer = dht.ReleaseBuffer();
  tensorflow::Tensor tensor = MoveHostBufferToTfTensor(
      std::move(host_buffer), dht.dtype(), dht.shape());

  // TODO(zhangqiaorjc): Use CreateLocalHandle with device args.
  OwnedTensorHandle tensor_handle{
      tensorflow::TensorHandle::CreateLocalHandle(tensor)};

  return RuntimeFallbackTensor(dht.shape(), dht.dtype(),
                               std::move(tensor_handle));
}

RuntimeFallbackTensor CopyRefDHTToRuntimeFallbackTensor(
    const DenseHostTensor& dht, HostContext* host) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_tensorDTcc mht_3(mht_3_v, 365, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.cc", "CopyRefDHTToRuntimeFallbackTensor");

  // Do not copy the host buffer, TF_NewTensor simply CopyRef.
  RCReference<HostBuffer> host_buffer = dht.buffer();
  tensorflow::Tensor tensor = MoveHostBufferToTfTensor(
      std::move(host_buffer), dht.dtype(), dht.shape());

  OwnedTensorHandle tensor_handle{
      tensorflow::TensorHandle::CreateLocalHandle(tensor)};

  return RuntimeFallbackTensor(dht.shape(), dht.dtype(),
                               std::move(tensor_handle));
}

RuntimeFallbackTensor CopySHTToRuntimeFallbackTensor(
    const StringHostTensor& sht, HostContext* host) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_tensorDTcc mht_4(mht_4_v, 382, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.cc", "CopySHTToRuntimeFallbackTensor");

  tensorflow::Tensor tensor = CopyShtToTfTensor(sht);
  OwnedTensorHandle tensor_handle{
      tensorflow::TensorHandle::CreateLocalHandle(tensor)};

  return RuntimeFallbackTensor(sht.shape(), sht.dtype(),
                               std::move(tensor_handle));
}

}  // namespace tfd
}  // namespace tensorflow
