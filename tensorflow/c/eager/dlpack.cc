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
class MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc() {
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

#include "tensorflow/c/eager/dlpack.h"

#include <string>

#include "include/dlpack/dlpack.h"  // from @dlpack
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

// Managing context for the DLManagedTensor, will manage the lifetime of
// DLManagedTensor. When calling DLManagedTensor::deleter, it will notify the
// original framework of destruction, and this context will be deleted also.
struct TfDlManagedTensorCtx {
  TensorReference reference;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor tensor;

  explicit TfDlManagedTensorCtx(const TensorReference& ref) : reference(ref) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc mht_0(mht_0_v, 212, "", "./tensorflow/c/eager/dlpack.cc", "TfDlManagedTensorCtx");
}
};

// Gets tensor from eager tensor handle.
const Tensor* GetTensorFromHandle(TFE_TensorHandle* h, TF_Status* status) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc mht_1(mht_1_v, 219, "", "./tensorflow/c/eager/dlpack.cc", "GetTensorFromHandle");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  tensorflow::TensorHandle* handle =
      tensorflow::TensorHandleFromInterface(tensorflow::unwrap(h));
  if (handle->Type() != TensorHandle::LOCAL) {
    status->status = tensorflow::errors::InvalidArgument(
        "DLPack doesn't support ", handle->TypeString(), " tensor");
    return nullptr;
  }
  const tensorflow::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return nullptr;
  }
  return tensor;
}

// Deleter for DLManagedTensor
void DLManagedTensorDeleter(DLManagedTensor* arg) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc mht_2(mht_2_v, 243, "", "./tensorflow/c/eager/dlpack.cc", "DLManagedTensorDeleter");

  TfDlManagedTensorCtx* owner =
      static_cast<TfDlManagedTensorCtx*>(arg->manager_ctx);
  owner->reference.Unref();
  delete owner;
}

// Converts TF_DATAType to DLPack data type.
DLDataType GetDlDataType(TF_DataType data_type, TF_Status* status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc mht_3(mht_3_v, 254, "", "./tensorflow/c/eager/dlpack.cc", "GetDlDataType");

  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = TF_DataTypeSize(data_type) * 8;
  switch (data_type) {
    case TF_DataType::TF_HALF:
    case TF_DataType::TF_FLOAT:
    case TF_DataType::TF_DOUBLE:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case TF_DataType::TF_INT8:
    case TF_DataType::TF_INT16:
    case TF_DataType::TF_INT32:
    case TF_DataType::TF_INT64:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case TF_DataType::TF_BOOL:
    case TF_DataType::TF_UINT8:
    case TF_DataType::TF_UINT16:
    case TF_DataType::TF_UINT32:
    case TF_DataType::TF_UINT64:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    case TF_DataType::TF_BFLOAT16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      break;
    case TF_DataType::TF_COMPLEX64:
    case TF_DataType::TF_COMPLEX128:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    default:
      status->status = tensorflow::errors::InvalidArgument(
          DataType_Name(static_cast<DataType>(data_type)),
          " is not supported by dlpack");
      break;
  }
  return dtype;
}

// Gets DLPack's DLDevice from eager tensor handle.
DLDevice GetDlContext(TFE_TensorHandle* h, TF_Status* status) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc mht_4(mht_4_v, 297, "", "./tensorflow/c/eager/dlpack.cc", "GetDlContext");

  DLDevice ctx;
  const char* device_name =
      tensorflow::unwrap(h)->BackingDeviceName(&status->status);
  DeviceNameUtils::ParsedName parsed_name;
  tensorflow::DeviceNameUtils::ParseFullName(device_name, &parsed_name);
  std::string device_type = parsed_name.type;
  int device_id = 0;
  if (parsed_name.has_id) {
    device_id = parsed_name.id;
  }

  ctx.device_id = device_id;
  if (device_type == "CPU") {
    ctx.device_type = DLDeviceType::kDLCPU;
  } else if (device_type == "GPU") {
    ctx.device_type = DLDeviceType::kDLCUDA;
  } else {
    status->status = tensorflow::errors::InvalidArgument(
        "Unsupported Device Type for dlpack");
  }

  return ctx;
}

// Converts DLDevice to TF device name.
absl::optional<std::string> DeviceNameFromDlContext(const DLDevice& ctx,
                                                    TF_Status* status) {
  switch (ctx.device_type) {
    case DLDeviceType::kDLCPU:
      return "CPU:0";
    case DLDeviceType::kDLCUDA:
      return absl::StrCat("GPU:", ctx.device_id);
    default:
      return absl::nullopt;
  }
}

// Converts DLPack data type to TF_DATATYPE.
Status TfDataTypeFormDlDataType(const DLDataType& dtype,
                                TF_DataType* tf_dtype) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc mht_5(mht_5_v, 340, "", "./tensorflow/c/eager/dlpack.cc", "TfDataTypeFormDlDataType");

  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          *tf_dtype = TF_DataType::TF_UINT8;
          return Status::OK();
        case 16:
          *tf_dtype = TF_DataType::TF_UINT16;
          return Status::OK();
        case 32:
          *tf_dtype = TF_DataType::TF_UINT32;
          return Status::OK();
        case 64:
          *tf_dtype = TF_DataType::TF_UINT64;
          return Status::OK();
        default:
          return tensorflow::errors::InvalidArgument("Unsupported UInt bits: ",
                                                     dtype.bits);
      }
      return Status::OK();
    case DLDataTypeCode::kDLInt:
      switch (dtype.bits) {
        case 8:
          *tf_dtype = TF_DataType::TF_INT8;
          return Status::OK();
        case 16:
          *tf_dtype = TF_DataType::TF_INT16;
          return Status::OK();
        case 32:
          *tf_dtype = TF_DataType::TF_INT32;
          return Status::OK();
        case 64:
          *tf_dtype = TF_DataType::TF_INT64;
          return Status::OK();
        default:
          return tensorflow::errors::InvalidArgument("Unsupported Int bits: ",
                                                     dtype.bits);
      }
      return Status::OK();
    case DLDataTypeCode::kDLFloat:
      switch (dtype.bits) {
        case 16:
          *tf_dtype = TF_DataType::TF_HALF;
          return Status::OK();
        case 32:
          *tf_dtype = TF_DataType::TF_FLOAT;
          return Status::OK();
        case 64:
          *tf_dtype = TF_DataType::TF_DOUBLE;
          return Status::OK();
        default:
          return tensorflow::errors::InvalidArgument("Unsupported Float bits: ",
                                                     dtype.bits);
      }
      break;
    case DLDataTypeCode::kDLBfloat:
      switch (dtype.bits) {
        case 16:
          *tf_dtype = TF_DataType::TF_BFLOAT16;
          return Status::OK();
        default:
          return tensorflow::errors::InvalidArgument(
              "Unsupported BFloat bits: ", dtype.bits);
      }
      break;
    case DLDataTypeCode::kDLComplex:
      switch (dtype.bits) {
        case 64:
          *tf_dtype = TF_DataType::TF_COMPLEX64;
          return Status::OK();
        case 128:
          *tf_dtype = TF_DataType::TF_COMPLEX128;
          return Status::OK();
        default:
          return tensorflow::errors::InvalidArgument(
              "Unsupported Complex bits: ", dtype.bits);
      }
      break;
    default:
      return tensorflow::errors::InvalidArgument("Unsupported Type Codes: ",
                                                 dtype.code);
  }
}

// Wraps the deleter function of DLManagedTensor to match the function signature
// TFE_NewTensorHandleFromDeviceMemory.
void DeallocatorWrapperFunc(void* data, size_t len, void* dlmt_vptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc mht_6(mht_6_v, 430, "", "./tensorflow/c/eager/dlpack.cc", "DeallocatorWrapperFunc");

  TFE_CallDLManagedTensorDeleter(dlmt_vptr);
}

// Checks whether the stride array matches the layout of compact, row-majored
// data.
bool IsValidStrideCompactRowMajorData(int64_t* shape_arr, int64_t* stride_arr,
                                      int ndim) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc mht_7(mht_7_v, 440, "", "./tensorflow/c/eager/dlpack.cc", "IsValidStrideCompactRowMajorData");

  if (ndim >= 1 && stride_arr[ndim - 1] != 1) {
    return false;
  }
  for (int i = ndim - 2; i >= 0; --i) {
    if (stride_arr[i] != shape_arr[i + 1] * stride_arr[i + 1]) {
      return false;
    }
  }
  return true;
}
}  // namespace

void TFE_CallDLManagedTensorDeleter(void* dlm_ptr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc mht_8(mht_8_v, 456, "", "./tensorflow/c/eager/dlpack.cc", "TFE_CallDLManagedTensorDeleter");

  DLManagedTensor* dlMTensor = static_cast<DLManagedTensor*>(dlm_ptr);
  if (dlMTensor->deleter != nullptr) {
    dlMTensor->deleter(dlMTensor);
  }
}

void* TFE_HandleToDLPack(TFE_TensorHandle* h, TF_Status* status) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc mht_9(mht_9_v, 466, "", "./tensorflow/c/eager/dlpack.cc", "TFE_HandleToDLPack");

  auto tf_dlm_context = GetDlContext(h, status);
  if (!status->status.ok()) {
    return nullptr;
  }

  auto* tf_dlm_data = TFE_TensorHandleDevicePointer(h, status);
  if (!status->status.ok()) {
    return nullptr;
  }

  const Tensor* tensor = GetTensorFromHandle(h, status);
  TF_DataType data_type = static_cast<TF_DataType>(tensor->dtype());

  auto tf_dlm_type = GetDlDataType(data_type, status);
  if (!status->status.ok()) {
    return nullptr;
  }

  TensorReference tensor_ref(*tensor);  // This will call buf_->Ref()
  auto* tf_dlm_tensor_ctx = new TfDlManagedTensorCtx(tensor_ref);
  tf_dlm_tensor_ctx->reference = tensor_ref;

  DLManagedTensor* dlm_tensor = &tf_dlm_tensor_ctx->tensor;
  dlm_tensor->manager_ctx = tf_dlm_tensor_ctx;
  dlm_tensor->deleter = &DLManagedTensorDeleter;
  dlm_tensor->dl_tensor.device = tf_dlm_context;
  int ndim = tensor->dims();
  dlm_tensor->dl_tensor.ndim = ndim;
  dlm_tensor->dl_tensor.data = tf_dlm_data;
  dlm_tensor->dl_tensor.dtype = tf_dlm_type;

  std::vector<int64_t>* shape_arr = &tf_dlm_tensor_ctx->shape;
  std::vector<int64_t>* stride_arr = &tf_dlm_tensor_ctx->strides;
  shape_arr->resize(ndim);
  stride_arr->resize(ndim, 1);
  for (int i = 0; i < ndim; i++) {
    (*shape_arr)[i] = tensor->dim_size(i);
  }
  for (int i = ndim - 2; i >= 0; --i) {
    (*stride_arr)[i] = (*shape_arr)[i + 1] * (*stride_arr)[i + 1];
  }

  dlm_tensor->dl_tensor.shape = shape_arr->data();
  // There are two ways to represent compact row-major data
  // 1) nullptr indicates tensor is compact and row-majored.
  // 2) fill in the strides array as the real case for compact row-major data.
  // Here we choose option 2, since some frameworks didn't handle the strides
  // argument properly.
  dlm_tensor->dl_tensor.strides = stride_arr->data();

  dlm_tensor->dl_tensor.byte_offset =
      0;  // TF doesn't handle the strides and byte_offsets here
  return static_cast<void*>(dlm_tensor);
}

TFE_TensorHandle* TFE_HandleFromDLPack(void* dlm, TF_Status* status,
                                       TFE_Context* ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSeagerPSdlpackDTcc mht_10(mht_10_v, 526, "", "./tensorflow/c/eager/dlpack.cc", "TFE_HandleFromDLPack");

  DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(dlm);
  DLTensor* dl_tensor = &dlmt->dl_tensor;
  absl::optional<std::string> device_name =
      DeviceNameFromDlContext(dl_tensor->device, status);
  if (!device_name.has_value()) {
    status->status =
        tensorflow::errors::InvalidArgument("Unsupported Device Type");
    return nullptr;
  }
  TF_DataType dtype;
  Status s = TfDataTypeFormDlDataType(dl_tensor->dtype, &dtype);
  if (!s.ok()) {
    status->status = std::move(s);
    return nullptr;
  }
  int num_dims = dl_tensor->ndim;
  const int64_t* dims = dl_tensor->shape;
  void* data = dl_tensor->data;

  size_t total_bytes = dl_tensor->dtype.bits / 8;
  for (int i = 0; i < num_dims; i++) {
    total_bytes *= dims[i];
  }

  if (dl_tensor->strides != nullptr &&
      !IsValidStrideCompactRowMajorData(dl_tensor->shape, dl_tensor->strides,
                                        num_dims)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Invalid strides array from DLPack");
    return nullptr;
  }

  TFE_TensorHandle* handle = TFE_NewTensorHandleFromDeviceMemory(
      ctx, device_name.value().c_str(), dtype, dims, num_dims, data,
      total_bytes, &DeallocatorWrapperFunc, dlmt, status);

  return handle;
}

}  // namespace tensorflow
