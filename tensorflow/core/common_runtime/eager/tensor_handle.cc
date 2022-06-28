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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc() {
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
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle_data.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.h"
#endif  // IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

namespace {
int64_t GetRemoteDeviceIncarnation(Device* device) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_0(mht_0_v, 222, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "GetRemoteDeviceIncarnation");

  if (device == nullptr || device->IsLocal()) return 0;
  return device->attributes().incarnation();
}

string SafeDeviceDebugString(Device* device) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "SafeDeviceDebugString");

  if (device == nullptr) {
    return "[]";
  } else {
    return device->DebugString();
  }
}
}  // namespace

TensorHandle::PackedTensorHandleData::PackedTensorHandleData(
    std::vector<TensorHandle*>&& handles, const TensorShape& shape)
    : handles_(std::move(handles)), shape_(shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_2(mht_2_v, 244, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::PackedTensorHandleData");

  for (auto* handle : handles_) {
    handle->Ref();
  }
}

TensorHandle::PackedTensorHandleData::~PackedTensorHandleData() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_3(mht_3_v, 253, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::~PackedTensorHandleData");

  for (auto* handle : handles_) {
    handle->Unref();
  }
}

Status TensorHandle::PackedTensorHandleData::Shape(TensorShape* shape) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::Shape");

  *shape = shape_;
  return Status::OK();
}

Status TensorHandle::PackedTensorHandleData::NumDims(int* num_dims) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_5(mht_5_v, 270, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::NumDims");

  *num_dims = shape_.dims();
  return Status::OK();
}

Status TensorHandle::PackedTensorHandleData::Dim(int dim_index,
                                                 int64_t* dim) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_6(mht_6_v, 279, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::Dim");

  *dim = shape_.dim_size(dim_index);
  return Status::OK();
}

Status TensorHandle::PackedTensorHandleData::NumElements(
    int64_t* num_elements) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_7(mht_7_v, 288, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::NumElements");

  *num_elements = shape_.num_elements();
  return Status::OK();
}

Status TensorHandle::PackedTensorHandleData::Unprotect() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_8(mht_8_v, 296, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::Unprotect");

  for (auto* handle : handles_) {
    TF_RETURN_IF_ERROR(absl::visit([](auto& data) { return data.Unprotect(); },
                                   handle->data_));
  }
  return Status::OK();
}

bool TensorHandle::PackedTensorHandleData::IsReady() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_9(mht_9_v, 307, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::IsReady");

  {
    tf_shared_lock l(mu_);
    if (!is_poisoned_.ok()) {
      return true;
    }
  }
  for (auto* handle : handles_) {
    if (!handle->IsReady()) {
      return false;
    }
  }
  return true;
}

Status TensorHandle::PackedTensorHandleData::WaitReady(
    const char* caller) const {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("caller: \"" + (caller == nullptr ? std::string("nullptr") : std::string((char*)caller)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_10(mht_10_v, 327, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::WaitReady");

  {
    tf_shared_lock l(mu_);
    if (!is_poisoned_.ok()) {
      return is_poisoned_;
    }
  }
  for (auto* handle : handles_) {
    TF_RETURN_IF_ERROR(handle->WaitReady(caller));
  }
  return Status::OK();
}

void TensorHandle::PackedTensorHandleData::Poison(Status status) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_11(mht_11_v, 343, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::Poison");

  mutex_lock l(mu_);
  is_poisoned_ = status;
}

string TensorHandle::PackedTensorHandleData::DebugString() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_12(mht_12_v, 351, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::DebugString");

  string debug_str = "PackedTensorHandleData: ";
  for (const auto* handle : handles_) {
    debug_str.append(
        absl::StrCat(absl::visit([](auto& data) { return data.DebugString(); },
                                 handle->data_),
                     "; "));
  }
  return debug_str;
}

int TensorHandle::PackedTensorHandleData::NumPackedHandles() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_13(mht_13_v, 365, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::NumPackedHandles");

  return handles_.size();
}

Status TensorHandle::PackedTensorHandleData::ExtractPackedHandle(
    const int index, TensorHandle** handle) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_14(mht_14_v, 373, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PackedTensorHandleData::ExtractPackedHandle");

  if (index < 0 || index >= handles_.size()) {
    return errors::InvalidArgument("Expect an index within [0, ",
                                   handles_.size(), "), but got ", index);
  }
  *handle = handles_.at(index);
  return Status::OK();
}

void TensorHandle::SetResourceHandleDtypeAndShape(
    std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_15(mht_15_v, 386, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::SetResourceHandleDtypeAndShape");

  handle_dtypes_and_shapes_ = std::move(dtypes_and_shapes);
}

Status TensorHandle::GetResourceHandleDtypesAndShapes(
    std::vector<DtypeAndPartialTensorShape>* result) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_16(mht_16_v, 394, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::GetResourceHandleDtypesAndShapes");

  if (dtype != DT_RESOURCE) {
    return errors::InvalidArgument(
        "TensorHandle::GetResourceDtypeAndShape should be called on tensor "
        "handles with data type DT_RESOURCE. Actual tensor: ",
        dtype);
  }

  if (Type() != LOCAL) {
    *result = handle_dtypes_and_shapes_;
    return Status::OK();
  }

  // Wait for this TensorHandle to be ready.
  profiler::TraceMe activity("TensorHandle::GetResourceHandleInfo WaitReady",
                             profiler::TraceMeLevel::kVerbose);
  auto& data = absl::get<LocalTensorHandleData>(data_);
  TF_RETURN_IF_ERROR(data.WaitReady("TensorHandle::GetResourceHandleInfo"));

  *result = handle_dtypes_and_shapes_;
  return Status::OK();
}

int TensorHandle::NumPackedHandles() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_17(mht_17_v, 420, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::NumPackedHandles");

  if (Type() != PACKED) {
    return 0;
  }
  return absl::get<PackedTensorHandleData>(data_).NumPackedHandles();
}

Status TensorHandle::ExtractPackedHandle(const int index,
                                         TensorHandle** handle) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_18(mht_18_v, 431, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::ExtractPackedHandle");

  if (Type() != PACKED) {
    return errors::Internal("Invalid ExtractPackedHandleOnDevice call on a",
                            TypeString(), " handle: ", this);
  }
  return absl::get<PackedTensorHandleData>(data_).ExtractPackedHandle(index,
                                                                      handle);
}

TensorHandle* TensorHandle::CreateLocalHandle(const tensorflow::Tensor& t) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_19(mht_19_v, 443, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::CreateLocalHandle");

  // TODO(b/136608821): Move away from nullptr
  tensorflow::Tensor tensor = t;
  return CreateLocalHandle(std::move(tensor),
                           /*d=*/nullptr,
                           /*op_device=*/nullptr,
                           /*ctx=*/nullptr);
}

TensorHandle* TensorHandle::CreateLocalHandle(tensorflow::Tensor&& t, Device* d,
                                              Device* op_device,
                                              EagerContext* ctx) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_20(mht_20_v, 457, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::CreateLocalHandle");

  return CreateLocalHandle(std::move(t), d, op_device, nullptr, ctx);
}

TensorHandle* TensorHandle::CreateLocalHandle(tensorflow::Tensor&& t, Device* d,
                                              Device* op_device,
                                              Device* resource_device,
                                              EagerContext* ctx) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_21(mht_21_v, 467, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::CreateLocalHandle");

  if (t.dtype() == DT_RESOURCE && t.NumElements() > 0) {
    return new TensorHandle(std::move(t), d, op_device, ctx);
  } else {
    return new TensorHandle(std::move(t), d, op_device, resource_device, ctx);
  }
}

TensorHandle::TensorHandle(tensorflow::Tensor&& t, Device* d, Device* op_device,
                           Device* resource_device, EagerContext* ctx)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(t.dtype()),
      device_((!ctx || d == ctx->HostCPU()) ? nullptr : d),
      op_device_(op_device),
      resource_device_(resource_device),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      ctx_(ctx),
      data_(absl::in_place_type<LocalTensorHandleData>, std::move(t)) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_22(mht_22_v, 488, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::TensorHandle");

  DVLOG(3) << "Creating Local TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_)
           << " tensor: " << t.DeviceSafeDebugString();
}

TensorHandle::TensorHandle(tensorflow::Tensor&& t, Device* d, Device* op_device,
                           EagerContext* ctx)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(DT_RESOURCE),
      device_((!ctx || d == ctx->HostCPU()) ? nullptr : d),
      op_device_(op_device),
      resource_device_(
          GetResourceDevice(t.flat<class ResourceHandle>()(0), ctx)),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      ctx_(ctx),
      handle_dtypes_and_shapes_(
          t.flat<class ResourceHandle>()(0).dtypes_and_shapes()),
      data_(absl::in_place_type<LocalTensorHandleData>, std::move(t)) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_23(mht_23_v, 510, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::TensorHandle");

  DVLOG(3) << "Creating Local TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_)
           << " tensor: " << t.DeviceSafeDebugString();
}


TensorHandle* TensorHandle::CreateEmptyLocalHandle(Device* d, Device* op_device,
                                                   Device* resource_device,
                                                   tensorflow::DataType dtype,
                                                   EagerContext* ctx) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_24(mht_24_v, 523, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::CreateEmptyLocalHandle");

  return new TensorHandle(d, op_device, resource_device, dtype, ctx);
}

TensorHandle::TensorHandle(Device* d, Device* op_device,
                           Device* resource_device, tensorflow::DataType dtype,
                           EagerContext* ctx)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(dtype),
      device_((d == ctx->HostCPU()) ? nullptr : d),
      op_device_(op_device),
      resource_device_(resource_device),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      ctx_(ctx),
      data_(absl::in_place_type<LocalTensorHandleData>) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_25(mht_25_v, 541, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::TensorHandle");

  DVLOG(3) << "Creating empty Local TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_);
}

Status TensorHandle::CreatePackedHandle(std::vector<TensorHandle*>&& handles,
                                        const tensorflow::DataType dtype,
                                        const tensorflow::TensorShape& shape,
                                        const string& device_name,
                                        EagerContext* ctx,
                                        TensorHandle** packed_handle) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_26(mht_26_v, 555, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::CreatePackedHandle");

  if (handles.empty()) {
    return errors::InvalidArgument("Handles should not be empty.");
  }

  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes;
  if (dtype == DT_RESOURCE) {
    TF_RETURN_IF_ERROR(
        handles.at(0)->GetResourceHandleDtypesAndShapes(&dtypes_and_shapes));
  }
  std::vector<string> devices;
  devices.reserve(handles.size());
  for (auto* handle : handles) {
    devices.push_back(handle->op_device() ? handle->op_device()->name()
                                          : ctx->HostCPU()->name());
  }

  CompositeDevice* composite_device = nullptr;
  TF_RETURN_IF_ERROR(ctx->FindOrCreateCompositeDevice(devices, device_name,
                                                      &composite_device));
  *packed_handle =
      new TensorHandle(std::move(handles), composite_device, dtype, shape, ctx);
  (*packed_handle)
      ->SetResourceHandleDtypeAndShape(std::move(dtypes_and_shapes));
  return Status::OK();
}

Status TensorHandle::CreatePackedHandle(std::vector<TensorHandle*>&& handles,
                                        EagerContext* ctx,
                                        TensorHandle** packed_handle) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_27(mht_27_v, 587, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::CreatePackedHandle");

  if (handles.empty()) {
    return errors::InvalidArgument("Handles should not be empty.");
  }

  // Get the dtype and shape from the first handle since all handles have the
  // same dtype and shape.
  tensorflow::DataType dtype = handles.at(0)->dtype;
  tensorflow::TensorShape shape;
  TF_RETURN_IF_ERROR(handles.at(0)->Shape(&shape));
  return CreatePackedHandle(std::move(handles), dtype, shape,
                            /*device_name*/ "", ctx, packed_handle);
}

TensorHandle::TensorHandle(std::vector<TensorHandle*>&& handles, Device* device,
                           const tensorflow::DataType dtype,
                           const tensorflow::TensorShape& shape,
                           EagerContext* ctx)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(dtype),
      device_(device),
      op_device_(device),
      resource_device_(dtype == DT_RESOURCE ? device : nullptr),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      ctx_(ctx),
      data_(absl::in_place_type<PackedTensorHandleData>, std::move(handles),
            shape) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_28(mht_28_v, 617, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::TensorHandle");

  DVLOG(3) << "Creating a packed TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_);
}

#if !defined(IS_MOBILE_PLATFORM)
TensorHandle* TensorHandle::CreateUnshapedRemoteHandle(
    int64_t op_id, int32_t output_num, const string& remote_task,
    tensorflow::DataType dtype, Device* d, EagerContext* ctx,
    const bool unknown_device) {
  return new TensorHandle(op_id, output_num, remote_task, dtype, d, ctx,
                          unknown_device);
}

TensorHandle::TensorHandle(int64_t op_id, int32_t output_num,
                           const string& remote_task,
                           tensorflow::DataType dtype, Device* d,
                           EagerContext* ctx, const bool unknown_device)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(dtype),
      device_(d),
      op_device_(d),
      resource_device_(dtype == DT_RESOURCE ? d : nullptr),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      unknown_device_(unknown_device),
      ctx_(ctx),
      data_(absl::in_place_type<RemoteTensorHandleData>, op_id, output_num,
            remote_task, ctx) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("remote_task: \"" + remote_task + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_29(mht_29_v, 649, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::TensorHandle");

  DVLOG(3) << "Creating Unshaped Remote TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_);
}

TensorHandle* TensorHandle::CreateLazyRemoteHandle(
    int64_t op_id, int32_t output_num, tensorflow::DataType dtype, Device* d,
    const bool is_ready, EagerContext* ctx) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_30(mht_30_v, 659, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::CreateLazyRemoteHandle");

  return new TensorHandle(op_id, output_num, dtype, d, is_ready, ctx);
}

TensorHandle::TensorHandle(int64_t op_id, int32_t output_num,
                           tensorflow::DataType dtype, Device* d,
                           const bool is_ready, EagerContext* ctx)
    : ImmediateExecutionTensorHandle(kEager),
      dtype(dtype),
      device_(d),
      op_device_(d),
      resource_device_(dtype == DT_RESOURCE ? d : nullptr),
      resource_remote_device_incarnation_(
          GetRemoteDeviceIncarnation(resource_device_)),
      ctx_(ctx),
      data_(absl::in_place_type<RemoteTensorHandleData>, op_id, output_num,
            ctx->GetContextViewId(), is_ready) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_31(mht_31_v, 678, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::TensorHandle");

  DVLOG(3) << "Creating Lazy Remote TensorHandle: " << this
           << " device: " << SafeDeviceDebugString(device_);
}
#endif

TensorHandle::~TensorHandle() { DVLOG(3) << "Deleting tensor handle " << this; }

void TensorHandle::Release() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_32(mht_32_v, 689, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::Release");

  DVLOG(3) << "Releasing tensor handle " << this;
  Unref();
}

tensorflow::DataType TensorHandle::DataType() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_33(mht_33_v, 697, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::DataType");
 return dtype; }

bool TensorHandle::IsReady() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_34(mht_34_v, 702, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::IsReady");

  return absl::visit([](auto& data) { return data.IsReady(); }, data_);
}

Status TensorHandle::WaitReady(const char* caller) const {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("caller: \"" + (caller == nullptr ? std::string("nullptr") : std::string((char*)caller)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_35(mht_35_v, 710, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::WaitReady");

  return absl::visit([caller](auto& data) { return data.WaitReady(caller); },
                     data_);
}

TensorHandle::HandleType TensorHandle::Type() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_36(mht_36_v, 718, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::Type");

  if (data_.index() == 0) {
    return LOCAL;
  } else if (data_.index() == 1) {
    return PACKED;
  } else {
    return REMOTE;
  }
}

string TensorHandle::TypeString() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_37(mht_37_v, 731, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::TypeString");

  if (data_.index() == 0) {
    return "LOCAL";
  } else if (data_.index() == 1) {
    return "PACKED";
  } else {
    return "REMOTE";
  }
}

Status TensorHandle::Tensor(const tensorflow::Tensor** t) const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_38(mht_38_v, 744, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::Tensor");

  DVLOG(3) << "Tensor on TensorHandle: " << this;

  if (Type() != LOCAL) {
    return errors::Internal("Invalid Tensor call on a ", TypeString(),
                            " handle: ", this);
  }

  auto& data = absl::get<LocalTensorHandleData>(data_);
  return data.Tensor(t);
}

Status TensorHandle::TensorFromDevice(const Device* d,
                                      const tensorflow::Tensor** t) const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_39(mht_39_v, 760, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::TensorFromDevice");

  DVLOG(3) << "TensorFromDevice on TensorHandle: " << this << " device: " << d;

  if (d == device_) {
    if (Type() != LOCAL) {
      return errors::Internal("Invalid Tensor call on a ", TypeString(),
                              " handle: ", this);
    }

    auto& data = absl::get<LocalTensorHandleData>(data_);
    return data.Tensor(t);
  }

  tf_shared_lock l(mu_);
  auto elem = local_mirrors_.find(d);
  if (elem == local_mirrors_.end()) {
    return errors::Internal("Invalid device: ", d,
                            " in Tensor call to handle: ", this);
  }

  auto& mirror = elem->second;
  return mirror.Tensor(t);
}

Status TensorHandle::TensorValue(const Device* d, tensorflow::TensorValue* t) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_40(mht_40_v, 787, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::TensorValue");

  DVLOG(3) << "TensorValue on TensorHandle: " << this << " device: " << d;

  if (d == device_) {
    if (Type() != LOCAL) {
      return errors::Internal("Invalid TensorValue call on a ", TypeString(),
                              " handle: ", this);
    }

    auto& data = absl::get<LocalTensorHandleData>(data_);
    return data.TensorValue(t);
  }

  tf_shared_lock l(mu_);
  auto elem = local_mirrors_.find(d);
  if (elem == local_mirrors_.end()) {
    return errors::Internal("Invalid device: ", d,
                            " in TensorValue call to handle: ", this);
  }

  auto& mirror = elem->second;
  return mirror.TensorValue(t);
}

Status TensorHandle::WaitUnknownDevice() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_41(mht_41_v, 814, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::WaitUnknownDevice");

  if (unknown_device_) {
    TF_RETURN_IF_ERROR(absl::visit(
        [](auto& data) {
          return data.WaitReady("TensorHandle::UnknownDevice");
        },
        data_));
  }
  return Status::OK();
}

Device* TensorHandle::DeviceOrHostCPU(const EagerContext& ctx) const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_42(mht_42_v, 828, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::DeviceOrHostCPU");

  return (device_ == nullptr) ? ctx.HostCPU() : device_;
}

Status TensorHandle::Shape(tensorflow::TensorShape* shape) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_43(mht_43_v, 835, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::Shape");

  if (!IsReady() && inference_shape_.IsFullyDefined()) {
    bool fill = inference_shape_.AsTensorShape(shape);
    DCHECK(fill);
    return Status::OK();
  } else {
    return absl::visit([shape](auto& data) { return data.Shape(shape); },
                       data_);
  }
}

Status TensorHandle::InferenceShape(
    shape_inference::InferenceContext* const inference_context,
    shape_inference::ShapeHandle* shape_handle) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_44(mht_44_v, 851, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::InferenceShape");

  if (IsReady()) {
    TF_RETURN_IF_ERROR(is_poisoned_);
    std::vector<shape_inference::DimensionHandle> dims_handle;
    int num_dims;
    TF_RETURN_IF_ERROR(NumDims(&num_dims));
    for (int i = 0; i < num_dims; i++) {
      int64_t dims;
      TF_RETURN_IF_ERROR(Dim(i, &dims));
      dims_handle.push_back(inference_context->MakeDim(dims));
    }
    *shape_handle = inference_context->MakeShape(dims_handle);
    return Status::OK();
  } else {
    if (inference_shape_.unknown_rank()) {
      *shape_handle = inference_context->UnknownShape();
      return Status::OK();
    }
    std::vector<shape_inference::DimensionHandle> dims_handle(
        inference_shape_.dims());
    for (int i = 0; i < dims_handle.size(); i++) {
      dims_handle[i] = inference_context->MakeDim(inference_shape_.dim_size(i));
    }
    *shape_handle = inference_context->MakeShape(dims_handle);
    return Status::OK();
  }
}

void TensorHandle::SetInferenceShape(
    shape_inference::InferenceContext* const inference_context,
    const shape_inference::ShapeHandle& shape_handle) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_45(mht_45_v, 884, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::SetInferenceShape");

  auto num_dims = inference_context->Rank(shape_handle);
  std::vector<int64_t> dims;
  if (num_dims == shape_inference::InferenceContext::kUnknownRank) {
    inference_shape_ = PartialTensorShape();
    return;
  }
  DCHECK_GE(num_dims, 0);
  dims.resize(num_dims);
  for (size_t i = 0; i < num_dims; ++i) {
    dims[i] = inference_context->Value(inference_context->Dim(shape_handle, i));
  }
  auto s = PartialTensorShape::MakePartialShape(dims.data(), num_dims,
                                                &inference_shape_);
  DCHECK(s.ok());
}

Status TensorHandle::CopyInferenceShape(TensorHandle* other) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_46(mht_46_v, 904, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::CopyInferenceShape");

  if (IsReady()) {
    TF_RETURN_IF_ERROR(is_poisoned_);
    return Status::OK();
  }
  if (other->IsReady()) {
    TensorShape other_shape;
    TF_RETURN_IF_ERROR(other->Shape(&other_shape));
    inference_shape_ = other_shape;
  } else {
    inference_shape_ = other->inference_shape_;
  }
  return Status::OK();
}

Status TensorHandle::Shape(tensorflow::PartialTensorShape* shape) const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_47(mht_47_v, 922, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::Shape");

  DCHECK(shape != nullptr);
  if (!IsReady() && !inference_shape_.unknown_rank()) {
    *shape = inference_shape_;
    return Status::OK();
  } else {
    auto result = absl::visit(
        [](auto& data) {
          TensorShape shape;
          Status s = data.Shape(&shape);
          return std::make_pair(shape, s);
        },
        data_);
    TF_RETURN_IF_ERROR(result.second);
    *shape = result.first;
  }
  return Status::OK();
}

Status TensorHandle::NumDims(int* num_dims) const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_48(mht_48_v, 944, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::NumDims");

  DCHECK(num_dims != nullptr);
  if (!IsReady() && !inference_shape_.unknown_rank()) {
    *num_dims = inference_shape_.dims();
    return Status::OK();
  } else {
    return absl::visit(
        [num_dims](auto& data) { return data.NumDims(num_dims); }, data_);
  }
}

Status TensorHandle::Dim(int dim_index, int64_t* dim) const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_49(mht_49_v, 958, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::Dim");

  DCHECK(dim != nullptr);
  if (!IsReady() && !inference_shape_.unknown_rank() &&
      inference_shape_.dim_size(dim_index) != -1) {
    *dim = inference_shape_.dim_size(dim_index);
    return Status::OK();
  } else {
    return absl::visit(
        [dim_index, dim](auto& data) { return data.Dim(dim_index, dim); },
        data_);
  }
}

Status TensorHandle::NumElements(int64_t* num_elements) const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_50(mht_50_v, 974, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::NumElements");

  DCHECK(num_elements != nullptr);
  if (!IsReady() && inference_shape_.IsFullyDefined()) {
    *num_elements = inference_shape_.num_elements();
    return Status::OK();
  } else {
    return absl::visit(
        [num_elements](auto& data) { return data.NumElements(num_elements); },
        data_);
  }
}

Status TensorHandle::Unprotect(const Device* d) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_51(mht_51_v, 989, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::Unprotect");

  DVLOG(3) << "Unprotect on TensorHandle: " << this << " device: " << d;

  if (d == device_) {
    return absl::visit([](auto& data) { return data.Unprotect(); }, data_);
  }

  tf_shared_lock l(mu_);
  auto elem = local_mirrors_.find(d);
  if (elem == local_mirrors_.end()) {
    return errors::Internal("Invalid device: ", d,
                            " in Unprotect call to handle: ", this);
  }

  // Check if the handle is non-empty
  auto& mirror = elem->second;
  return mirror.Unprotect();
}

bool TensorHandle::HasLocalMirror(const Device* d) const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_52(mht_52_v, 1011, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::HasLocalMirror");

  DVLOG(3) << "HasLocalMirror on TensorHandle: " << this << " device: " << d;

  tf_shared_lock l(mu_);
  return local_mirrors_.find(d) != local_mirrors_.end();
}

Status TensorHandle::AddEmptyLocalMirror(const Device* d) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_53(mht_53_v, 1021, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::AddEmptyLocalMirror");

  DVLOG(3) << "AddEmptyLocalMirror on TensorHandle: " << this
           << " device: " << d;

  if (d == device_) {
    return errors::Internal("Cannot add mirror for primary device.");
  }

  mutex_lock l(mu_);
  if (local_mirrors_.find(d) != local_mirrors_.end()) {
    return errors::AlreadyExists("Attempted to duplicate a local mirror.");
  }

  local_mirrors_.emplace(std::piecewise_construct, std::forward_as_tuple(d),
                         std::forward_as_tuple());

  return Status::OK();
}

#if !defined(IS_MOBILE_PLATFORM)
Status TensorHandle::RemoteAddress(const Device* d, const bool wait_until_ready,
                                   int64_t* op_id, int32* output_num) const {
  DVLOG(3) << "RemoteAddress on TensorHandle: " << this << " device: " << d
           << " " << d->name();

  if (d != device_) {
    tf_shared_lock l(mu_);
    auto mirror = remote_mirrors_.find(d->name());
    if (mirror != remote_mirrors_.end()) {
      return mirror->second.OpIdAndOutputNum(wait_until_ready, op_id,
                                             output_num);
    }

    return errors::FailedPrecondition(
        "Could not find remote mirror for specified device");
  }

  if (Type() != REMOTE) {
    return errors::InvalidArgument("Primary device is not remote");
  }

  auto& data = absl::get<RemoteTensorHandleData>(data_);
  return data.OpIdAndOutputNum(wait_until_ready, op_id, output_num);
}

bool TensorHandle::HasRemoteMirror(const Device* d,
                                   uint64 context_view_id) const {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_54(mht_54_v, 1070, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::HasRemoteMirror");

  DVLOG(3) << "HasRemoteMirror on TensorHandle: " << this << " device: " << d
           << " " << d->name();

  tf_shared_lock l(mu_);
  auto mirror = remote_mirrors_.find(d->name());
  if (mirror != remote_mirrors_.end()) {
    // Check if mirror is stale
    if (mirror->second.context_view_id() != context_view_id) {
      return false;
    }
    return true;
  }

  return false;
}

bool TensorHandle::HasResourceShapeMirror(const Device* d,
                                          uint64 context_view_id) const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_55(mht_55_v, 1091, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::HasResourceShapeMirror");

  DVLOG(3) << "HasResourceShapeMirror on TensorHandle: " << this
           << " device: " << d << " " << d->name();

  tf_shared_lock l(mu_);
  auto mirror = resource_shape_mirrors_.find(d->name());
  if (mirror != resource_shape_mirrors_.end()) {
    // Check if mirror is stale
    if (mirror->second.context_view_id() != context_view_id) {
      return false;
    }
    return true;
  }
  return false;
}

Status TensorHandle::AddUnshapedRemoteMirror(const Device* d, int64_t op_id,
                                             int output_num,
                                             const string& remote_task,
                                             EagerContext* ctx) {
   std::vector<std::string> mht_56_v;
   mht_56_v.push_back("remote_task: \"" + remote_task + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_56(mht_56_v, 1114, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::AddUnshapedRemoteMirror");

  DVLOG(3) << "AddUnshapedRemoteMirror on TensorHandle: " << this
           << " device: " << d << " " << d->name() << " op_id: " << op_id
           << " output_num: " << output_num;

  mutex_lock l(mu_);
  auto remote_mirror = remote_mirrors_.find(d->name());
  if (remote_mirror != remote_mirrors_.end()) {
    if (remote_mirror->second.context_view_id() >= ctx->GetContextId()) {
      return errors::Internal("Attempted to duplicate a remote mirror.");
    }
    // Remove stale mirror
    remote_mirrors_.erase(remote_mirror);
  }

  remote_mirrors_.emplace(
      std::piecewise_construct, std::forward_as_tuple(d->name()),
      std::forward_as_tuple(op_id, output_num, remote_task, ctx));

  return Status::OK();
}

Status TensorHandle::AddResourceShapeMirror(const Device* d, int64_t op_id,
                                            int output_num, EagerContext* ctx) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_57(mht_57_v, 1140, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::AddResourceShapeMirror");

  DVLOG(3) << "AddResourceShapeMirror on TensorHandle: " << this;

  mutex_lock l(mu_);
  auto mirror = resource_shape_mirrors_.find(d->name());
  if (mirror != resource_shape_mirrors_.end()) {
    if (mirror->second.context_view_id() == ctx->GetContextViewId()) {
      return errors::Internal(
          "Attempted to duplicate a resource shape mirror.");
    }
    // Remove stale mirror
    resource_shape_mirrors_.erase(mirror);
  }

  resource_shape_mirrors_.emplace(
      std::piecewise_construct, std::forward_as_tuple(d->name()),
      std::forward_as_tuple(op_id, output_num, ctx->GetContextViewId(),
                            /*is_ready=*/true));

  return Status::OK();
}

Status TensorHandle::SetRemoteShape(const TensorShape& shape, const Device* d,
                                    uint64 context_view_id) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_58(mht_58_v, 1166, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::SetRemoteShape");

  return SetRemoteShapeAndDevice(shape, d, context_view_id, /*op_device=*/"");
}

Status TensorHandle::SetRemoteShapeAndDevice(const TensorShape& shape,
                                             const Device* d,
                                             uint64 context_view_id,
                                             string op_device) {
   std::vector<std::string> mht_59_v;
   mht_59_v.push_back("op_device: \"" + op_device + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_59(mht_59_v, 1177, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::SetRemoteShapeAndDevice");

  DVLOG(3) << "SetRemoteShape on TensorHandle: " << this << " device: " << d
           << " " << d->name();

  if (d != device_) {
    tf_shared_lock l(mu_);
    auto remote_mirror = remote_mirrors_.find(d->name());
    if (remote_mirror == remote_mirrors_.end()) {
      return Status::OK();
    }
    auto& mirror = remote_mirror->second;
    if (mirror.context_view_id() == context_view_id) {
      return mirror.SetShape(shape);
    } else if (mirror.context_view_id() < context_view_id) {
      return errors::Internal(
          absl::Substitute("Unexpected context_view_id ($0) which should not "
                           "be newer than the "
                           "one ($1) associated to the remote mirror.",
                           context_view_id, mirror.context_view_id()));
    } else {
      LOG(WARNING) << "SetRemoteShape is ignored for a remote mirror that is "
                      "accociated with a newer context_view_id.";
    }
    return Status::OK();
  }

  DCHECK(Type() == REMOTE)
      << "SetRemoteShape is only called on remote handles.";

  auto& data = absl::get<RemoteTensorHandleData>(data_);
  // context_view_id is currently used to validate mirrors. The shape of
  // RemoteTensorHandleData should be set without checking context_view_id.
  // The reason behind it is that for the primary copy of data, if the remote
  // worker / device is removed, the consumer should report a connection error
  // indicating the remote tensor is no longer available.
  // For mirrors, this is not the case because they colocate with the data
  // consuming op/function device, and we (for now) have to aggressively
  // invalidate those copies to avoid any false positives during cluster update.
  if (op_device.empty()) {
    return data.SetShape(shape);
  } else {
    if (!unknown_device_) {
      return errors::Internal("Cannot reset known devices.");
    }
    Device* device;
    TF_RETURN_IF_ERROR(ctx_->FindDeviceFromName(op_device.c_str(), &device));
    device_ = device;
    op_device_ = device;
    resource_device_ = dtype == DT_RESOURCE ? device : nullptr;
    resource_remote_device_incarnation_ =
        GetRemoteDeviceIncarnation(resource_device_);
    string remote_task;
    if (!DeviceNameUtils::GetTaskName(device->parsed_name(), &remote_task)) {
      return errors::InvalidArgument(
          "Unable to find remote task corresponding to device ",
          device->name());
    }
    return data.SetShapeAndRemoteTask(shape, remote_task);
  }
}

void TensorHandle::PoisonRemote(Status status, const Device* d,
                                uint64 context_view_id) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_60(mht_60_v, 1242, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::PoisonRemote");

  DVLOG(3) << "PoisonRemote on TensorHandle: " << this << " device: " << d
           << " " << d->name();

  if (d == device_) {
    DCHECK(Type() == REMOTE)
        << "Poison can only be on remote handles: " << this;

    auto& data = absl::get<RemoteTensorHandleData>(data_);
    data.Poison(status);
  } else {
    tf_shared_lock l(mu_);
    auto mirror = remote_mirrors_.find(d->name());
    if (mirror != remote_mirrors_.end()) {
      if (mirror->second.context_view_id() == context_view_id) {
        mirror->second.Poison(status);
      }
    }
  }
}
#endif

Status TensorHandle::AddLocalMirror(tensorflow::Tensor&& tensor,
                                    const Device* d) {
  if (d == device_) {
    return errors::Internal(
        "Local mirror assign conflicts with primary device.");
  }

  mutex_lock l(mu_);
  auto elem =
      local_mirrors_.emplace(std::piecewise_construct, std::forward_as_tuple(d),
                             std::forward_as_tuple(std::move(tensor)));
  if (!elem.second) {
    return errors::AlreadyExists("Attempted to add existing mirror.");
  }

  return Status::OK();
}

Status TensorHandle::SetTensor(tensorflow::Tensor&& t, const Device* d) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_61(mht_61_v, 1285, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::SetTensor");

  DVLOG(3) << "SetTensor on TensorHandle: " << this << " device: " << d;

  if (d == device_) {
    DCHECK(Type() == LOCAL) << "SetTensor is not called on local handles.";

    if (t.dtype() == DT_RESOURCE && t.NumElements() > 0) {
      auto& resource_handle = t.flat<class ResourceHandle>()(0);
      handle_dtypes_and_shapes_ = resource_handle.dtypes_and_shapes();
    }
    auto& data = absl::get<LocalTensorHandleData>(data_);
    return data.SetTensor(std::move(t));
  } else {
    tf_shared_lock l(mu_);
    auto elem = local_mirrors_.find(d);
    if (elem == local_mirrors_.end()) {
      return errors::Internal(
          "Attempted to set tensor for non-existent local mirror.");
    }

    auto& mirror = elem->second;
    return mirror.SetTensor(std::move(t));
  }

  return Status::OK();
}

void TensorHandle::Poison(Status status, const Device* d) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_62(mht_62_v, 1315, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::Poison");

  DVLOG(3) << "Poison on TensorHandle: " << this << " device: " << d;

  if (d == device_) {
    DCHECK(Type() != REMOTE) << "Poison can only be on local handles: " << this;
    absl::visit([status](auto& data) { data.Poison(status); }, data_);
  } else {
    tf_shared_lock l(mu_);
    auto elem = local_mirrors_.find(d);
    DCHECK(elem != local_mirrors_.end())
        << "Attempted to poison non-existent local mirror, handle: " << this
        << " device: " << d;

    auto& mirror = elem->second;
    mirror.Poison(status);
  }
}

Status TensorHandle::CopyToDevice(const EagerContext& ctx,
                                  tensorflow::Device* d,
                                  tensorflow::Tensor* output) const {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_63(mht_63_v, 1338, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::CopyToDevice");

  tensorflow::Device* dstd = (d == nullptr) ? ctx.HostCPU() : d;
  tensorflow::Device* srcd = DeviceOrHostCPU(ctx);
  const bool dst_cpu = dstd->tensorflow_accelerator_device_info() == nullptr;
  const bool src_cpu = srcd->tensorflow_accelerator_device_info() == nullptr;
  bool is_same_device =
      (srcd == dstd) || (srcd->name() == dstd->name()) || (dst_cpu && src_cpu);

  const tensorflow::Tensor* src = nullptr;
  TF_RETURN_IF_ERROR(Tensor(&src));
  if (is_same_device) {
    *output = *src;
    return Status::OK();
  }
  if (!dst_cpu && (src->dtype() != tensorflow::DT_VARIANT &&
                   !tensorflow::DataTypeCanUseMemcpy(src->dtype()))) {
    return tensorflow::errors::InvalidArgument(
        "Can't copy Tensor with type ",
        tensorflow::DataTypeString(src->dtype()), " to device ", dstd->name(),
        ".");
  }
  tensorflow::AllocatorAttributes attr;
  if (src->dtype() == tensorflow::DT_VARIANT) {
    attr.set_on_host(true);
  }
  tensorflow::Tensor dst(dstd->GetAllocator(attr), src->dtype(), src->shape());
  if (src->shape().num_elements() == 0) {
    *output = dst;
    return Status::OK();
  }
  tensorflow::DeviceContext* src_device_context = nullptr;
  if (!src_cpu) {
    src_device_context =
        srcd->tensorflow_accelerator_device_info()->default_context;
  }
  tensorflow::DeviceContext* dst_device_context = nullptr;
  if (!dst_cpu) {
    dst_device_context =
        dstd->tensorflow_accelerator_device_info()->default_context;
  }
  // TODO(ashankar): The Sync() call below may be more aggressive than
  // necessary. It is based on knowledge of implementation details - that
  // GPU devices are implemented using 3 streams - one for host->device copies,
  // one for device->host copies and one for sending operations to the GPU.
  // With that setup, Sync()ing across all 3 streams should be sufficient
  // but more than necessary (since it waits for operations that might have
  // nothing to do with this tensor to complete).
  TF_RETURN_IF_ERROR(srcd->Sync());
  tensorflow::Notification n;
  tensorflow::Status status;
  tensorflow::CopyTensor::ViaDMA("copy", src_device_context, dst_device_context,
                                 srcd, dstd, tensorflow::AllocatorAttributes(),
                                 tensorflow::AllocatorAttributes(), src, &dst,
                                 0 /*dev_to_dev_stream_index*/,
                                 [&status, &n](const tensorflow::Status& s) {
                                   status = s;
                                   n.Notify();
                                 });
  n.WaitForNotification();
  if (status.ok()) {
    *output = dst;
    return Status::OK();
  }
  return status;
}

Device* GetResourceDevice(const ResourceHandle& handle, EagerContext* ctx) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_64(mht_64_v, 1407, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "GetResourceDevice");

  if (ctx == nullptr) {
    return nullptr;
  }
  Device* device = nullptr;
  if (!ctx->FindDeviceFromName(handle.device().c_str(), &device).ok()) {
    LOG(ERROR) << "Cannot find resource device: " << handle.device() << ".";
    return nullptr;
  }
  return device;
}

const char* TensorHandle::DeviceName(Status* status) const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_65(mht_65_v, 1422, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::DeviceName");

  status->Update(WaitUnknownDevice());
  tensorflow::Device* d = op_device();
  return (d == nullptr) ? "/job:localhost/replica:0/task:0/device:CPU:0"
                        : d->name().c_str();
}

const char* TensorHandle::BackingDeviceName(Status* status) const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_66(mht_66_v, 1432, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::BackingDeviceName");

  status->Update(WaitUnknownDevice());
  tensorflow::Device* d = device();
  return (d == nullptr) ? "/job:localhost/replica:0/task:0/device:CPU:0"
                        : d->name().c_str();
}

const char* TensorHandle::DeviceType(Status* status) const {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_67(mht_67_v, 1442, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::DeviceType");

  status->Update(WaitUnknownDevice());
  tensorflow::Device* d = op_device();
  return (d == nullptr) ? "CPU" : d->parsed_name().type.c_str();
}

int TensorHandle::DeviceId(Status* status) const {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_68(mht_68_v, 1451, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::DeviceId");

  status->Update(WaitUnknownDevice());
  tensorflow::Device* d = op_device();
  return (d == nullptr) ? 0 : d->parsed_name().id;
}

tensorflow::ImmediateExecutionTensorHandle* TensorHandle::Copy() {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTcc mht_69(mht_69_v, 1460, "", "./tensorflow/core/common_runtime/eager/tensor_handle.cc", "TensorHandle::Copy");

  Ref();
  return this;
}

}  // namespace tensorflow
