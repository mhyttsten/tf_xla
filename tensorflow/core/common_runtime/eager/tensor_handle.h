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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTh() {
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


#include <algorithm>
#include <cstddef>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "absl/types/variant.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle_data.h"
#include "tensorflow/core/common_runtime/function.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.h"
#endif  // IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/lib/core/stringpiece.h"

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

class EagerContext;

// Associates a Tensor and a Device, used in the eager runtime. Internal version
// of the TFE_TensorHandle struct and the python EagerTensor class
// (unrelated to python TensorHandle).
class TensorHandle : public ImmediateExecutionTensorHandle {
  // TensorHandle for dtype != DT_RESOURCE
  TensorHandle(tensorflow::Tensor&& t, Device* d, Device* op_device,
               Device* resource_device, EagerContext* ctx);
  // TensorHandle for dtype == DT_RESOURCE
  TensorHandle(tensorflow::Tensor&& t, Device* d, Device* op_device,
               EagerContext* ctx);
  TensorHandle(Device* d, Device* op_device, Device* resource_device,
               tensorflow::DataType dtype, EagerContext* ctx);

#if !defined(IS_MOBILE_PLATFORM)
  TensorHandle(int64_t op_id, int32_t output_num, const string& remote_task,
               tensorflow::DataType dtype, Device* device, EagerContext* ctx,
               const bool unknown_device);
  TensorHandle(int64_t op_id, int32_t output_num, tensorflow::DataType dtype,
               Device* device, const bool is_ready, EagerContext* ctx);
#endif  // IS_MOBILE_PLATFORM

 public:
  // TensorHandle with no assigned device
  static TensorHandle* CreateLocalHandle(const tensorflow::Tensor& t);
  static TensorHandle* CreateLocalHandle(tensorflow::Tensor&& t, Device* d,
                                         Device* op_device, EagerContext* ctx);
  static TensorHandle* CreateLocalHandle(tensorflow::Tensor&& t, Device* d,
                                         Device* op_device,
                                         Device* resource_device,
                                         EagerContext* ctx);
  static TensorHandle* CreateEmptyLocalHandle(Device* d, Device* op_device,
                                              Device* resource_device,
                                              tensorflow::DataType dtype,
                                              EagerContext* ctx);

  // Create a handle which packs the given handles of the same dtype and shape.
  // If handles are on different devices, assign the packed handle to a
  // CompositeDevice.
  //
  // The new tensor handle shares ownership of the given handle: their reference
  // count will be increased by one after a call to `CreatePackedHandle`.
  // TODO(b/170414377): Use `TensorHandlePtr` instead.
  static Status CreatePackedHandle(std::vector<TensorHandle*>&& handles,
                                   const tensorflow::DataType dtype,
                                   const tensorflow::TensorShape& shape,
                                   const string& device_name, EagerContext* ctx,
                                   TensorHandle** packed_handle);
  static Status CreatePackedHandle(std::vector<TensorHandle*>&& handles,
                                   EagerContext* ctx,
                                   TensorHandle** packed_handle);

#if !defined(IS_MOBILE_PLATFORM)
  // An unshaped remote handle refers to a tensor on a remote worker. It's not
  // ready until the shape is set. It controls the lifetime of the remote
  // tensor.
  static TensorHandle* CreateUnshapedRemoteHandle(
      int64_t op_id, int32_t output_num, const string& remote_task,
      tensorflow::DataType dtype, Device* d, EagerContext* ctx,
      const bool unknown_device = false);
  // A lazy remote handle refers to a tensor on a remote worker. The lifetime of
  // the remote tensor is controlled by the remote worker, but not by the lazy
  // remote handle. Lazy handles are normally created on a default function
  // device.
  static TensorHandle* CreateLazyRemoteHandle(int64_t op_id, int32_t output_num,
                                              tensorflow::DataType dtype,
                                              Device* d, const bool is_ready,
                                              EagerContext* ctx);
#endif  // IS_MOBILE_PLATFORM

  void Release() override;

  tensorflow::DataType DataType() const override;
  Status Shape(tensorflow::PartialTensorShape* shape) const override;
  Status NumDims(int* num_dims) const override;
  Status NumElements(int64_t* num_elements) const override;
  Status Dim(int dim_index, int64_t* dim) const override;

  const char* DeviceName(Status* status) const override;
  const char* BackingDeviceName(Status* status) const override;
  const char* DeviceType(Status* status) const override;
  int DeviceId(Status* status) const override;
  AbstractTensorInterface* Resolve(Status* status) override;

  ImmediateExecutionTensorHandle* Copy() override;

  // Subclasses may return True to instruct the string formatter
  // to use SummarizeValue instead of the NumPy formatter.
  bool PreferCustomSummarizer() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTh mht_0(mht_0_v, 309, "", "./tensorflow/core/common_runtime/eager/tensor_handle.h", "PreferCustomSummarizer");

    return dtype == DT_VARIANT || dtype == DT_RESOURCE;
  }

  // Return the Tensor from the default device.
  Status Tensor(const tensorflow::Tensor** t) const;
  // Return the Tensor from the specified device which could be either the
  // default device or a local mirror. The device pointer should be nullptr if
  // requesting the HostCPU.
  Status TensorFromDevice(const Device* d, const tensorflow::Tensor** t) const;

  // Return the TensorValue from the specified device which could be either the
  // default device or a local mirror. The device pointer should be nullptr if
  // requesting the HostCPU.
  Status TensorValue(const Device* d, tensorflow::TensorValue* t);

  Device* device() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTh mht_1(mht_1_v, 328, "", "./tensorflow/core/common_runtime/eager/tensor_handle.h", "device");
 return device_; }
  Device* op_device() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTh mht_2(mht_2_v, 332, "", "./tensorflow/core/common_runtime/eager/tensor_handle.h", "op_device");
 return op_device_; }
  Device* resource_device() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTh mht_3(mht_3_v, 336, "", "./tensorflow/core/common_runtime/eager/tensor_handle.h", "resource_device");
 return resource_device_; }
  int64_t resource_remote_device_incarnation() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTh mht_4(mht_4_v, 340, "", "./tensorflow/core/common_runtime/eager/tensor_handle.h", "resource_remote_device_incarnation");

    return resource_remote_device_incarnation_;
  }

  // If the devices are unknown at creation time, block until the actual devices
  // are set (data is ready).
  Status WaitUnknownDevice() const;

  Device* DeviceOrHostCPU(const EagerContext& ctx) const;

  Status Shape(tensorflow::TensorShape* shape);

  Status Unprotect(const Device* d);

  // Checks if a mirror tensor exists for the specified device. Mirrors are only
  // maintained for local devices, like CPUs & GPUs. Note a mirror may be empty,
  // as it is still to be set by an async operation.
  bool HasLocalMirror(const Device* d) const;
  // Add an empty mirror placeholder for the specified device. The expectation
  // is this will be populated by a call to SetTensor.
  Status AddEmptyLocalMirror(const Device* d);
  // Add a local mirror. This will fail if an empty local mirror was previously
  // added. For that case, SetTensor should be used instead.
  Status AddLocalMirror(tensorflow::Tensor&& tensor, const Device* d);

#if !defined(IS_MOBILE_PLATFORM)
  bool HasRemoteMirror(const Device* d, uint64 context_view_id) const;
  bool HasResourceShapeMirror(const Device* d, uint64 context_view_id) const;

  Status AddUnshapedRemoteMirror(const Device* d, int64_t op_id, int output_num,
                                 const string& remote_task, EagerContext* ctx);
  Status AddResourceShapeMirror(const Device* d, int64_t op_id, int output_num,
                                EagerContext* ctx);

  // Return the op_id and output num if the handle refers to a remote tensor.
  // If wait_until_ready is true, block until the remote tensor is ready on the
  // given remote worker.
  Status RemoteAddress(const Device* d, const bool wait_until_ready,
                       int64_t* op_id, int32* output_num) const;

  // Called on an async remote tensor once it's shape has been determined. This
  // transitions the tensor handle from a non-ready to a ready state by
  // replacing the backing data abstraction to allow for the shape to be
  // queried.
  // creating a TensorHandle (e.g. a remote output of a remote function).
  // This method or Poison must be called exactly once for remote tensors that
  // were created without a known shape.
  Status SetRemoteShape(const TensorShape& shape, const Device* d,
                        uint64 context_view_id);
  // If op_device is not empty, reset the devices of a remote tensor which is
  // created without known devices (e.g. function outputs).
  Status SetRemoteShapeAndDevice(const TensorShape& shape, const Device* d,
                                 uint64 context_view_id, string op_device);

  // Poisons either this handle or a remote mirror with error `status`.
  // Poisoning means that the handle will become ready and methods trying
  // to access the remote shape will return this error `status`.
  // Exactly one of SetRemoteShape or PoisonRemote methods must be called on a
  // unshaped handle on a remote device.
  void PoisonRemote(Status status, const Device* d, uint64 context_view_id);
#endif

  // Sets the `tensor` for this async non-ready handle making it ready.
  // This method or Poison must be called exactly once for non-ready async
  // handles to make them ready.
  Status SetTensor(tensorflow::Tensor&& tensor, const Device* d);

  // Poisons either this handle or a local mirror with error `status`.
  // Poisoning means that the handle will become ready and methods trying
  // to access the actual tensor or shape will return this error `status`.
  // Exactly one of SetTensor or Poison methods must be called on a non-ready
  // tensor for a specific device.
  void Poison(Status status, const Device* d);

  // TODO(b/154282629): Consider moving it to EagerContext.
  // Copies to the tensor on the given device `d`, or to host iff `d` is null.
  Status CopyToDevice(const EagerContext& ctx, tensorflow::Device* d,
                      tensorflow::Tensor* output) const;

  Status InferenceShape(
      shape_inference::InferenceContext* const inference_context,
      shape_inference::ShapeHandle* shape_handle);
  void SetInferenceShape(
      shape_inference::InferenceContext* const inference_context,
      const shape_inference::ShapeHandle& shape_handle);
  Status CopyInferenceShape(TensorHandle* other);

  // dtype for the handle. It must be the same as t.dtype() once the handle is
  // ready.
  const tensorflow::DataType dtype;

  enum HandleType { LOCAL = 0, PACKED = 1, REMOTE = 2 };

  HandleType Type() const;
  string TypeString() const;

  void SetResourceHandleDtypeAndShape(
      std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes);

  // If this TensorHandle is 1) a local tensor, and 2) a resource handle,
  // return data types and shapes of the underlying resource.
  Status GetResourceHandleDtypesAndShapes(
      std::vector<DtypeAndPartialTensorShape>* result);

  // Returns the number of packed handles. 0 if the handle type is not PACKED.
  int NumPackedHandles() const;
  // It's called on a packed TensorHandle. Extract a handle with the given
  // index.
  Status ExtractPackedHandle(const int index, TensorHandle** handle) const;

  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTh mht_5(mht_5_v, 454, "", "./tensorflow/core/common_runtime/eager/tensor_handle.h", "classof");

    return ptr->getKind() == kEager;
  }

 private:
  friend class PackedTensorHandleTest;

  TensorHandle(std::vector<TensorHandle*>&& handles, Device* device,
               const tensorflow::DataType dtype,
               const tensorflow::TensorShape& shape, EagerContext* ctx);

  ~TensorHandle() override;

  // The TensorHandleData can either represent a local or remote tensor handle.
  // Further, it can be in a non-ready state. It would become ready with a call
  // to either SetTensor or SetRemoteShape which replaces the underlying data
  // with a ready version of the tensor handle data.
  bool IsReady() const;
  Status WaitReady(const char* caller) const;

  tensorflow::Device* device_;

  // Device in which the op producing this tensor was executed. Equals to
  // device_ for constant tensors.
  // Can be nullptr if the op producing this tensor was a function executed
  // with function library runtime.
  tensorflow::Device* op_device_;

  // If the tensor dtype is DT_RESOURCE, resource_device_ holds the device
  // backing the resource. Else resource_device_ is nullptr.
  tensorflow::Device* resource_device_;
  // Incarnation ID of the resource device if it locates on a remote device, or
  // 0 if it locates on a local device.
  int64_t resource_remote_device_incarnation_;

  // If true, the handle refers to a remote tensor which is created without
  // known devices. The actual devices are set by SetRemoteShape. The devices
  // should be accessed once the handle is ready.
  const bool unknown_device_ = false;

  mutable mutex mu_;

  // Map of local mirrors. This can include both ready and non-ready mirrors.
  std::unordered_map<const tensorflow::Device*, LocalTensorHandleData>
      local_mirrors_ TF_GUARDED_BY(mu_);
#if !defined(IS_MOBILE_PLATFORM)
  // TODO(yujingzhang): Remove resource_shape_mirrors_ once scalable per-replica
  // variable is ready, since we could get the shape locally without remote copy
  // then.
  std::unordered_map<string, RemoteTensorHandleData> resource_shape_mirrors_
      TF_GUARDED_BY(mu_);
  // TODO(gjn): Is std::map the most optimal choice here? Perhaps this should be
  // a fixed size map.
  std::unordered_map<string, RemoteTensorHandleData> remote_mirrors_
      TF_GUARDED_BY(mu_);
#endif

  // `ctx` is only guaranteed to be set if the handle is not "ready". This is
  // typically true when the handle was produced during async execution.
  // `ctx` object is not owned and should outlive this handle.
  //
  // TODO(b/150614042): Reference count EagerContext to ensure that 'device_' of
  // a TensorHandle does not outlive the EagerContext from which it came?
  EagerContext* const ctx_;

  // Does not need synchronization because it can be accessed only after
  // WaitReady() has returned. At that point, is_poisoned_ is immutable.
  Status is_poisoned_;

  // If this TensorHandle 1) is a local tensor, and 2) is a resource handle or
  // refers to a remote resource handle, we store data types and shapes for
  // the underlying resource.
  std::vector<DtypeAndPartialTensorShape> handle_dtypes_and_shapes_;

  // A handle data which refers to multiple TensorHandles of the same dtype and
  // shape.
  class PackedTensorHandleData {
   public:
    // Initialize handle data from list of tensor handles.
    // Ownership of the tensor handles is shared between the
    // `PackedTensorHandleData` and the caller (the reference count for the
    // given handles is incremented).
    // TODO(b/170414377): Use `TensorHandlePtr` instead.
    PackedTensorHandleData(std::vector<TensorHandle*>&& handles,
                           const TensorShape& shape);

    ~PackedTensorHandleData();

    Status Shape(TensorShape* shape) const;
    Status NumDims(int* num_dims) const;
    Status Dim(int dim_index, int64_t* dim) const;
    Status NumElements(int64_t* num_elements) const;
    Status Unprotect();
    bool IsReady() const;
    Status WaitReady(const char* caller) const;
    void Poison(Status status);
    string DebugString() const;

    // Number of packed handles.
    int NumPackedHandles() const;
    // Extract a handle on the given index.
    Status ExtractPackedHandle(const int index, TensorHandle** handle) const;

   private:
    // TODO(b/170414377): Use `TensorHandlePtr` instead.
    const std::vector<TensorHandle*> handles_;
    const TensorShape shape_;

    mutable mutex mu_;
    Status is_poisoned_ TF_GUARDED_BY(mu_);
  };

  // Does not need synchronization because it can be accessed only after
  // WaitReady() has returned. At that point, data_ is immutable.
#if !defined(IS_MOBILE_PLATFORM)
  absl::variant<LocalTensorHandleData, PackedTensorHandleData,
                RemoteTensorHandleData>
      data_;
#else
  absl::variant<LocalTensorHandleData, PackedTensorHandleData> data_;
#endif

  PartialTensorShape inference_shape_;
};

// Returns the device backing the resource. Else, returns nullptr.
Device* GetResourceDevice(const ResourceHandle& handle, EagerContext* ctx);

class TensorHandleInterface : public ImmediateExecutionTensorHandle {
 public:
};

template <typename T>
inline TensorHandle* TensorHandleFromInterface(T* handle) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handleDTh mht_6(mht_6_v, 590, "", "./tensorflow/core/common_runtime/eager/tensor_handle.h", "TensorHandleFromInterface");

  return down_cast<TensorHandle*>(handle);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_H_
