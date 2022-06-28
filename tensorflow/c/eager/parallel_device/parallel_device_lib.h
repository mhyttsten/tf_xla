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

#ifndef TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_LIB_H_
#define TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_LIB_H_
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
class MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_libDTh {
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
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_libDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_libDTh() {
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


#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace parallel_device {

// Functor for making unique_ptrs slightly more ergonomic. Using
// decltype(delete_fn) in the unique_ptr's second template argument requires
// passing a function pointer to delete_fn when constructing the unique_ptr.
class TensorHandleDeleter {
 public:
  void operator()(TFE_TensorHandle* to_delete) const {
    TFE_DeleteTensorHandle(to_delete);
  }
};

using TensorHandlePtr = std::unique_ptr<TFE_TensorHandle, TensorHandleDeleter>;

class ParallelTensor;
class DeviceThread;

// Forwards operations to `devices`, maintaining ParallelTensor with components
// placed on each underlying device.
class ParallelDevice {
 public:
  // Eager async execution is only supported when remote eager is not in use
  // (b/157523095).
  explicit ParallelDevice(const std::vector<std::string>& devices,
                          const bool is_async = false);

  ~ParallelDevice();

  // Helper to copy a tensor handle from another device once for each component
  // of the ParallelDevice.
  //
  // Sets a bad status and returns a nullptr if `tensor` is already on the
  // ParallelDevice, or if the individual copies fail.
  std::unique_ptr<ParallelTensor> CopyToParallelDevice(TFE_Context* context,
                                                       TFE_TensorHandle* tensor,
                                                       TF_Status* status) const;

  // Construct a parallel tensor consisting of the scalar values from `values`.
  template <typename DataType>
  std::unique_ptr<ParallelTensor> ScalarsFromSequence(
      absl::Span<const DataType> values, TFE_Context* context,
      TF_Status* status) const;

  // A parallel tensor with scalar integers numbering component devices.
  std::unique_ptr<ParallelTensor> DeviceIDs(TFE_Context* context,
                                            TF_Status* status) const;

  // The number of devices operations run on.
  size_t num_underlying_devices() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_libDTh mht_0(mht_0_v, 252, "", "./tensorflow/c/eager/parallel_device/parallel_device_lib.h", "num_underlying_devices");
 return underlying_devices_.size(); }

  // The devices operations run on.
  const std::vector<std::string>& underlying_devices() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_libDTh mht_1(mht_1_v, 258, "", "./tensorflow/c/eager/parallel_device/parallel_device_lib.h", "underlying_devices");

    return underlying_devices_;
  }

  // Takes a description of a single operation being executed on the
  // ParallelDevice, and in turn runs one operation per component device with
  // its corresponding inputs from the input ParallelTensors. Wraps the
  // resulting per-device and per-output TFE_TensorHandles into one
  // ParallelTensor per output of the original operation.
  //
  // Attributes are forwarded to executed operations unmodified.
  //
  // The returned optional has a value if and only if `status` evaluates to
  // TF_OK. Bad statuses are forwarded from underlying `TFE_Execute` calls, or
  // if sanity checks on dtypes/metadata fail.
  absl::optional<std::vector<std::unique_ptr<ParallelTensor>>> Execute(
      TFE_Context* context, const std::vector<ParallelTensor*>& inputs,
      const char* operation_name, const TFE_OpAttrs* attributes,
      int expected_max_outputs, TF_Status* status) const;

  // A non-blocking version of `Execute`. After each call, `Join` must be called
  // before `StartExecute` is called again. Using `StartExecute` with `Join`
  // allows the caller to schedule computation on multiple ParallelDevices
  // without sequencing those operations (first call `StartExecute` on each
  // parallel device, then call `Join` on each; even if some of the `Join`s
  // return a bad status the caller must run all of the `Join`s or any future
  // `StartExecute`s will deadlock).
  //
  // If `is_async=false` (constructor argument), `cancellation_manager` must
  // live until `Join` finishes. If `is_async=true` it must live until `Join` is
  // followed by `TFE_ContextAsyncWait` to clear pending operations. It will be
  // used to cancel all other operations if any fails.
  //
  // Set step_id to configure the step id used for rendezvous creation. step id
  // of value -1 is reserved for global rendezvous and should not be set here.
  void StartExecute(TFE_Context* context,
                    const std::vector<ParallelTensor*>& inputs,
                    const char* operation_name, const TFE_OpAttrs* attributes,
                    int expected_max_outputs,
                    CancellationManager& cancellation_manager,
                    absl::optional<int64_t> step_id = absl::nullopt) const;

  // Blocks until the previous `StartExecute` has run `TFE_Execute` on each
  // device. If is_async=false (constructor argument) this means the ops have
  // run and have results. If is_async=true it means that all of the
  // device-specific executors have scheduled the op.
  //
  // Accepts inferred shapes for outputs (`expected_output_shapes`), which if
  // fully defined will avoid querying the shapes of the underlying
  // TensorHandles when ParallelTensor::Shape is called. This allows async
  // computation to continue without blocking.
  //
  // The return status and value is the same as `Execute`.
  absl::optional<std::vector<std::unique_ptr<ParallelTensor>>> Join(
      const std::vector<PartialTensorShape>& expected_output_shapes,
      TF_Status* status) const;

  void AsyncWait(TFE_Context* context, TF_Status* status) const;

  // Device strings for component devices that only include a
  // worker/task/replica if any of those differ across components. Useful for
  // printing debug messages.
  std::vector<std::string> SummarizeDeviceNames() const;

 private:
  // A sequence of device names, indicating which devices replicated operations
  // are forwarded to.
  const std::vector<std::string> underlying_devices_;
  // A sequence of thread wrappers, one per device, for executing operations in
  // parallel.
  //
  // Conceptually this is a thread pool with one thread per device. It requires
  // less synchronization than a thread pool would for this task, since Execute
  // acquires each thread in order (and so only one Execute will schedule
  // blocking collective operations at a time), and avoids some dynamic
  // allocation/scheduling.
  //
  // TODO(allenl): Keep a map from outer thread to list of inner threads rather
  // than a single list of threads so aliased nested parallel devices don't
  // re-use a thread.
  std::vector<std::unique_ptr<DeviceThread>> device_threads_;
  // A cancellation manager to use if the caller does not provide one. When ops
  // are executed asynchronously this must outlive the queued op, so it can't be
  // function-local to Execute.
  mutable std::unique_ptr<CancellationManager> default_cancellation_manager_;
};

// Contains a tuple of tensors, one on each of the `underlying_devices_` of the
// ParallelDevice.
class ParallelTensor {
 public:
  // Construct a ParallelTensor from TensorHandles placed on the component
  // devices of a ParallelDevice. If called, ParallelTensor::Shape inspects
  // `components` to determine a shape.
  static std::unique_ptr<ParallelTensor> FromTensorHandles(
      const ParallelDevice& parallel_device,
      std::vector<TensorHandlePtr> components, TF_Status* status);
  // Uses the provided shape without additional checks, which avoids blocking
  // when ParallelTensor::Shape is called.
  static std::unique_ptr<ParallelTensor> FromTensorHandles(
      const ParallelDevice& parallel_device,
      std::vector<TensorHandlePtr> components, absl::Span<const int64_t> shape,
      TF_Status* status);

  size_t num_tensors() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_libDTh mht_2(mht_2_v, 365, "", "./tensorflow/c/eager/parallel_device/parallel_device_lib.h", "num_tensors");
 return tensors_.size(); }
  TFE_TensorHandle* tensor(size_t index) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_libDTh mht_3(mht_3_v, 369, "", "./tensorflow/c/eager/parallel_device/parallel_device_lib.h", "tensor");
 return tensors_[index].get(); }

  // If the `shape` argument to `FromTensorHandles` is specified, returns that.
  //
  // Otherwise if all of the tensors have the same shape, returns that via the
  // `shape` output argument. This blocks waiting for async tensors, may return
  // a delayed bad status encountered during async execution, and will return a
  // bad status unless all tensors have the same shape.
  Status Shape(const std::vector<int64_t>** shape) const;
  TF_DataType dtype() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_libDTh mht_4(mht_4_v, 381, "", "./tensorflow/c/eager/parallel_device/parallel_device_lib.h", "dtype");
 return dtype_; }

  // Sets its output argument to a summary of the values of this tensor on every
  // component device.
  Status SummarizeValue(std::string& summary);

 private:
  ParallelTensor(const ParallelDevice& device,
                 std::vector<TensorHandlePtr> tensors,
                 absl::Span<const int64_t> shape, const TF_DataType dtype)
      : device_(device),
        tensors_(std::move(tensors)),
        shape_(std::vector<int64_t>(shape.begin(), shape.end())),
        dtype_(dtype) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_libDTh mht_5(mht_5_v, 397, "", "./tensorflow/c/eager/parallel_device/parallel_device_lib.h", "ParallelTensor");
}
  ParallelTensor(const ParallelDevice& device,
                 std::vector<TensorHandlePtr> tensors, const TF_DataType dtype)
      : device_(device),
        tensors_(std::move(tensors)),
        shape_(absl::nullopt),
        dtype_(dtype) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_libDTh mht_6(mht_6_v, 406, "", "./tensorflow/c/eager/parallel_device/parallel_device_lib.h", "ParallelTensor");
}

  const ParallelDevice& device_;
  const std::vector<TensorHandlePtr> tensors_;
  // Parallel tensors are immutable but compute their shape lazily unless it is
  // provided on construction. The optional has a value if the lazy computation
  // has been completed or the shape was provided on construction.
  mutable absl::optional<std::vector<int64_t>> shape_;
  const TF_DataType dtype_;
};

template <typename DataType>
std::unique_ptr<ParallelTensor> ParallelDevice::ScalarsFromSequence(
    absl::Span<DataType const> values, TFE_Context* context,
    TF_Status* status) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_libDTh mht_7(mht_7_v, 423, "", "./tensorflow/c/eager/parallel_device/parallel_device_lib.h", "ParallelDevice::ScalarsFromSequence");

  std::vector<TensorHandlePtr> components;
  components.reserve(underlying_devices_.size());

  if (values.size() != num_underlying_devices()) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        "Number of values did not match number of underlying devices.");
    return nullptr;
  }
  TF_DataType datatype_enum(
      static_cast<TF_DataType>(DataTypeToEnum<DataType>().value));
  for (int device_index = 0; device_index < num_underlying_devices();
       ++device_index) {
    auto device_value = absl::make_unique<DataType>();
    *device_value = values[device_index];
    std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tensor(
        TF_NewTensor(
            datatype_enum, /*dims=*/nullptr, /*num_dims=*/0,
            device_value.release(), sizeof(DataType),
            [](void* data, size_t, void* arg) {
              delete reinterpret_cast<DataType*>(data);
            },
            nullptr),
        TF_DeleteTensor);
    // TODO(allenl): Here and when executing regular operations, we could hold
    // on to one TFE_Op per device and just call TFE_ResetOp to avoid parsing
    // device names repeatedly.
    std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> const_op(
        TFE_NewOp(context, "Const", status), TFE_DeleteOp);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    TFE_OpSetDevice(const_op.get(), underlying_devices_[device_index].c_str(),
                    status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    TFE_OpSetAttrTensor(const_op.get(), "value", tensor.get(), status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    TFE_OpSetAttrType(const_op.get(), "dtype", datatype_enum);
    TFE_TensorHandle* device_handle;
    int num_outputs = 1;
    TFE_Execute(const_op.get(), &device_handle, &num_outputs, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    components.emplace_back(device_handle);
  }
  return ParallelTensor::FromTensorHandles(*this, std::move(components),
                                           status);
}

}  // namespace parallel_device
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_LIB_H_
