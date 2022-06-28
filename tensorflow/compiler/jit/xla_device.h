/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// The XlaDevice executes a TensorFlow graph using the XLA linear algebra
// runtime.
//
// Operators assigned to an XlaDevice are compiled into XLA computations.
// Tensors on an XlaDevice are thin wrappers around XLA ScopedShapedBuffers.
//
// XlaDevice is instantiated separately for each XLA backend (e.g., CPU or GPU),
// under different names (e.g., XLA_CPU or XLA_GPU).

#ifndef TENSORFLOW_COMPILER_JIT_XLA_DEVICE_H_
#define TENSORFLOW_COMPILER_JIT_XLA_DEVICE_H_
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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTh {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTh() {
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

#include <set>

#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/xla_device_context.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace tensorflow {

class XlaDevice : public LocalDevice {
 public:
  // Given a tensor, sets `xla::Shape*` the shape of tensor's representation
  // on device, fully padded. On error, the contents of `xla::Shape*`
  // are undefined.
  typedef std::function<Status(const Tensor&, xla::Shape*)> PaddedShapeFn;

  // Wrapper class to store metadata about the XlaDevice, where it can be
  // retrieved e.g., when lazily creating the XlaCompilationCache device.
  class Metadata {
   public:
    Metadata(int device_ordinal, se::Platform* platform,
             const DeviceType& device_type,
             std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
                 shape_determination_fns,
             PaddedShapeFn padded_shape_fn, bool use_multiple_streams);

    // The index of the device on this host.
    int device_ordinal() const;

    se::Platform* platform() const;
    xla::LocalClient* client() const;
    const DeviceType& jit_device_type() const;
    const XlaShapeLayoutHelpers::ShapeDeterminationFns&
    default_shape_determination_fns() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTh mht_0(mht_0_v, 244, "", "./tensorflow/compiler/jit/xla_device.h", "default_shape_determination_fns");

      return shape_determination_fns_.at(0);
    }
    const PaddedShapeFn& padded_shape_fn() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTh mht_1(mht_1_v, 250, "", "./tensorflow/compiler/jit/xla_device.h", "padded_shape_fn");
 return padded_shape_fn_; }

    bool UseMultipleStreams() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTh mht_2(mht_2_v, 255, "", "./tensorflow/compiler/jit/xla_device.h", "UseMultipleStreams");
 return use_multiple_streams_; }

   private:
    const int device_ordinal_;
    const DeviceType device_type_;
    se::Platform* platform_;  // Not owned.
    std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
        shape_determination_fns_;
    PaddedShapeFn padded_shape_fn_;
    const bool use_multiple_streams_;

    TF_DISALLOW_COPY_AND_ASSIGN(Metadata);
  };

  // Sets `*metadata` to the XlaDevice Metadata in the XLA device used by `ctx`.
  static Status GetMetadata(OpKernelContext* ctx, const Metadata** metadata);

  // Sets `*metadata` to the XlaDevice Metadata in the XLA device used by `ctx`.
  static Status GetMetadata(OpKernelConstruction* ctx,
                            const Metadata** metadata);

  // Sets `*metadata` to the XlaDevice Metadata in the XLA device used by
  // `device`.
  static Status GetMetadataFromDevice(DeviceBase* device,
                                      const XlaDevice::Metadata** metadata);

  struct Options {
    // The StreamExecutor platform. Not owned. Must be non-null.
    se::Platform* platform = nullptr;

    // The device name's prefix (e.g., "/task:7")
    string device_name_prefix;

    // The name of the XLA device (e.g., "XLA_CPU")
    string device_name;

    // The number of the device.
    int device_ordinal = -1;

    // The name of the compilation device (e.g., "XLA_CPU_JIT");
    string compilation_device_name;

    // If 'use_multiple_streams' is true, we create separate streams for
    // compute, host-to-device, and device-to-host communication.
    bool use_multiple_streams = false;

    // If true, the XLA devices with the same device ordinal will share the same
    // compute stream. Otherwise each XLA device will having their own compute
    // streams.
    bool use_global_compute_stream = false;

    // A vector of ShapeDeterminationFn (i.e., a bundle of LayoutSelectionFn,
    // ShapeRepresentationFn). Each bundle describes how the on-host shapes of
    // a) argument and return value, for entry computations b) variables, for
    // all computations, should be represented in XLA. Parameters/return values
    // will be shaped according to the function pair, and reshaped back to/from
    // their declared shapes for computations. Must be non-empty.
    std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
        shape_determination_fns;

    // If padded_shape_fn is empty, a default implementation that returns
    // the logical on-device shape without padding is used.
    PaddedShapeFn padded_shape_fn;

    // Set of devices to use. This controls which of the devices on the given
    // platform will have resources allocated. For GPUs this will be
    // filled from visible_gpu_devices list from session configuration.
    absl::optional<std::set<int>> allowed_devices;
  };

  // Creates a new XLA Device.
  XlaDevice(const SessionOptions& session_options, const Options& options);
  ~XlaDevice() override;

  Allocator* GetAllocator(AllocatorAttributes attr) override
      TF_LOCKS_EXCLUDED(mu_);
  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;
  Status Sync() override;
  void Sync(const DoneCallback& done) override;

  Status TryGetDeviceContext(DeviceContext** out_context) override
      TF_LOCKS_EXCLUDED(mu_);

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override TF_LOCKS_EXCLUDED(mu_);

  Status MakeTensorFromProto(XlaDeviceContext* device_context,
                             const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor);

  const Metadata& metadata() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_deviceDTh mht_3(mht_3_v, 352, "", "./tensorflow/compiler/jit/xla_device.h", "metadata");
 return xla_metadata_; }

  // Ensures the DeviceContext associated with this XlaDevice is created and
  // valid (i.e. all streams are ok). If any state is not valid, a new
  // DeviceContext will be created.
  //
  // TODO(b/111859745): The Eager context needs to call this method to recover
  // from failures.
  Status EnsureDeviceContextOk() TF_LOCKS_EXCLUDED(mu_);

  // Two convenient methods to get the underlying device context.
  // Get the default device context, created by the first
  // shape_representation_fn.
  StatusOr<XlaDeviceContext*> GetDeviceContextDefault();
  // Get the device context given the index.
  StatusOr<XlaDeviceContext*> GetDeviceContextWithIndex(int index);

  // Instructs this XlaDevice to set a GpuDeviceInfo, which holds extra
  // information for GPU and TPU devices.
  Status UseGpuDeviceInfo() TF_LOCKS_EXCLUDED(mu_);

  // Instructs this XlaDevice to return 'sync_on_completion' for
  // AllowsSyncOnCompletion().
  void SetAllowsSyncOnCompletion(bool sync_on_completion)
      TF_LOCKS_EXCLUDED(mu_);
  bool AllowsSyncOnCompletion() const override TF_LOCKS_EXCLUDED(mu_);

  // Installs an error handling callback when RefreshStatus sees !status.ok().
  void SetHandleDeviceErrorCallback(std::function<Status()> callback);

  Status RefreshStatus() override TF_LOCKS_EXCLUDED(mu_);

 private:
  StatusOr<xla::LocalClient*> GetOrCreateClient() const;
  Allocator* GetAllocatorLocked(AllocatorAttributes attr)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status EnsureStreamOkLocked(xla::Backend* backend, const string& name,
                              std::shared_ptr<se::Stream>* stream,
                              bool* stream_was_changed)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Return a vector of device context, ordered by the sequence in the given
  // shape_representation_fns.
  StatusOr<std::vector<XlaDeviceContext*>> GetDeviceContextLocked()
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Handles error when RefreshStatus sees !status.ok().
  Status HandleDeviceError();

  mutable mutex mu_;
  // The metadata of this XlaDevice.
  const Metadata xla_metadata_;
  // Which hardware device in the client's platform this XlaDevice controls.
  const int device_ordinal_;
  // The name of the device that is used to compile Ops for this XlaDevice.
  const DeviceType jit_device_name_;
  // The platform for this device.
  se::Platform* const platform_;  // Not owned.
  // Intra-op threads to spawn (from SessionOptions).
  const int intra_op_parallelism_threads_;
  // Memory allocator associated with this device.
  Allocator* xla_allocator_ TF_GUARDED_BY(mu_) = nullptr;  // Not owned.

  // Stream associated with this device. Operations enqueued on this
  // stream are executed on the device. Operations include data
  // copying back and forth between CPU and the device, and
  // computations enqueued by XLA.
  std::shared_ptr<se::Stream> stream_ TF_GUARDED_BY(mu_);
  // If false, only stream_ is valid and all computation and transfers use
  // stream_. If true, computation is performed by stream_ and transfers are
  // performed by host_to_device/device_to_device stream or borrowing a stream
  // for each device to host transfer.
  const bool use_multiple_streams_;
  // If use_multiple_streams_, host to device transfers are performed using this
  // stream.
  std::shared_ptr<se::Stream> host_to_device_stream_ TF_GUARDED_BY(mu_);
  // If use_multiple_streams_, transfers between different devices are performed
  // using these streams.
  std::vector<std::shared_ptr<se::Stream>> device_to_device_streams_
      TF_GUARDED_BY(mu_);

  // See comments in options.
  std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
      shape_determination_fns_;

  // A list of the device context accessed by all users of the XlaDevice, set by
  // calls to EnsureDeviceContextOk. The number of device conetexts is based on
  // the number of shape representation functions in XlaDevice::Options. If
  // gpu_device_info_ is non-null, this pointer is also filled in to that
  // struct. XlaDeviceContext is a ref-counted object.
  std::vector<XlaDeviceContext*> device_contexts_ TF_GUARDED_BY(mu_);

  // Holds extra information for GPU and TPU devices, e.g. the device context.
  bool use_gpu_device_info_ TF_GUARDED_BY(mu_) = false;
  std::unique_ptr<DeviceBase::AcceleratorDeviceInfo> gpu_device_info_
      TF_GUARDED_BY(mu_);

  // Thread pool used for running closures
  std::unique_ptr<thread::ThreadPool> thread_pool_;

  // True if the device allows XlaDevice::Sync to be called on completion
  // regardless of status.
  bool sync_on_completion_ TF_GUARDED_BY(mu_) = true;

  // A callback that will be invoked when RefreshStatus sees a status error.
  std::function<Status()> device_error_callback_ TF_GUARDED_BY(mu_);

  // Set of devices to use. This controls which of the devices on the given
  // platform will have resources allocated. For GPUs this will be
  // filled from visible_gpu_devices list from session configuration.
  absl::optional<std::set<int>> allowed_devices_;

  const bool use_global_compute_stream_;

  // A static vector with device_ordinal as its index, describing the global
  // compute streams used in each XLA device. It is only used if
  // `use_global_compute_stream` in `XlaDevice::Options` is set to true.
  static mutex global_mu_;
  static std::vector<std::shared_ptr<se::Stream>>* global_compute_streams_
      TF_GUARDED_BY(global_mu_);
};

// Builds OpKernel registrations on 'device' for the JIT operators
// registered on 'jit_device'. Returns ownership of a XlaDeviceOpRegistrations
// object that encapsulates the kernel registrations.
struct XlaDeviceOpRegistrations {
  std::vector<std::unique_ptr<kernel_factory::OpKernelRegistrar>>
      op_kernel_registrars;
};
XlaDeviceOpRegistrations* RegisterXlaDeviceKernels(const char* device,
                                                   const char* jit_device);

Status DefaultPaddedShapeFn(const Tensor& tensor, xla::Shape* shape);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_DEVICE_H_
