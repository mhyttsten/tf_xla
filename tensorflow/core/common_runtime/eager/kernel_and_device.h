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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh() {
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


// Support for eager execution of TensorFlow kernels.

#include <memory>
#include <unordered_map>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "absl/memory/memory.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/util/managed_stack_trace.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#endif  // IS_MOBILE_PLATFORM

namespace tensorflow {

static constexpr const char* const kOutputsOnOpDevice = "_OutputsOnOpDevice";

class ProcessFunctionLibraryRuntime;
class FunctionLibraryRuntime;

const int64_t kInvalidOpId = -1;

// This struc is used for:
// 1. setting op_id and step_id for single-host remote function scenario, and
// 2. setting step_id for multi-client parallel_device scenario.
struct EagerFunctionParams {
  int64_t op_id = kInvalidOpId;
  absl::optional<int64_t> step_id = absl::nullopt;
};

class EagerKernelArgs : public FunctionArgsInterface {
 public:
  EagerKernelArgs() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_0(mht_0_v, 237, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "EagerKernelArgs");
}

  explicit EagerKernelArgs(int count) : tensor_args_(count) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_1(mht_1_v, 242, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "EagerKernelArgs");
}

  explicit EagerKernelArgs(gtl::InlinedVector<TensorValue, 4>&& tensor_args)
      : tensor_args_(std::move(tensor_args)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_2(mht_2_v, 248, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "EagerKernelArgs");
}

  ~EagerKernelArgs() override{
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_3(mht_3_v, 253, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "~EagerKernelArgs");
};

  bool HasRemoteOrPackedInputs() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_4(mht_4_v, 258, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "HasRemoteOrPackedInputs");
 return false; };
  TensorValue* MutableInput(int i) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_5(mht_5_v, 262, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "MutableInput");
 return &tensor_args_[i]; }

  Status GetLocalArg(const FunctionArgIndex& index, Tensor* val) const override;

  std::vector<Tensor> GetLocalTensors() const override;

  const gtl::InlinedVector<TensorValue, 4>* GetTensorValues() const {
    return &tensor_args_;
  }

 protected:
  gtl::InlinedVector<TensorValue, 4> tensor_args_;
};

typedef absl::variant<Tensor, TensorShape> EagerKernelRet;

// KernelAndDevice encapsulates the logic needed to run a computation eagerly.
// The computation can be a single instantiated kernel (implemented by
// KernelAndDeviceOp below) or a multi-device function (implemented by
// KernelAndDeviceFunc below).
//
// Also see:
// https://www.tensorflow.org/code/tensorflow/core/common_runtime/kernel_benchmark_testlib.h
// and
// https://www.tensorflow.org/code/tensorflow/core/kernels/ops_testutil.h
class KernelAndDevice : public core::RefCounted {
 public:
  // Populates this with a kernel appropriate for 'ndef'.
  //
  // The provided FunctionLibraryRuntime MUST outlive all calls to
  // Run() on the returned KernelAndDevice.
  virtual Status Init(const bool log_device_placement, const NodeDef& ndef,
                      GraphCollector* graph_collector) = 0;

  // Non-multi-device functions are run using regular CallOp and look like
  // primitive operations from KernelAndDevice perspective.
  // `flr` can be nullptr if the operation is not run on any specific device
  // (currently can happen only for multi-device functions).
  KernelAndDevice(
      FunctionLibraryRuntime* flr,
      std::function<void(std::function<void()>)>* runner,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      Device* host_cpu_device)
      : device_(flr == nullptr ? nullptr : flr->device()),
        host_cpu_device_(host_cpu_device),
        flr_(flr),
        collective_executor_(std::move(collective_executor)),
        runner_(runner) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_6(mht_6_v, 312, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "KernelAndDevice");
}

  // Not thread safe.
  ~KernelAndDevice() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_7(mht_7_v, 318, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "~KernelAndDevice");
}

  virtual bool IsFunction() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_8(mht_8_v, 323, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "IsFunction");
 return false; }

  virtual bool IsCrossProcess() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_9(mht_9_v, 328, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "IsCrossProcess");
 return false; }

  // TODO(ashankar): Handle list-valued inputs.
  virtual Status Run(
      ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
      std::vector<EagerKernelRet>* outputs,
      CancellationManager* cancellation_manager,
      const absl::optional<EagerFunctionParams>& eager_func_params,
      const absl::optional<ManagedStackTrace>& stack_trace,
      CoordinationServiceAgent* coordination_service_agent) = 0;

  // Execute kernel asynchronously when applicable. Different from `Run` which
  // blocks the caller thread and waits for the execution of the op/function,
  // `RunAsync` could return before finishing the execution. The `done` callback
  // will be triggered once the op/function execution finishes.
  // Currently, calling RunAsync on ops might not honor the asynchronicity when
  // it is called on an instance with only sync implementation, execute the
  // kernel synchronously and then call the callback with the return status
  // from sync execution.
  virtual void RunAsync(
      ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
      std::vector<EagerKernelRet>* outputs,
      CancellationManager* cancellation_manager,
      const absl::optional<EagerFunctionParams>& eager_func_params,
      CoordinationServiceAgent* coordination_service_agent,
      StatusCallback done) = 0;

  virtual Device* InputDevice(int i) const = 0;
  virtual Device* OutputDevice(int idx) const = 0;
  // If idx'th output is a resource, returns the device backing the resource.
  // Else, returns nullptr.
  virtual Device* OutputResourceDevice(int idx) const = 0;

  // Returns the kernel that will be used to run this.
  // Returns nullptr if this will be run using function library runtime.
  virtual const OpKernel* kernel() const = 0;

  // Returns the device on which this kernel will run. In the case of
  // multi-device functions, this is the default device that is passed to the
  // placer but actual computation can happen on a different set of devices.
  // Also, outputs can be produced on devices different from what this method
  // returns.
  Device* device() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_10(mht_10_v, 373, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "device");
 return device_; }

  virtual const DataTypeVector& input_dtypes() const = 0;
  virtual const DataTypeVector& output_dtypes() const = 0;

  virtual int num_inputs() const = 0;
  virtual int num_outputs() const = 0;
  virtual const string& name() const = 0;

 protected:
  std::function<void(std::function<void()>)>* get_runner() const;

  Device* const device_;               // can be null
  Device* const host_cpu_device_;      // non-null
  FunctionLibraryRuntime* const flr_;  // can be null
  const std::unique_ptr<CollectiveExecutor::Handle> collective_executor_;

 private:
  std::function<void(std::function<void()>)>* const runner_;  // can be null
};

// Represents an op kernel and the device it will be run on.
class KernelAndDeviceOp final : public KernelAndDevice {
 public:
  KernelAndDeviceOp(
      tensorflow::Rendezvous* rendezvous, bool log_memory,
      FunctionLibraryRuntime* flr,
      std::function<void(std::function<void()>)>* runner,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      Device* host_cpu_device)
      : KernelAndDevice(flr, runner, std::move(collective_executor),
                        host_cpu_device),
        rendezvous_(rendezvous),
        log_memory_(log_memory) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_11(mht_11_v, 409, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "KernelAndDeviceOp");
}

  ~KernelAndDeviceOp() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_12(mht_12_v, 414, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "~KernelAndDeviceOp");
}

  Status Init(const bool log_device_placement, const NodeDef& ndef,
              GraphCollector* graph_collector) override;

  Status Run(ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
             std::vector<EagerKernelRet>* outputs,
             CancellationManager* cancellation_manager,
             const absl::optional<EagerFunctionParams>& eager_func_params,
             const absl::optional<ManagedStackTrace>& stack_trace,
             CoordinationServiceAgent* coordination_service_agent) override;

  void RunAsync(ScopedStepContainer* step_container,
                const EagerKernelArgs& inputs,
                std::vector<EagerKernelRet>* outputs,
                CancellationManager* cancellation_manager,
                const absl::optional<EagerFunctionParams>& eager_func_params,
                CoordinationServiceAgent* coordination_service_agent,
                StatusCallback done) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_13(mht_13_v, 435, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "RunAsync");

    // Trivial async implementation on top of the sync version
    done(Run(step_container, inputs, outputs, cancellation_manager,
             eager_func_params, {}, coordination_service_agent));
  }

  const OpKernel* kernel() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_14(mht_14_v, 444, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "kernel");
 return kernel_.get(); }

  Device* InputDevice(int i) const override;
  Device* OutputDevice(int idx) const override;
  Device* OutputResourceDevice(int idx) const override;

  const DataTypeVector& input_dtypes() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_15(mht_15_v, 453, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "input_dtypes");

    return kernel_->input_types();
  }
  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_16(mht_16_v, 459, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "output_dtypes");

    return kernel_->output_types();
  }
  int num_inputs() const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_17(mht_17_v, 465, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "num_inputs");
 return kernel_->num_inputs(); }
  int num_outputs() const override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_18(mht_18_v, 469, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "num_outputs");
 return kernel_->num_outputs(); }
  const string& name() const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_19(mht_19_v, 473, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "name");
 return kernel_->name(); }

 private:
  std::unique_ptr<OpKernel> kernel_;
  bool is_distributed_communication_op_;
  gtl::InlinedVector<AllocatorAttributes, 4> input_alloc_attrs_;
  std::vector<Device*> input_devices_;
  gtl::InlinedVector<AllocatorAttributes, 1> output_alloc_attrs_;
  Rendezvous* const rendezvous_;
  checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_;
  const bool log_memory_;
};

// Represents a multi-device function. Functions can also be run using
// various function-calling kernels including CallOp and PartitionedCallOp.
// In such cases, KernelAndDeviceOp is used.
class KernelAndDeviceFunc : public KernelAndDevice {
 public:
  // `flr` can be nullptr.
  // `pflr` must not be nullptr.
  // `host_cpu_device` must not be nullptr.
  KernelAndDeviceFunc(
      FunctionLibraryRuntime* flr, ProcessFunctionLibraryRuntime* pflr,
      std::vector<Device*> input_devices,
      absl::flat_hash_map<string, const std::vector<string>*> composite_devices,
      std::unordered_map<int, DtypeAndPartialTensorShape>
          input_resource_dtypes_and_shapes,
      std::function<void(std::function<void()>)>* runner,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      Device* host_cpu_device, const string& name,
      const bool outputs_on_op_device,
      const bool allow_small_function_optimizations,
      const bool allow_control_flow_sync_execution,
      const bool shape_inference_on_tfe_dialect_import,
      const bool int_args_and_retvals_on_device,
      absl::optional<string> xla_compile_device_type,
      std::function<Rendezvous*(const int64_t)> rendezvous_creator,
      std::function<int64_t()> get_op_id)
      : KernelAndDevice(flr, runner, std::move(collective_executor),
                        host_cpu_device),
        pflr_(pflr),
        handle_(kInvalidHandle),
        outputs_on_op_device_(outputs_on_op_device),
        allow_small_function_optimizations_(allow_small_function_optimizations),
        allow_control_flow_sync_execution_(allow_control_flow_sync_execution),
        shape_inference_on_tfe_dialect_import_(
            shape_inference_on_tfe_dialect_import),
        int_args_and_retvals_on_device_(int_args_and_retvals_on_device),
        xla_compile_device_type_(xla_compile_device_type),
        input_devices_(std::move(input_devices)),
        composite_devices_(std::move(composite_devices)),
        input_resource_dtypes_and_shapes_(
            std::move(input_resource_dtypes_and_shapes)),
        name_(name),
        rendezvous_creator_(std::move(rendezvous_creator)),
        get_op_id_(std::move(get_op_id)) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_20(mht_20_v, 532, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "KernelAndDeviceFunc");
}

  ~KernelAndDeviceFunc() override;

  bool IsFunction() override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_21(mht_21_v, 539, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "IsFunction");
 return true; };

  bool IsCrossProcess() override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_22(mht_22_v, 544, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "IsCrossProcess");
 return is_cross_process_; }

  Status InstantiateFunc(const bool log_device_placement, const NodeDef& ndef,
                         GraphCollector* graph_collector);

  Status Init(const bool log_device_placement, const NodeDef& ndef,
              GraphCollector* graph_collector) override;

  Status Run(ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
             std::vector<EagerKernelRet>* outputs,
             CancellationManager* cancellation_manager,
             const absl::optional<EagerFunctionParams>& eager_func_params,
             const absl::optional<ManagedStackTrace>& stack_trace,
             CoordinationServiceAgent* coordination_service_agent) override;

  void RunAsync(ScopedStepContainer* step_container,
                const EagerKernelArgs& inputs,
                std::vector<EagerKernelRet>* outputs,
                CancellationManager* cancellation_manager,
                const absl::optional<EagerFunctionParams>& eager_func_params,
                CoordinationServiceAgent* coordination_service_agent,
                StatusCallback done) override;

  const OpKernel* kernel() const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_23(mht_23_v, 570, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "kernel");
 return nullptr; }

  Device* InputDevice(int i) const override;
  Device* OutputDevice(int idx) const override;
  Device* OutputResourceDevice(int idx) const override;

  const DataTypeVector& input_dtypes() const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_24(mht_24_v, 579, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "input_dtypes");
 return input_dtypes_; }
  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_25(mht_25_v, 583, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "output_dtypes");

    return output_dtypes_;
  }
  int num_inputs() const override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_26(mht_26_v, 589, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "num_inputs");
 return input_dtypes_.size(); }
  int num_outputs() const override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_27(mht_27_v, 593, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "num_outputs");
 return output_dtypes_.size(); }
  const string& name() const override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTh mht_28(mht_28_v, 597, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.h", "name");
 return name_; };

 private:
  std::shared_ptr<FunctionLibraryRuntime::Options> PrepareForRun(
      ScopedStepContainer* step_container, std::vector<EagerKernelRet>* outputs,
      CancellationManager* cancellation_manager,
      const absl::optional<EagerFunctionParams>& eager_func_params,
      const absl::optional<ManagedStackTrace>& stack_trace,
      CoordinationServiceAgent* coordination_service_agent);

  ProcessFunctionLibraryRuntime* const pflr_;  // non-null
  FunctionLibraryRuntime::Handle handle_;
  // Indicates whether the function needs to execute cross process.
  bool is_cross_process_;

  // If true, function outputs are explicitly assigned to the default device;
  // if false, the output devices are inferred by pflr_.
  bool outputs_on_op_device_;

  // If True, allow optimizations which should be targeted at a limited
  // set of small functions.  (For example, running kernels synchronously can
  // be faster under some conditions.)
  const bool allow_small_function_optimizations_;

  // If True, allows control nodes to run on the single threaded executor.
  const bool allow_control_flow_sync_execution_;

  // TODO(b/176491312): Remove this if shape inference on import flag is
  // removed. If True, allows mlir roundtrip to run shape inference on import.
  const bool shape_inference_on_tfe_dialect_import_;

  const bool int_args_and_retvals_on_device_;

  const absl::optional<string> xla_compile_device_type_;

  // CPU devices are null. Resource handles' devices are actual backing
  // devices.
  std::vector<Device*> output_devices_;
  // CPU devices are not null. Resource handles' devices are actual backing
  // devices.
  std::vector<Device*> input_devices_;
  // Maps from a CompositeDevice name to a list of physical device names.
  absl::flat_hash_map<string, const std::vector<string>*> composite_devices_;
  std::unordered_map<int, DtypeAndPartialTensorShape>
      input_resource_dtypes_and_shapes_;

  DataTypeVector input_dtypes_;
  DataTypeVector output_dtypes_;
  string name_;

  std::function<Rendezvous*(const int64_t)> rendezvous_creator_;
  std::function<int64_t()> get_op_id_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_KERNEL_AND_DEVICE_H_
