/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_OP_KERNEL_H_
#define TENSORFLOW_CORE_FRAMEWORK_OP_KERNEL_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh() {
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


#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/op.h"  // TODO(b/62899350): Remove
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"  // TODO(b/62899350): Remove
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace Eigen {
struct ThreadPoolDevice;
struct GpuDevice;
}  // end namespace Eigen

namespace tensorflow {

namespace checkpoint {
class TensorSliceReaderCacheWrapper;
}  // namespace checkpoint

class AsyncOpKernel;
class CallFrameInterface;
class DeviceMgr;
class FunctionLibraryRuntime;
class OpKernelConstruction;  // declared below
class OpKernelContext;       // declared below,
class OpRegistryInterface;
class ResourceMgr;
class ScopedStepContainer;
class CollectiveExecutor;
class StepStatsCollectorInterface;
class CoordinationServiceAgent;

// A label that is added to kernels that are JIT compiled. These labels will be
// removed before kernels are looked up, so they can be used without specifying
// the label. This label is a temporary measure to allow JIT kernels to be
// disabled if needed.
extern const char* kJitKernelLabel;
extern const char* kDisableJitKernelsEnvVar;

class OpKernel {
 public:
  // OpKernel won't be instantiated by the scheduler, so you may perform
  // expensive initialization in the descendant's constructor.
  explicit OpKernel(OpKernelConstruction* context);

  // Specialized constructor that allows a kernel implementation to mark itself
  // as a "deferred" op. If true, the executor will provide access to the
  // `OpKernelContext::inc_num_deferred_ops_function()` and
  // `OpKernelContext::dec_num_deferred_ops_function()` methods at run-time.
  OpKernel(OpKernelConstruction* context, bool is_deferred);

  // Specialized constructor that enables the descendant to provide a custom
  // `NodeDef` value. For example, this constructor can be used to provide a
  // stripped-down `NodeDef` that does not contain the full set of attrs (such
  // as tensor values) if the descendant stores them in a different form.
  OpKernel(OpKernelConstruction* context, NodeDef&& custom_def,
           bool is_deferred);

  virtual ~OpKernel();

  // An OpKernel's computation can be either synchronous or
  // asynchronous. All OpKernel Compute() methods must be thread-safe as they
  // may be called concurrently (e.g. by multiple executions of the same graph
  // concurrently).
  //
  // Most OpKernels should compute synchronously. They should
  // subclass OpKernel and override the Compute() method and have it
  // return after completing the supplied work.
  //
  // A synchronous OpKernel *MUST NOT* block the calling thread on a
  // synchronization mechanism (condition variable, Notification, etc.) that
  // will be unblocked by the execution of another OpKernel. Execution may
  // deadlock in that case, because the executor may use a bounded number of
  // threads.
  //
  // If an OpKernel must block on the execution of another OpKernel (e.g. a
  // RecvOp, or a DequeueOp), the implementation *MUST* subclass AsyncOpKernel,
  // and override `AsyncOpKernel::ComputeAsync()`. In addition, because the
  // unblocking kernel may never run (due to an error or cancellation), in most
  // cases the AsyncOpKernel should implement cancellation support via
  // `ctx->cancellation_manager()`.
  //
  // In both cases, implementations of Compute() and ComputeAsync()
  // get inputs and write outputs through the given OpKernelContext
  // and returns a status via context->SetStatus(). They must be
  // thread-safe.

  // Synchronous compute.
  //
  // "context" is guaranteed to be alive until Compute() returns.
  virtual void Compute(OpKernelContext* context) = 0;

  // Returns nullptr iff this op kernel is synchronous.
  virtual AsyncOpKernel* AsAsync() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_0(mht_0_v, 315, "", "./tensorflow/core/framework/op_kernel.h", "AsAsync");
 return nullptr; }

  // Returns true iff this op kernel is considered "expensive". The
  // runtime may use this flag to optimize graph execution for example
  // to "inline" inexpensive kernels.
  virtual bool IsExpensive() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_1(mht_1_v, 323, "", "./tensorflow/core/framework/op_kernel.h", "IsExpensive");
 return expensive_; }

  // Returns a pointer to the tensor stored inside constant ops.
  virtual const Tensor* const_tensor() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_2(mht_2_v, 329, "", "./tensorflow/core/framework/op_kernel.h", "const_tensor");
 return nullptr; }

  // Accessors.
  const NodeDef& def() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_3(mht_3_v, 335, "", "./tensorflow/core/framework/op_kernel.h", "def");
 return props_->node_def; }
  const std::string& name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_4(mht_4_v, 339, "", "./tensorflow/core/framework/op_kernel.h", "name");
 return props_->node_def.name(); }
  absl::string_view name_view() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_5(mht_5_v, 343, "", "./tensorflow/core/framework/op_kernel.h", "name_view");
 return name_view_; }
  const std::string& type_string() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_6(mht_6_v, 347, "", "./tensorflow/core/framework/op_kernel.h", "type_string");
 return props_->node_def.op(); }
  absl::string_view type_string_view() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_7(mht_7_v, 351, "", "./tensorflow/core/framework/op_kernel.h", "type_string_view");
 return type_string_view_; }
  const std::string& requested_input(int i) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_8(mht_8_v, 355, "", "./tensorflow/core/framework/op_kernel.h", "requested_input");

    return props_->node_def.input(i);
  }
  const std::string& requested_device() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_9(mht_9_v, 361, "", "./tensorflow/core/framework/op_kernel.h", "requested_device");

    return props_->node_def.device();
  }

  int num_inputs() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_10(mht_10_v, 368, "", "./tensorflow/core/framework/op_kernel.h", "num_inputs");
 return props_->input_types.size(); }
  DataType input_type(int i) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_11(mht_11_v, 372, "", "./tensorflow/core/framework/op_kernel.h", "input_type");
 return props_->input_types[i]; }
  const DataTypeVector& input_types() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_12(mht_12_v, 376, "", "./tensorflow/core/framework/op_kernel.h", "input_types");
 return props_->input_types; }
  const MemoryTypeVector& input_memory_types() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_13(mht_13_v, 380, "", "./tensorflow/core/framework/op_kernel.h", "input_memory_types");

    return input_memory_types_;
  }

  int num_outputs() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_14(mht_14_v, 387, "", "./tensorflow/core/framework/op_kernel.h", "num_outputs");
 return props_->output_types.size(); }
  DataType output_type(int o) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_15(mht_15_v, 391, "", "./tensorflow/core/framework/op_kernel.h", "output_type");
 return props_->output_types[o]; }
  const DataTypeVector& output_types() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_16(mht_16_v, 395, "", "./tensorflow/core/framework/op_kernel.h", "output_types");
 return props_->output_types; }
  const MemoryTypeVector& output_memory_types() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_17(mht_17_v, 399, "", "./tensorflow/core/framework/op_kernel.h", "output_memory_types");

    return output_memory_types_;
  }

  Status InputRange(StringPiece input_name, int* start, int* stop) const;
  Status OutputRange(StringPiece output_name, int* start, int* stop) const;

  // Returns `true` if and only if this kernel uses deferred execution.
  bool is_deferred() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_18(mht_18_v, 410, "", "./tensorflow/core/framework/op_kernel.h", "is_deferred");
 return is_deferred_; }

  // Returns a trace string for current computation, op name/type and input
  // tensor shape/dtype are encoded for profiler cost analysis. Most OpKernel
  // should use the default implementation.
  virtual std::string TraceString(const OpKernelContext& ctx,
                                  bool verbose) const;

 protected:
  std::string ShapeTraceString(const OpKernelContext& ctx) const;

 private:
  const std::shared_ptr<const NodeProperties> props_;
  const MemoryTypeVector input_memory_types_;
  const MemoryTypeVector output_memory_types_;
  NameRangeMap input_name_map_;
  NameRangeMap output_name_map_;
  const absl::string_view name_view_;
  const absl::string_view type_string_view_;
  const int graph_def_version_;
  const bool is_deferred_;
  bool expensive_;

  TF_DISALLOW_COPY_AND_ASSIGN(OpKernel);
};

class AsyncOpKernel : public OpKernel {
 public:
  using OpKernel::OpKernel;  // Lift OpKernel constructors.

  // Asynchronous compute.
  //
  // Implementations of ComputeAsync() must ensure that `done` is (eventually)
  // called exactly once to signal the completion of the computation. The
  // implementation of ComputeAsync() must not block on the execution of another
  // OpKernel. `done` may be called by the current thread, or by another thread.
  // `context` is guaranteed to stay alive until the `done` callback starts.
  //
  // Since it is possible that the unblocking kernel may never run (due to an
  // error or cancellation), in most cases the AsyncOpKernel should implement
  // cancellation support via `context->cancellation_manager()`.
  //
  // WARNING: As soon as the `done` callback starts, `context` and `this` may be
  // deleted. No code depending on these objects should execute after the call
  // to `done`.
  typedef std::function<void()> DoneCallback;
  virtual void ComputeAsync(OpKernelContext* context, DoneCallback done) = 0;

  AsyncOpKernel* AsAsync() override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_19(mht_19_v, 461, "", "./tensorflow/core/framework/op_kernel.h", "AsAsync");
 return this; }

  void Compute(OpKernelContext* context) override;
};

class OpKernelConstruction {
 public:
  OpKernelConstruction(DeviceType device_type, DeviceBase* device,
                       Allocator* allocator, FunctionLibraryRuntime* flib,
                       ResourceMgr* resource_mgr,
                       const std::shared_ptr<const NodeProperties>& props,
                       const MemoryTypeSlice& input_memory_types,
                       const MemoryTypeSlice& output_memory_types,
                       int graph_def_version, Status* status);

  Env* env() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_20(mht_20_v, 479, "", "./tensorflow/core/framework/op_kernel.h", "env");
 return device_->env(); }

  // Allocation of tensors during kernel construction:
  //
  // It is legal to temporarily allocate scratch tensor storage during
  // Op kernel construction. Scratch tensors should be allocated using
  // allocate_temp below. Some kernels need to keep tensors in between
  // invocations. If such a Tensor is allocated during kernel
  // construction this also must be done using allocate_temp, and the
  // Op may only store the returned Tensor object.

  // Allocates a temporary Tensor of the specified type and shape. The
  // Tensor must not be used after kernel construction is
  // complete. See comment above.
  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp);
  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp, AllocatorAttributes allocator_attr);

  // User-supplied configuration of this operation.
  const NodeDef& def() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_21(mht_21_v, 502, "", "./tensorflow/core/framework/op_kernel.h", "def");
 return props_->node_def; }

  // For inspecting the inputs to this operation.
  int num_inputs() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_22(mht_22_v, 508, "", "./tensorflow/core/framework/op_kernel.h", "num_inputs");
 return props_->input_types.size(); }
  DataType input_type(int i) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_23(mht_23_v, 512, "", "./tensorflow/core/framework/op_kernel.h", "input_type");
 return props_->input_types[i]; }
  const DataTypeSlice& input_types() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_24(mht_24_v, 516, "", "./tensorflow/core/framework/op_kernel.h", "input_types");
 return props_->input_types_slice; }
  const MemoryTypeSlice& input_memory_types() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_25(mht_25_v, 520, "", "./tensorflow/core/framework/op_kernel.h", "input_memory_types");

    return input_memory_types_;
  }

  // For inspecting the outputs expected from this operation.
  int num_outputs() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_26(mht_26_v, 528, "", "./tensorflow/core/framework/op_kernel.h", "num_outputs");
 return props_->output_types.size(); }
  DataType output_type(int i) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_27(mht_27_v, 532, "", "./tensorflow/core/framework/op_kernel.h", "output_type");
 return props_->output_types[i]; }
  const DataTypeSlice& output_types() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_28(mht_28_v, 536, "", "./tensorflow/core/framework/op_kernel.h", "output_types");

    return props_->output_types_slice;
  }
  const MemoryTypeSlice& output_memory_types() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_29(mht_29_v, 542, "", "./tensorflow/core/framework/op_kernel.h", "output_memory_types");

    return output_memory_types_;
  }

  // If expected_inputs == inputs() and expected_outputs == output_types(),
  // returns OK, else returns INVALID_ARGUMENT with an error message.
  // Recommended for Ops with dynamic signatures.
  Status MatchSignature(const DataTypeSlice expected_inputs,
                        const DataTypeSlice expected_outputs);

  // For recording configuration errors during construction.
  void SetStatus(const Status& status);
  const Status& status() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_30(mht_30_v, 557, "", "./tensorflow/core/framework/op_kernel.h", "status");
 return *status_; }

  // Look up the attr with name attr_name and set *value to its value.  If no
  // attr with attr_name is found in def(), or the attr does not have
  // a matching type, a non-ok status will be returned.
  template <class T>
  Status GetAttr(StringPiece attr_name, T* value) const;

  // Return true if the attr_name is defined in def().
  bool HasAttr(StringPiece attr_name) const;

  // Return the device type.
  const DeviceType& device_type() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_31(mht_31_v, 572, "", "./tensorflow/core/framework/op_kernel.h", "device_type");
 return device_type_; }

  // If not nullptr, the kernel can instantiate functions defined in
  // the library. E.g.,
  // CHECK_NOTNULL(function_library())->Instantiate("Foo", ...).
  FunctionLibraryRuntime* function_library() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_32(mht_32_v, 580, "", "./tensorflow/core/framework/op_kernel.h", "function_library");
 return flib_; }

  // Shared resources accessible to this kernel.
  ResourceMgr* resource_manager() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_33(mht_33_v, 586, "", "./tensorflow/core/framework/op_kernel.h", "resource_manager");
 return resource_mgr_; }

  // The GraphDef version whose behavior we should follow.
  int graph_def_version() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_34(mht_34_v, 592, "", "./tensorflow/core/framework/op_kernel.h", "graph_def_version");
 return graph_def_version_; }

  // Helper routines for the OP_REQUIRES macros
  void CtxFailure(const Status& s);
  void CtxFailureWithWarning(const Status& s);
  void CtxFailure(const char* file, int line, const Status& s);
  void CtxFailureWithWarning(const char* file, int line, const Status& s);

  // Unrecommended functions: these are functions that have some
  // current uses but are not recommended for use, and may go away at
  // some future major version release.

  // May be used, e.g., to get GPU handles, etc.
  //
  // Currently only used to call MakeTensorFromProto() for
  // implementing ConstantOp for every device.  See comments
  // on Device::MakeTensorFromProto for longer-term replacement
  // ideas.
  DeviceBase* device() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_35(mht_35_v, 613, "", "./tensorflow/core/framework/op_kernel.h", "device");
 return device_; }

 private:
  const DeviceType device_type_;
  DeviceBase* const device_;
  Allocator* allocator_;
  FunctionLibraryRuntime* flib_;
  ResourceMgr* const resource_mgr_;
  std::shared_ptr<const NodeProperties> props_;
  MemoryTypeSlice input_memory_types_;
  MemoryTypeSlice output_memory_types_;
  const int graph_def_version_;
  Status* status_;

  // Allow access from OpKernel ctor.
  friend class OpKernel;

  TF_DISALLOW_COPY_AND_ASSIGN(OpKernelConstruction);
};

// TODO(mrry): Consider converting to a random_access_iterator, and upgrading
// tensorflow::gtl::iterator_range to make the below container classes
// unnecessary.
template <typename ListType, typename ElementType>
class OpArgIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = ElementType;
  using pointer = ElementType*;
  using const_pointer = const ElementType*;
  using reference = ElementType&;
  using const_reference = const ElementType&;
  using difference_type = ptrdiff_t;

  OpArgIterator(const ListType* list, int i) : list_(list), i_(i) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_36(mht_36_v, 650, "", "./tensorflow/core/framework/op_kernel.h", "OpArgIterator");
}

  bool operator==(const OpArgIterator& rhs) {
    DCHECK(list_ == rhs.list_);
    return i_ == rhs.i_;
  }

  bool operator!=(const OpArgIterator& rhs) {
    DCHECK(list_ == rhs.list_);
    return i_ != rhs.i_;
  }

  OpArgIterator operator++() {  // prefix ++it
    ++i_;
    return *this;
  }

  OpArgIterator operator++(int) {  // postfix it++
    OpArgIterator old_value = *this;
    ++i_;
    return old_value;
  }

  reference operator*() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_37(mht_37_v, 676, "", "./tensorflow/core/framework/op_kernel.h", "*");
 return (*list_)[i_]; }
  pointer operator->() { return &(*list_)[i_]; }

  const_reference operator*() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_38(mht_38_v, 682, "", "./tensorflow/core/framework/op_kernel.h", "*");
 return (*list_)[i_]; }
  const_pointer operator->() const { return &(*list_)[i_]; }

 private:
  const ListType* const list_;
  int i_;
};

// Utility class for representing a list of immutable input tensors
// that are passed to the op as a single named argument.
class OpInputList {
 public:
  typedef OpArgIterator<OpInputList, const Tensor> Iterator;
  OpInputList() : ctx_(nullptr), start_(0), stop_(0) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_39(mht_39_v, 698, "", "./tensorflow/core/framework/op_kernel.h", "OpInputList");
}
  OpInputList(OpKernelContext* ctx, int start, int stop)
      : ctx_(ctx), start_(start), stop_(stop) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_40(mht_40_v, 703, "", "./tensorflow/core/framework/op_kernel.h", "OpInputList");
}
  OpInputList& operator=(const OpInputList& other) = default;
  const Tensor& operator[](int i) const;
  int size() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_41(mht_41_v, 709, "", "./tensorflow/core/framework/op_kernel.h", "size");
 return stop_ - start_; }
  Iterator begin() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_42(mht_42_v, 713, "", "./tensorflow/core/framework/op_kernel.h", "begin");
 return Iterator(this, 0); }
  Iterator end() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_43(mht_43_v, 717, "", "./tensorflow/core/framework/op_kernel.h", "end");
 return Iterator(this, size()); }

 private:
  OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;
};

// Utility class for representing a list of mutable ("ref") input tensors
// that are passed to the op as a single named argument.
class OpMutableInputList {
 public:
  typedef OpArgIterator<OpMutableInputList, Tensor*> Iterator;
  OpMutableInputList(OpKernelContext* ctx, int start, int stop)
      : ctx_(ctx), start_(start), stop_(stop) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_44(mht_44_v, 734, "", "./tensorflow/core/framework/op_kernel.h", "OpMutableInputList");
}
  OpMutableInputList() : ctx_(nullptr), start_(0), stop_(0) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_45(mht_45_v, 738, "", "./tensorflow/core/framework/op_kernel.h", "OpMutableInputList");
}
  OpMutableInputList& operator=(const OpMutableInputList& other) = default;
  Tensor at(int i, bool lock_held);
  mutex* ref_mutex(int i);
  int size() const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_46(mht_46_v, 745, "", "./tensorflow/core/framework/op_kernel.h", "size");
 return stop_ - start_; }
  Iterator begin() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_47(mht_47_v, 749, "", "./tensorflow/core/framework/op_kernel.h", "begin");
 return Iterator(this, 0); }
  Iterator end() const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_48(mht_48_v, 753, "", "./tensorflow/core/framework/op_kernel.h", "end");
 return Iterator(this, size()); }

 private:
  OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;
};

// Utility class for representing a list of output tensors that are
// grouped as a single named output.
class OpOutputList {
 public:
  typedef OpArgIterator<OpOutputList, const Tensor*> Iterator;
  OpOutputList() : ctx_(nullptr), start_(0), stop_(0) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_49(mht_49_v, 769, "", "./tensorflow/core/framework/op_kernel.h", "OpOutputList");
}
  OpOutputList(OpKernelContext* ctx, int start, int stop)
      : ctx_(ctx), start_(start), stop_(stop) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_50(mht_50_v, 774, "", "./tensorflow/core/framework/op_kernel.h", "OpOutputList");
}
  OpOutputList& operator=(const OpOutputList& other) = default;
  Tensor* operator[](int i);
  bool required(int i) const;
  DataType expected_output_dtype(int i) const;
  Status allocate(int i, const TensorShape& shape, Tensor** output);
  void set(int i, const Tensor& tensor);
  void set(int i, Tensor&& tensor);
  void set_ref(int i, mutex* mu, Tensor* tensor_for_ref);
  int size() const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_51(mht_51_v, 786, "", "./tensorflow/core/framework/op_kernel.h", "size");
 return stop_ - start_; }
  Iterator begin() const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_52(mht_52_v, 790, "", "./tensorflow/core/framework/op_kernel.h", "begin");
 return Iterator(this, 0); }
  Iterator end() const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_53(mht_53_v, 794, "", "./tensorflow/core/framework/op_kernel.h", "end");
 return Iterator(this, size()); }

 private:
  OpKernelContext* ctx_;  // not owned
  int start_;
  int stop_;
};

// Holds a tensor or tensor reference. For tensor references, we need
// a mutex to prevent concurrent access to the tensor.
struct TensorValue {
  TensorValue() : mutex_if_ref(nullptr), tensor(nullptr) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_54(mht_54_v, 808, "", "./tensorflow/core/framework/op_kernel.h", "TensorValue");
}
  explicit TensorValue(Tensor* t) : mutex_if_ref(nullptr), tensor(t) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_55(mht_55_v, 812, "", "./tensorflow/core/framework/op_kernel.h", "TensorValue");
}
  TensorValue(mutex* mu, Tensor* t) : mutex_if_ref(mu), tensor(t) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_56(mht_56_v, 816, "", "./tensorflow/core/framework/op_kernel.h", "TensorValue");
}
  Tensor* operator->() const { return tensor; }
  bool is_ref() const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_57(mht_57_v, 821, "", "./tensorflow/core/framework/op_kernel.h", "is_ref");
 return mutex_if_ref != nullptr; }

  // Return the dtype of the Tensor. For references, return the underlying type.
  DataType dtype() const {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_58(mht_58_v, 827, "", "./tensorflow/core/framework/op_kernel.h", "dtype");

    if (is_ref()) {
      return MakeRefType(tensor->dtype());
    } else {
      return tensor->dtype();
    }
  }

  // Return the dtype of the Tensor. For references, return the underlying type.
  // This variation on the dtype() acquires the lock for references.
  //
  // TODO(b/133843385): Disallow dtype modifications
  DataType dtype_safe() const {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_59(mht_59_v, 842, "", "./tensorflow/core/framework/op_kernel.h", "dtype_safe");

    if (is_ref()) {
      tf_shared_lock ml(*mutex_if_ref);
      return MakeRefType(tensor->dtype());
    } else {
      return tensor->dtype();
    }
  }

  mutex* mutex_if_ref;  // nullptr if not a ref, != nullptr if a ref
  Tensor* tensor;
};

// Used to store partitioned graphs from function-calling ops.
struct GraphCollector {
  mutex mu;
  std::vector<GraphDef> partitioned_graphs TF_GUARDED_BY(mu);
  GraphDef raw_graph TF_GUARDED_BY(mu);
  GraphDef optimized_graph TF_GUARDED_BY(mu);

  bool dirty TF_GUARDED_BY(mu);

  GraphCollector() : dirty(false) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_60(mht_60_v, 867, "", "./tensorflow/core/framework/op_kernel.h", "GraphCollector");
}

  void CollectRawGraph(const GraphDef& graph) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_61(mht_61_v, 872, "", "./tensorflow/core/framework/op_kernel.h", "CollectRawGraph");

    mutex_lock ml(mu);
    raw_graph.MergeFrom(graph);
    dirty = true;
  }

  void CollectOptimizedGraph(const GraphDef& graph) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_62(mht_62_v, 881, "", "./tensorflow/core/framework/op_kernel.h", "CollectOptimizedGraph");

    mutex_lock ml(mu);
    optimized_graph.MergeFrom(graph);
    dirty = true;
  }

  void CollectPartitionedGraph(const GraphDef& graph) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_63(mht_63_v, 890, "", "./tensorflow/core/framework/op_kernel.h", "CollectPartitionedGraph");

    mutex_lock ml(mu);
    partitioned_graphs.push_back(graph);
    dirty = true;
  }

  void ClearGraphs() TF_EXCLUSIVE_LOCKS_REQUIRED(mu) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_64(mht_64_v, 899, "", "./tensorflow/core/framework/op_kernel.h", "ClearGraphs");

    raw_graph.Clear();
    optimized_graph.Clear();
    partitioned_graphs.clear();
    dirty = false;
  }

  bool HasUpdatedGraphs() {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_65(mht_65_v, 909, "", "./tensorflow/core/framework/op_kernel.h", "HasUpdatedGraphs");

    mutex_lock ml(mu);
    return dirty;
  }
};

class OpKernelContext {
 public:
  // The first element of a WrappedAllocator is a "base" Allocator and
  // the second element is that Allocator wrapped by a
  // TrackingAllocator
  typedef std::pair<Allocator*, TrackingAllocator*> WrappedAllocator;

  // TODO(zhifengc): Do some cleanup of Params.
  // The Params struct is passed in to initialize an OpKernelContext,
  // and must outlive the OpKernelContext.
  struct Params {
    ~Params() {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_66(mht_66_v, 929, "", "./tensorflow/core/framework/op_kernel.h", "~Params");
 delete eigen_gpu_device; }

    // The step being executed.
    int64_t step_id = 0;

    // Timestamp for the start of graph execution. Used for latency metrics.
    int64_t start_time_usecs = 0;

    // The deadline for the session to complete by. Empty if unspecified.
    absl::optional<absl::Time> deadline;

    // The op kernel being computed.
    OpKernel* op_kernel = nullptr;

    // The device on which the kernel is running.
    DeviceBase* device = nullptr;

    // The Eigen GPU device wrapper, which may include a per-op
    // wrapped allocator. The concrete type of this object depends on
    // the type of this->device, so eigen_gpu_device can't be an
    // inline member and must be heap allocated. However, we don't
    // want to allocate a new eigen_gpu_device for every Op that is
    // executed. Instead this member is allocated on first use using
    // ensure_eigen_gpu_device, and then if the Params structure is
    // re-used for subsequent Ops, the eigen_gpu_device is
    // ReInitialized in the OpKernelContext constructor. Unlike the
    // other pointers in Params, this one is owned by Params.
    PerOpGpuDevice* eigen_gpu_device = nullptr;

    inline void ensure_eigen_gpu_device() {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_67(mht_67_v, 961, "", "./tensorflow/core/framework/op_kernel.h", "ensure_eigen_gpu_device");

      DCHECK(device);
      if (nullptr == eigen_gpu_device) {
        // Surprisingly, MakeGpuDevice will return nullptr if the
        // device is not a GPU device. This is ok, since those devices
        // will never use eigen_gpu_device. It seems better to have
        // ensure_eigen_gpu_device fall through and regenerate the
        // nullptr every time an OpKernelContext is instantiated, than
        // to do an unnecessary allocation of a dummy eigen GPU
        // device for CPU device Ops.
        eigen_gpu_device = device->MakeGpuDevice();
      }
    }

    bool track_allocations = false;
    bool log_memory = false;

    // Array indexed by output number for this node
    const AllocatorAttributes* output_attr_array = nullptr;

    // Shared resources accessible by this op kernel invocation.
    ResourceMgr* resource_manager = nullptr;

    // Per-step resources accessible by this op kernel invocation should be
    // stored in this container..
    ScopedStepContainer* step_container = nullptr;

    // Mechanism used by this op kernel invocation to communicate with
    // computations running on other devices.
    RendezvousInterface* rendezvous = nullptr;

    // Mechanism for executing a collective op that needs to coordinate
    // with parallel instances running on other devices.
    CollectiveExecutor* collective_executor = nullptr;

    // The session state for this op.
    SessionState* session_state = nullptr;

    // Unique session identifier. Can be empty.
    std::string session_handle;

    // Metadata about the session. Can be nullptr.
    const SessionMetadata* session_metadata = nullptr;

    // The tensor store for this op.
    TensorStore* tensor_store = nullptr;

    // Mechanism used by this op kernel invocation to register a callback
    // for its cancellation.
    CancellationManager* cancellation_manager = nullptr;

    // Inputs to this op kernel.
    const gtl::InlinedVector<TensorValue, 4>* inputs = nullptr;
    bool is_input_dead = false;

    const gtl::InlinedVector<AllocatorAttributes, 4>* input_alloc_attrs =
        nullptr;

    // Device context.
    DeviceContext* op_device_context = nullptr;

    // Control-flow op supports.
    FrameAndIter frame_iter;

    // Function call supports.
    CallFrameInterface* call_frame = nullptr;
    FunctionLibraryRuntime* function_library = nullptr;
    std::function<void(std::function<void()>)>* runner = nullptr;
    StepStatsCollectorInterface* stats_collector = nullptr;
    GraphCollector* graph_collector = nullptr;
    bool run_all_kernels_inline = false;
    const std::string* executor_type = nullptr;

    // TensorSliceReaderCache support.
    checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache = nullptr;

    // Support for forwarding reservations (used by ScopedAllocator).
    static constexpr int kNeverForward = -2;
    static constexpr int kNoReservation = -1;
    // Values in [0,...) represent reservations for the indexed output.
    const int* forward_from_array = nullptr;

    // For tracking actively running deferred ops.
    std::function<void()> inc_num_deferred_ops_function;
    std::function<void()> dec_num_deferred_ops_function;

    absl::optional<ManagedStackTrace> stack_trace = {};

    // For implementing `OpKernelContext::output_required()`. If null, all
    // outputs are required.
    bool* outputs_required_array = nullptr;

    // For access to distributed coordination service.
    CoordinationServiceAgent* coordination_service_agent = nullptr;
  };

  // params must outlive the OpKernelContext.
  explicit OpKernelContext(Params* params);
  OpKernelContext(Params* params, int num_outputs);
  ~OpKernelContext();

  Env* env() const {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_68(mht_68_v, 1065, "", "./tensorflow/core/framework/op_kernel.h", "env");
 return params_->device->env(); }

  int64_t step_id() const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_69(mht_69_v, 1070, "", "./tensorflow/core/framework/op_kernel.h", "step_id");
 return params_->step_id; }

  int64_t start_time_usecs() const {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_70(mht_70_v, 1075, "", "./tensorflow/core/framework/op_kernel.h", "start_time_usecs");
 return params_->start_time_usecs; }

  // The deadline for the session to complete by. Empty if unspecified in
  // RunOptions.
  absl::optional<absl::Time> deadline() const { return params_->deadline; }

  const OpKernel& op_kernel() const {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_71(mht_71_v, 1084, "", "./tensorflow/core/framework/op_kernel.h", "op_kernel");
 return *params_->op_kernel; }

  // Stack trace of where the op was defined (if defined in eager mode).
  const absl::optional<ManagedStackTrace>& stack_trace() const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_72(mht_72_v, 1090, "", "./tensorflow/core/framework/op_kernel.h", "stack_trace");

    return params_->stack_trace;
  }

  // Input/output signature.

  int num_inputs() const {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_73(mht_73_v, 1099, "", "./tensorflow/core/framework/op_kernel.h", "num_inputs");
 return params_->inputs->size(); }
  DataType input_dtype(int index) const;
  Status input_dtype(StringPiece name, DataType* dtype) const;
  MemoryType input_memory_type(int index) const;

  int num_outputs() const {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_74(mht_74_v, 1107, "", "./tensorflow/core/framework/op_kernel.h", "num_outputs");
 return outputs_.size(); }
  DataType expected_output_dtype(int index) const;
  MemoryType output_memory_type(int index) const;

  // Input

  // Returns an immutable input tensor. May only be used for non-Ref
  // inputs. For Ref inputs use mutable_input below.
  // REQUIRES: !IsRefType(input_dtype(index))
  // TODO(mrry): Convert this to return Status.
  const Tensor& input(int index) const;

  // Returns the named immutable input tensor in "tensor", as defined
  // in the OpDef. May only be used for non-Ref inputs. For Ref inputs
  // use mutable_input below.
  // REQUIRES: !IsRefType(input_dtype(index))
  // REQUIRES: the named input must not be a list.
  Status input(StringPiece name, const Tensor** tensor);

  // Returns the named list-valued immutable input in "list", as
  // defined in the OpDef.  If the named output is not list-valued,
  // returns a one-element list. May only be used for non-Ref
  // inputs. For Ref inputs use mutable_input below.
  // REQUIRES: !IsRefType(input_dtype(index))
  Status input_list(StringPiece name, OpInputList* list);

  // For mutable inputs, use the following together to make sure there
  // is no concurrent access to mutable_input(), e.g.:
  // {
  //   Tensor& t = context->mutable_input(index);
  //   mutex_lock lock(*context->input_ref_mutex(index));
  //   // modify the values in t
  // }
  // REQUIRES: IsRefType(input_dtype(index))
  Status input_ref_mutex(StringPiece name, mutex** out_mutex);

  // Returns a mutable input tensor. Must be used to access Ref
  // inputs.  REQUIRES: IsRefType(input_dtype(index)). The caller may
  // modify the values stored in the Tensor buffer, and modifications
  // will be visible to other Ops reading the same ref tensor. If
  // !lock_held the input mutex will be acquired before returning the
  // Tensor.
  // TODO(mrry): Convert this to return Status.
  Tensor mutable_input(int index, bool lock_held);

  // Returns the named mutable input tensor in "tensor", as defined in
  // the OpDef. Must be used to access Ref inputs. The values stored
  // in the Tensor buffer may be modified, and modifications will be
  // visible to other Ops reading the same ref tensor. If !lock_held
  // the input mutex will be acquired before returning the Tensor.
  // REQUIRES: the named input must not be a list.
  // REQUIRES: the named input must be a ref tensor.
  Status mutable_input(StringPiece name, Tensor* tensor, bool lock_held);

  // Returns the named list-valued mutable input in "list", as defined
  // in the OpDef.  If the named input is not list-valued, returns a
  // one-element list. Must be used to access Ref inputs. The values
  // stored in the Tensor buffer may be modified, and modifications
  // will be visible to other Ops reading the same ref tensor.
  // REQUIRES: the named input must be a ref tensor.
  Status mutable_input_list(StringPiece name, OpMutableInputList* list);

  // Replace the corresponding Ref Input to use the storage buffer
  // used by tensor. If !lock_held the input mutex will be acquired
  // before returning the Tensor.
  // REQUIRES: IsRefType(input_dtype(index)).
  void replace_ref_input(int index, const Tensor& tensor, bool lock_held);

  // Replace the corresponding named Ref Input to use the storage
  // buffer used by tensor. If !lock_held the input mutex will be
  // acquired before returning the Tensor.
  // REQUIRES: IsRefType(input_dtype(index)).
  Status replace_ref_input(StringPiece name, const Tensor& tensor,
                           bool lock_held);

  // Deletes the Tensor object used as the Ref Input at
  // input_index. This is not usually necessary and should be used
  // with caution. If !lock_held the input mutex will be acquired
  // before returning the Tensor.
  // REQUIRES: IsRefType(input_dtype(input_index)).
  void delete_ref_input(int input_index, bool lock_held);

  // Return true if there is input at the given index. An operator has no
  // input at index if its tensor is null. This is primarily used by the
  // merge operator.
  // TODO(mrry): Convert this to return Status.
  bool has_input(int index) const;

  // Returns true if all inputs are the same shape, otherwise sets the
  // status to a non-OK value and returns false.
  // Usage: if (!context->ValidateInputsAreSameShape(this)) return;
  bool ValidateInputsAreSameShape(OpKernel* op);

  // If non-null, kernels should populate with any partition subgraphs created.
  GraphCollector* graph_collector() {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_75(mht_75_v, 1204, "", "./tensorflow/core/framework/op_kernel.h", "graph_collector");
 return params_->graph_collector; }

  // If True, hint that all kernels in functions called by this kernel, should
  // be treated as "inexpensive", and hence executed on the scheduling thread.
  bool run_all_kernels_inline() const {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_76(mht_76_v, 1211, "", "./tensorflow/core/framework/op_kernel.h", "run_all_kernels_inline");

    return params_->run_all_kernels_inline;
  }

  // Returns the registered name for the executor type that is executing the
  // current kernel. If empty, the default executor is used.
  const std::string& executor_type() const;

  // Input to output forwarding.

  // Set the output Ref Tensor at output_index to be an alias of the
  // input Ref Tensor at input_index.
  // REQUIRES: IsRefType(input_dtype(input_index)).
  // REQUIRES: IsRefType(output_dtype(output_index)).
  void forward_ref_input_to_ref_output(int input_index, int output_index);

  // Returns true when an alias to input[input_index], reshaped to output_shape,
  // which is safe to use for in-place computation was written to *output.
  // Returns false if input[input_index] has a refcount greater than one, or if
  // its type does not match the expected output type of output[output_index],
  // or the number of elements in input[input_index] does not equal the number
  // of elements in output_shape.
  bool forward_input_to_output_with_shape(int input_index, int output_index,
                                          const TensorShape& output_shape,
                                          Tensor** output) TF_MUST_USE_RESULT;
  Status forward_input_to_output_with_shape(StringPiece input_name,
                                            StringPiece output_name,
                                            const TensorShape& output_shape,
                                            Tensor** output) TF_MUST_USE_RESULT;

  // Returns a pointer to a Tensor aliasing the underlying buffer backing
  // input[input_index] iff
  //   * input[input_index] is not a ref,
  //   * the data type, shape, memory type, and allocator attributes of
  //     input[input_index] are compatible with those given in dtype, shape,
  //     memory_type, and attr,
  //   * refcount on the underlying buffer is one.
  //   * Either there is no forwarding reservation for either input_index
  //     or output_index or the specified input is reserved for the specified
  //     output. More precisely:
  //
  //     These cases mean neither input nor output has a reservation:
  //        forward_from_array = nullptr
  //     OR (input_index is not in forward_from_array AND
  //         (output_index == kNoReservation OR
  //          forward_from_array[output_index] == kNoReservation))
  //
  //     This case means that input_index is reserved for output_index:
  //        forward_from_array[output_index] == input_index
  //
  //     This case means the output is reserved to always be allocated,
  //     never assigned a forwarded input:
  //        forward_from_array[output_index] == kNeverForward
  //
  // Otherwise returns nullptr.
  // NOTE: For Cuda kernels that read inputs using the __ldg() intrinsic,
  // forwarding is only safe if there are no reads via __ldg() after writes
  // to the same address.
  std::unique_ptr<Tensor> forward_input(
      int input_index, int output_index, DataType output_dtype,
      const TensorShape& output_shape, MemoryType output_memory_type,
      const AllocatorAttributes& output_attr) TF_MUST_USE_RESULT;

  // Tries to forward one of the inputs given in input_indices to
  // output[output_index]. If none of the given inputs can be forwarded, calls
  // allocate_output() to allocate a new output buffer. The index of the
  // forwarded input will be assign to output argument forwarded_input (if it's
  // not nullptr). If no inputs are forwarded, forwarded_input will be assigned
  // -1.
  Status forward_input_or_allocate_output(
      gtl::ArraySlice<int> candidate_input_indices, int output_index,
      const TensorShape& output_shape, Tensor** output,
      int* forwarded_input = nullptr) TF_MUST_USE_RESULT;
  Status forward_input_or_allocate_output(
      gtl::ArraySlice<StringPiece> candidate_input_names,
      StringPiece output_name, const TensorShape& output_shape,
      Tensor** output) TF_MUST_USE_RESULT;

  // Tries to reuse one of the inputs given in input_indices as a temporary.
  // If none of the given inputs can be forwarded, calls
  // allocate_temp() to allocate a new temporary buffer.
  Status forward_input_or_allocate_temp(
      gtl::ArraySlice<int> candidate_input_indices, DataType type,
      const TensorShape& shape, const AllocatorAttributes& allocator_attr,
      Tensor* out_temp) TF_MUST_USE_RESULT;

  Status forward_input_or_allocate_temp(
      gtl::ArraySlice<int> candidate_input_indices, DataType type,
      const TensorShape& shape, Tensor* out_temp) TF_MUST_USE_RESULT {
    return forward_input_or_allocate_temp(candidate_input_indices, type, shape,
                                          AllocatorAttributes(), out_temp);
  }

  // Output

  // Returns the named list-valued output in "list", as defined in the OpDef.
  // If the named output is not list-valued, returns a one-element list.
  Status output_list(StringPiece name, OpOutputList* list);

  // If output_required(index) returns true, the OpKernel's Compute() method
  // should call allocate_output(index, ...), set_output(index, ...),
  // set_output_ref(index, ...), or set the status to a non-ok value.
  // If it returns false, it may output, but is not required to do so.
  bool output_required(int index) const {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_77(mht_77_v, 1317, "", "./tensorflow/core/framework/op_kernel.h", "output_required");

    return !params_->outputs_required_array ||
           params_->outputs_required_array[index];
  }

  // If output_expects_forwarding returns true, the OpKernel's Compute() method
  // should not allocate the output with allocate_output but instead needs to
  // use forward_input.
  bool output_expects_forwarding(int index) const {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_78(mht_78_v, 1328, "", "./tensorflow/core/framework/op_kernel.h", "output_expects_forwarding");

    return params_->forward_from_array != nullptr &&
           params_->forward_from_array[index] >= 0;
  }

  // Allocation of tensors during kernel execution inside the Compute
  // method:
  //
  // There are two methods to allocate Tensors when an Op kernel
  // executes.
  //
  // 1) allocate_output. This should be used to allocate any tensor
  // that is going to be used as an output from the Op at the end of
  // the current execution. The caller indicates which output the
  // Tensor will be assigned to, and the call returns the
  // newly-allocated Tensor. The Tensor can subsequently be assigned
  // to during kernel execution, and will be used as the designated
  // output when the kernel execution completes.
  //
  // 2) allocate_temp. This should be used to allocate any scratch
  // storage that is needed while the kernel is executing, and will
  // not be retained by the Op.
  //
  // In some cases a Tensor needs to be used as an output even though
  // it was previously allocated elsewhere. The Tensor may have been
  // passed as an input, or stored in a Tensor during a
  // previous kernel execution, or allocated earlier in the kernel
  // execution at a time when it was not known which output it would
  // be assigned to. In this case the kernel can use set_output or
  // set_output_ref to indicate that the tensor should be used as the
  // designated output. It is legal to use any previously-allocated
  // Tensor as an argument to set_output or set_output_ref, including
  // Tensors allocated via allocate_temp. There may be a performance
  // penalty to using a Tensor that was not allocated using
  // allocate_output. This is because allocate_output uses the
  // AllocatorAttributes stored in output_attr_array for the
  // designated output. In some cases, using the wrong attributes may
  // cause an extra copy of the Tensor's buffer.

  // Allocates output for the specified output index with shape.
  // OpKernelContext retains ownership of the returned pointer. See
  // comment above.
  //
  // If memory allocation fails, returns an error status.
  //
  // REQUIRES: !IsRefType(expected_output_dtype(index))
  Status allocate_output(int index, const TensorShape& shape,
                         Tensor** tensor) TF_MUST_USE_RESULT;
  Status allocate_output(StringPiece name, const TensorShape& shape,
                         Tensor** tensor) TF_MUST_USE_RESULT;
  // The following methods use the supplied attributes instead of
  // those in output_attr_array. The caller is responsible for
  // ensuring that the attributes are "compatible" with the
  // output_attr_array, e.g. the tensor is allocated on the correct
  // device. See comment above.
  Status allocate_output(int index, const TensorShape& shape, Tensor** tensor,
                         AllocatorAttributes attr) TF_MUST_USE_RESULT;
  Status allocate_output(StringPiece name, const TensorShape& shape,
                         Tensor** tensor,
                         AllocatorAttributes attr) TF_MUST_USE_RESULT;

  // Allocates a temporary Tensor of the specified type and
  // shape. Devices such as GPUs that enqueue Ops for lazy execution
  // may retain references to the temporary tensors after the Op's
  // Compute method has run. See comment above.
  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp, AllocatorAttributes allocator_attr,
                       const AllocationAttributes& allocation_attr);
  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp, AllocatorAttributes allocator_attr) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_79(mht_79_v, 1400, "", "./tensorflow/core/framework/op_kernel.h", "allocate_temp");

    return allocate_temp(type, shape, out_temp, allocator_attr,
                         AllocationAttributes());
  }
  Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_80(mht_80_v, 1408, "", "./tensorflow/core/framework/op_kernel.h", "allocate_temp");

    return allocate_temp(type, shape, out_temp, AllocatorAttributes());
  }

  // Copies a tensor (allocated by the caller) to the specified output
  // index.  REQUIRES: !IsRefType(expected_output_dtype(index))
  // REQUIRES: 'tensor' must have the same MemoryType as
  // output_memory_types[index]. See comment above.
  Status set_output(StringPiece name, const Tensor& tensor);
  Status set_output(StringPiece name, Tensor&& tensor);
  void set_output(int index, const Tensor& tensor);
  void set_output(int index, Tensor&& tensor);

  // To output a reference.  Caller retains ownership of mu and tensor_for_ref,
  // and they must outlive all uses within the step. See comment above.
  // REQUIRES: IsRefType(expected_output_dtype(index))
  Status set_output_ref(StringPiece name, mutex* mu, Tensor* tensor_for_ref);

  // Returns nullptr if allocate_output() or set_output() have not been called.
  Status mutable_output(StringPiece name, Tensor** tensor);

  // Return the DeviceContext that should be used for this Op.
  //
  // If using the templated function, the type must be a subclass
  // of DeviceContext.
  //
  // Returns nullptr if the device did not provide one.
  template <typename T>
  T* op_device_context();
  DeviceContext* op_device_context() {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_81(mht_81_v, 1440, "", "./tensorflow/core/framework/op_kernel.h", "op_device_context");

    DeviceContext* ret = params_->op_device_context;
    if (ret == nullptr) {
      auto* dev_info = device()->tensorflow_accelerator_device_info();
      if (dev_info) ret = dev_info->default_context;
    }
    return ret;
  }

  AllocatorAttributes input_alloc_attr(int index) const {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_82(mht_82_v, 1452, "", "./tensorflow/core/framework/op_kernel.h", "input_alloc_attr");

    if (params_->input_alloc_attrs == nullptr) {
      return AllocatorAttributes();
    } else {
      DCHECK_GE(index, 0);
      DCHECK_LT(index, params_->input_alloc_attrs->size());
      return (*params_->input_alloc_attrs)[index];
    }
  }

  AllocatorAttributes output_alloc_attr(int index) const {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_83(mht_83_v, 1465, "", "./tensorflow/core/framework/op_kernel.h", "output_alloc_attr");

    return params_->output_attr_array[index];
  }

  gtl::InlinedVector<WrappedAllocator, 4> ConsumeWrappedAllocators() {
    gtl::InlinedVector<WrappedAllocator, 4> retrieved;
    if (tracking_state_) {
      mutex_lock lock(tracking_state_->mu);
      retrieved.swap(tracking_state_->wrapped_allocators);
    }
    return retrieved;
  }

  // Communication.
  //
  // An op kernel communicates with outside environment through
  // Rendezvous Send() and Recv().
  RendezvousInterface* rendezvous() const {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_84(mht_84_v, 1485, "", "./tensorflow/core/framework/op_kernel.h", "rendezvous");
 return params_->rendezvous; }

  CollectiveExecutor* collective_executor() const {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_85(mht_85_v, 1490, "", "./tensorflow/core/framework/op_kernel.h", "collective_executor");

    return params_->collective_executor;
  }

  // An op kernel can access the session state it belongs to.
  SessionState* session_state() const {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_86(mht_86_v, 1498, "", "./tensorflow/core/framework/op_kernel.h", "session_state");
 return params_->session_state; }

  // Unique identifier of the session it belongs to. Can be empty.
  std::string session_handle() const {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_87(mht_87_v, 1504, "", "./tensorflow/core/framework/op_kernel.h", "session_handle");
 return params_->session_handle; }

  // Metadata about the session. Can be nullptr.
  const SessionMetadata* session_metadata() const {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_88(mht_88_v, 1510, "", "./tensorflow/core/framework/op_kernel.h", "session_metadata");

    return params_->session_metadata;
  }

  // An op kernel can access the tensor store of the run it belongs to.
  TensorStore* tensor_store() const {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_89(mht_89_v, 1518, "", "./tensorflow/core/framework/op_kernel.h", "tensor_store");
 return params_->tensor_store; }

  // Function call support.
  //
  // If this kernel invocation is within a function execution,
  // call_frame() returns the call frame for the function call.
  CallFrameInterface* call_frame() const {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_90(mht_90_v, 1527, "", "./tensorflow/core/framework/op_kernel.h", "call_frame");
 return params_->call_frame; }

  // If not nullptr, the kernel invoke functions defined in the
  // library. E.g., CHECK_NOTNULL(function_library())->Run("Foo", ...).
  FunctionLibraryRuntime* function_library() const {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_91(mht_91_v, 1534, "", "./tensorflow/core/framework/op_kernel.h", "function_library");

    return params_->function_library;
  }

  std::function<void(std::function<void()>)>* runner() const {
    return params_->runner;
  }
  StepStatsCollectorInterface* stats_collector() const {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_92(mht_92_v, 1544, "", "./tensorflow/core/framework/op_kernel.h", "stats_collector");

    return params_->stats_collector;
  }

  // Shared resources accessible to this kernel.
  ResourceMgr* resource_manager() const {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_93(mht_93_v, 1552, "", "./tensorflow/core/framework/op_kernel.h", "resource_manager");
 return params_->resource_manager; }

  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache() const {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_94(mht_94_v, 1557, "", "./tensorflow/core/framework/op_kernel.h", "slice_reader_cache");

    return params_->slice_reader_cache;
  }

  // Execution.
  //
  // OpKernels can use these eigen devices to carry out their
  // numerical computation.
  const Eigen::ThreadPoolDevice& eigen_cpu_device() const {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_95(mht_95_v, 1568, "", "./tensorflow/core/framework/op_kernel.h", "eigen_cpu_device");

    return *device()->eigen_cpu_device();
  }
  const Eigen::GpuDevice& eigen_gpu_device() const {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_96(mht_96_v, 1574, "", "./tensorflow/core/framework/op_kernel.h", "eigen_gpu_device");

    return params_->eigen_gpu_device->device();
  }
  template <typename EigenDeviceType>
  const EigenDeviceType& eigen_device() const;

  // Error handling.

  // If expected_inputs == inputs() and expected_outputs == output_types(),
  // returns OK, else returns INVALID_ARGUMENT with an error message.
  // Recommended for Ops with dynamic signatures, where validation can only
  // be performed at runtime.
  Status MatchSignature(const DataTypeSlice expected_inputs,
                        const DataTypeSlice expected_outputs);

  // An OpKernel should call SetStatus() if Compute() encounters an
  // error.
  void SetStatus(const Status& status);
  const Status& status() const {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_97(mht_97_v, 1595, "", "./tensorflow/core/framework/op_kernel.h", "status");
 return status_; }

  // Cancellation.
  //
  // EXPERIMENTAL. See the implementation in tensorflow::FIFOQueue for an
  // example of how to use this API.
  CancellationManager* cancellation_manager() const {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_98(mht_98_v, 1604, "", "./tensorflow/core/framework/op_kernel.h", "cancellation_manager");

    return params_->cancellation_manager;
  }

  // Other accessors.

  // For control flow.
  FrameAndIter frame_iter() const {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_99(mht_99_v, 1614, "", "./tensorflow/core/framework/op_kernel.h", "frame_iter");
 return params_->frame_iter; }
  bool is_input_dead() const {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_100(mht_100_v, 1618, "", "./tensorflow/core/framework/op_kernel.h", "is_input_dead");
 return params_->is_input_dead; }

  // May be used, e.g., to get GPU handles, etc.
  // TODO(tucker): Add example usage.
  DeviceBase* device() const {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_101(mht_101_v, 1625, "", "./tensorflow/core/framework/op_kernel.h", "device");
 return params_->device; }

  // Per-step container for use by white-listed internal ops.
  ScopedStepContainer* step_container() const {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_102(mht_102_v, 1631, "", "./tensorflow/core/framework/op_kernel.h", "step_container");

    return params_->step_container;
  }

  // Access to distributed coordination service.
  CoordinationServiceAgent* coordination_service_agent() const {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_103(mht_103_v, 1639, "", "./tensorflow/core/framework/op_kernel.h", "coordination_service_agent");

    return params_->coordination_service_agent;
  }

  // Helper routines for the OP_REQUIRES macros
  void CtxFailure(const Status& s);
  void CtxFailureWithWarning(const Status& s);
  void CtxFailure(const char* file, int line, const Status& s);
  void CtxFailureWithWarning(const char* file, int line, const Status& s);

  // Unrecommended functions: these are functions that have some
  // current uses but are not recommended for use, and may go away at
  // some future major version release.
  //
  // The following functions all have versions that return Status
  // to capture error conditions, and are strongly preferred.
  Tensor* mutable_output(int index);
  mutex* input_ref_mutex(int index);
  void set_output_ref(int index, mutex* mu, Tensor* tensor_for_ref);
  TensorValue release_output(int index);

  bool track_allocations() const {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_104(mht_104_v, 1663, "", "./tensorflow/core/framework/op_kernel.h", "track_allocations");
 return params_->track_allocations; }

  // Records temp memory allocation. Tensor object is recorded to identify the
  // case where temp memory is used as output memory.
  void record_temp_memory_allocation(int64_t size, const Tensor& t)
      TF_LOCKS_EXCLUDED(tracking_state_->stats_mu);

  // Returns recorded size of temporary memory;
  int64_t temp_memory_allocated() const
      TF_LOCKS_EXCLUDED(tracking_state_->stats_mu);

  // Records persistent memory allocation, size can be negative indicating
  // deallocation.
  void record_persistent_memory_allocation(int64_t size, int64_t alloc_id = -1)
      TF_LOCKS_EXCLUDED(tracking_state_->stats_mu);

  // Returns recorded size and ids of persistent memory.
  int64_t persistent_memory_allocated() const
      TF_LOCKS_EXCLUDED(tracking_state_->stats_mu);

  std::vector<int64_t> persistent_alloc_ids() const
      TF_LOCKS_EXCLUDED(tracking_state_->stats_mu);

  // Resets counters for temp and persistent memory and recorded ids.
  void clear_recorded_memory() TF_LOCKS_EXCLUDED(tracking_state_->stats_mu);

  bool input_is_ref(int index) const;

  void set_record_memory_consumption(bool v);

  // Used by OpKernel implementations to track actively running deferred ops.
  //
  // A deferred op is one whose Compute method returns (or whose ComputeAsync
  // method invokes the callback) when work is scheduled onto a device. At that
  // point, we don't know when the work will actually complete (or if it has
  // already completed) on the device. These functions allow the executor to
  // track the status of deferred ops and act accordingly.
  //
  // Deferred OpKernel implementations must use these methods to get two
  // functions. It then must call these two functions in pairs, before and after
  // device execution, respectively.
  TF_MUST_USE_RESULT std::function<void()> inc_num_deferred_ops_function() {
    DCHECK(params_->op_kernel->is_deferred());
    return params_->inc_num_deferred_ops_function
               ? params_->inc_num_deferred_ops_function
               : []() {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_105(mht_105_v, 1711, "", "./tensorflow/core/framework/op_kernel.h", "lambda");
};
  }
  TF_MUST_USE_RESULT std::function<void()> dec_num_deferred_ops_function() {
    DCHECK(params_->op_kernel->is_deferred());
    return params_->dec_num_deferred_ops_function
               ? params_->dec_num_deferred_ops_function
               : []() {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_106(mht_106_v, 1720, "", "./tensorflow/core/framework/op_kernel.h", "lambda");
};
  }

  Allocator* get_allocator(AllocatorAttributes attr);

 private:
  bool record_memory_consumption_ = false;

  // Internal common method used when allocating tensor memory
  Status allocate_tensor(DataType type, const TensorShape& shape,
                         Tensor* out_tensor,
                         AllocatorAttributes allocator_attr) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_107(mht_107_v, 1734, "", "./tensorflow/core/framework/op_kernel.h", "allocate_tensor");

    return allocate_tensor(type, shape, out_tensor, allocator_attr,
                           AllocationAttributes());
  }

  Status allocate_tensor(DataType type, const TensorShape& shape,
                         Tensor* out_tensor, AllocatorAttributes allocator_attr,
                         const AllocationAttributes& allocation_attr);

  // Helpers for `set_output()`.

  // Returns `true` if the tensor was copied into an allocated output.
  bool maybe_set_output_by_allocate_and_copy(int index, const Tensor& tensor);

  void maybe_track_allocations_for_set_output(const Tensor& tensor);

  Status get_input_index(StringPiece name, int* out_index) const;
  Status get_output_index(StringPiece name, int* out_index) const;

  // Initialize the allocated_scope_ids_ set the first time this method is
  // called.
  void maybe_initialize_scope_id_set();

  Status status_;
  friend class CollectiveExecutor;  // for access to params_
  Params* params_;                  // not owned
  gtl::InlinedVector<TensorValue, 4> outputs_;

  // Keep track of calls to ScopedAllocator.
  // TODO(ayushd): change to absl::flat_hash_set.
  std::unique_ptr<std::unordered_set<int32>> allocated_scope_ids_;

  // The following data members are only used when allocation tracking is
  // enabled, memory consumption is being recorded, or tensor access is being
  // recorded.
  struct TrackingState {
    mutable mutex mu;
    gtl::InlinedVector<WrappedAllocator, 4> wrapped_allocators
        TF_GUARDED_BY(mu);

    mutable mutex stats_mu;
    int64_t temp_memory_allocated TF_GUARDED_BY(stats_mu) = 0;

    int64_t persistent_memory_allocated TF_GUARDED_BY(stats_mu) = 0;
    gtl::InlinedVector<std::pair<const void*, int64_t>, 2>
        temp_tensor_buffer_and_size TF_GUARDED_BY(stats_mu);
    gtl::InlinedVector<int64_t, 2> persistent_alloc_ids TF_GUARDED_BY(stats_mu);
  };
  std::unique_ptr<TrackingState> tracking_state_;

  // For access to `params_->op_kernel`.
  friend void CheckNotInComputeAsync(OpKernelContext* ctx,
                                     const char* correct_macro_name);

  TF_DISALLOW_COPY_AND_ASSIGN(OpKernelContext);
};

template <>
const Eigen::ThreadPoolDevice& OpKernelContext::eigen_device() const;

template <>
const Eigen::GpuDevice& OpKernelContext::eigen_device() const;

// Register your OpKernel by specifying the Op's name, the device the
// kernel runs on, any type attr constraints for this kernel, any
// host-memory args, and the class to instantiate.  Examples:
//
//  // A kernel that supports all types.
//  REGISTER_KERNEL_BUILDER(Name("Save").Device(DEVICE_CPU), SaveOp);
//
//  // The following are equivalent ways of specifying that the kernel only
//  // works if the "T" type attr is set to DT_FLOAT.
//  REGISTER_KERNEL_BUILDER(
//      Name("Sub").Device(DEVICE_CPU).TypeConstraint<float>("T"),
//      SubOp<float>);
//  // (You would then repeat this for every type supported by "Sub".)
//
//  // This form allows you to specify a list of types as the constraint.
//  REGISTER_KERNEL_BUILDER(Name("Sub")
//                              .Device(DEVICE_CPU)
//                              .TypeConstraint("T", {DT_FLOAT}),
//                          SubOp<float>);
//
//  // A kernel that expects one of the input tensors in host memory.
//  REGISTER_KERNEL_BUILDER(
//      Name("Reshape").Device(DEVICE_GPU).HostMemory("shape"), ReshapeOp);
//
// See kernel_def_builder for details.

// Instantiate an OpKernel that has been registered.  Returns nullptr
// if no operation for that type of device / input signature combination
// (and a NOT_FOUND *status), or there is an error in construction (and
// an INVALID_ARGUMENT *status).  Otherwise, the caller takes ownership
// of the returned pointer.
// EXPECTED USAGE: unique_ptr<OpKernel> op = CreateOpKernel(...);
// REQUIRES: def has all attrs specified (e.g. using AddDefaultsToNodeDef()).
std::unique_ptr<OpKernel> CreateOpKernel(DeviceType device_type,
                                         DeviceBase* device,
                                         Allocator* allocator,
                                         const NodeDef& node_def,
                                         int graph_def_version, Status* status);

std::unique_ptr<OpKernel> CreateOpKernel(
    DeviceType device_type, DeviceBase* device, Allocator* allocator,
    const std::shared_ptr<const NodeProperties>& props, int graph_def_version,
    Status* status);

Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
                      Allocator* allocator, FunctionLibraryRuntime* flib,
                      const std::shared_ptr<const NodeProperties>& props,
                      int graph_def_version, OpKernel** kernel);

Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
                      Allocator* allocator, FunctionLibraryRuntime* flib,
                      ResourceMgr* resource_mgr,
                      const std::shared_ptr<const NodeProperties>& props,
                      int graph_def_version, OpKernel** kernel);

// Returns into 'device_types' the subset of prioritized_types that this
// binary has registered for the given NodeDef.
//
// REQUIRES: * 'device_types' is not nullptr.
//           * def has all attrs specified (e.g. using AddDefaultsToNodeDef()).
Status SupportedDeviceTypesForNode(
    const std::vector<DeviceType>& prioritized_types, const NodeDef& def,
    PrioritizedDeviceTypeVector* device_types,
    const DeviceNameUtils::ParsedName* local_address_spec = nullptr);

// Returns a message with a description of the kernels registered for op
// `op_name`.
std::string KernelsRegisteredForOp(StringPiece op_name);

// Call once after Op registration has completed.
Status ValidateKernelRegistrations(const OpRegistryInterface& op_registry);

// -----------------------------------------------------------------------------
// OpKernel registration implementation follows, please ignore.

// Allow the REGISTER_KERNEL_BUILDER(Name("op_name").Device(...)...) syntax.
namespace register_kernel {

class Name : public KernelDefBuilder {
 public:
  explicit Name(const char* op) : KernelDefBuilder(op) {
   std::vector<std::string> mht_108_v;
   mht_108_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_108(mht_108_v, 1881, "", "./tensorflow/core/framework/op_kernel.h", "Name");
}
};

}  // namespace register_kernel

// Kernel registration appears as:
//   REGISTER_KERNEL_BUILDER(Name("OpName").Device(DEVICE_CPU)..., OpImpl)
// We'd like to have "OpName" as a constant-expression, without requiring that
// of the overall KernelDefBuilder expression (beginning with the
// register_kernel::Name constructor above).
//
// So, we pull the "OpName" part to a separate macro-level argument. This
// involves treating Name("OpName") as a macro call, via token-pasting (e.g.
// M_## =>  M_Name("OpName")), and having it expand to '"OpName",
// Name("OpName")' which is then usable as two arguments.
#define TF_EXTRACT_KERNEL_NAME_Name(name_str) \
  name_str, ::tensorflow::register_kernel::Name(name_str)
#define TF_EXTRACT_KERNEL_NAME_IMPL(m, ...) m(__VA_ARGS__)
#define TF_EXTRACT_KERNEL_NAME(m, kernel_builder, ...)                    \
  TF_EXTRACT_KERNEL_NAME_IMPL(m, TF_EXTRACT_KERNEL_NAME_##kernel_builder, \
                              __VA_ARGS__)

// REGISTER_KERNEL_BUILDER_IMPL_2, with a unique 'ctr' as the first argument.
// TODO(dodgen): There are some uses of this macro inside functions, where
// kernel_builder refers to (non-const) locals (they should be fixed). To
// accommodate those, kernel_builder.Build() appears as an argument to an
// immediately-called lambda (not in the lambda itself).
#define REGISTER_KERNEL_BUILDER_IMPL_3(ctr, op_name, kernel_builder_expr,   \
                                       is_system_kernel, ...)               \
  static ::tensorflow::InitOnStartupMarker const register_kernel_##ctr      \
      TF_ATTRIBUTE_UNUSED =                                                 \
          TF_INIT_ON_STARTUP_IF(is_system_kernel ||                         \
                                (SHOULD_REGISTER_OP_KERNEL(#__VA_ARGS__) && \
                                 SHOULD_REGISTER_OP(op_name)))              \
          << ([](::tensorflow::KernelDef const* kernel_def) {               \
               ::tensorflow::kernel_factory::OpKernelRegistrar registrar(   \
                   kernel_def, #__VA_ARGS__,                                \
                   [](::tensorflow::OpKernelConstruction* context)          \
                       -> ::tensorflow::OpKernel* {                         \
                     return new __VA_ARGS__(context);                       \
                   });                                                      \
               (void)registrar;                                             \
               return ::tensorflow::InitOnStartupMarker{};                  \
             })(kernel_builder_expr.Build());

// REGISTER_KERNEL_BUILDER_IMPL, but with kernel_builder split to op_name,
// kernel_builder_expr.
#define REGISTER_KERNEL_BUILDER_IMPL_2(op_name, kernel_builder_expr, \
                                       is_system_kernel, ...)        \
  TF_NEW_ID_FOR_INIT(REGISTER_KERNEL_BUILDER_IMPL_3, op_name,        \
                     kernel_builder_expr, is_system_kernel, __VA_ARGS__)

// REGISTER_KERNEL_BUILDER, but with is_system_kernel bound.
#define REGISTER_KERNEL_BUILDER_IMPL(kernel_builder, is_system_kernel, ...) \
  TF_EXTRACT_KERNEL_NAME(REGISTER_KERNEL_BUILDER_IMPL_2, kernel_builder,    \
                         is_system_kernel, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER(kernel_builder, ...) \
  TF_ATTRIBUTE_ANNOTATE("tf:kernel")                 \
  REGISTER_KERNEL_BUILDER_IMPL(kernel_builder, false, __VA_ARGS__)

// The `REGISTER_SYSTEM_KERNEL_BUILDER()` macro acts as
// `REGISTER_KERNEL_BUILDER()` except that the kernel is registered
// unconditionally even when selective registration is used.
#define REGISTER_SYSTEM_KERNEL_BUILDER(kernel_builder, ...) \
  TF_ATTRIBUTE_ANNOTATE("tf:kernel")                        \
  TF_ATTRIBUTE_ANNOTATE("tf:kernel:system")                 \
  REGISTER_KERNEL_BUILDER_IMPL(kernel_builder, true, __VA_ARGS__)

// Checks whether a given kernel is registered on device_type.
bool KernelDefAvailable(const DeviceType& device_type, const NodeDef& node_def);

// If node of node_name, experimental_debug_info, node_op, node_device and
// node_attrs has a corresponding kernel registered on device_type, returns OK
// and fill in the kernel def and kernel_class_name. <def> and
// <kernel_class_name> may be null.
Status FindKernelDef(
    const DeviceType& device_type, StringPiece node_name,
    bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info,
    StringPiece node_op, StringPiece node_device, AttrSlice node_attrs,
    const KernelDef** def, std::string* kernel_class_name);

// If node_def has a corresponding kernel registered on device_type,
// returns OK and fill in the kernel def and kernel_class_name. <def> and
// <kernel_class_name> may be null.
Status FindKernelDef(const DeviceType& device_type, const NodeDef& node_def,
                     const KernelDef** def, std::string* kernel_class_name);

// Writes a list of all registered kernels to LOG(INFO), to help users debug
// missing kernel errors.
void LogAllRegisteredKernels();

// Gets a list of all registered kernels.
KernelList GetAllRegisteredKernels();

// Gets a list of all registered kernels for which predicate returns true
KernelList GetFilteredRegisteredKernels(
    const std::function<bool(const KernelDef&)>& predicate);

// Gets a list of all registered kernels for a given op
KernelList GetRegisteredKernelsForOp(StringPiece op_name);

namespace kernel_factory {

// OpKernelFactory is responsible for creating OpKernels when TensorFlow needs
// them. You register factories with the TensorFlow core by constructing an
// OpKernelRegistrar and passing the factory as a constructor parameter.
class OpKernelFactory {
 public:
  virtual OpKernel* Create(OpKernelConstruction* context) = 0;
  virtual ~OpKernelFactory() = default;
};

class OpKernelRegistrar {
 public:
  // Registers the given kernel factory with TensorFlow. TF will call the
  // factory Create() method when it determines that a kernel matching the given
  // KernelDef is required.
  OpKernelRegistrar(const KernelDef* kernel_def, StringPiece kernel_class_name,
                    std::unique_ptr<OpKernelFactory> factory) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_109(mht_109_v, 2004, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelRegistrar");

    InitInternal(kernel_def, kernel_class_name, std::move(factory));
  }

  // Registers the given factory function with TensorFlow. This is equivalent
  // to registering a factory whose Create function invokes `create_fn`.
  OpKernelRegistrar(const KernelDef* kernel_def, StringPiece kernel_class_name,
                    OpKernel* (*create_fn)(OpKernelConstruction*)) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_110(mht_110_v, 2014, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelRegistrar");

    InitInternal(kernel_def, kernel_class_name,
                 absl::make_unique<PtrOpKernelFactory>(create_fn));
  }

 private:
  struct PtrOpKernelFactory : public OpKernelFactory {
    explicit PtrOpKernelFactory(OpKernel* (*create_func)(OpKernelConstruction*))
        : create_func_(create_func) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_111(mht_111_v, 2025, "", "./tensorflow/core/framework/op_kernel.h", "PtrOpKernelFactory");
}

    OpKernel* Create(OpKernelConstruction* context) override;

    OpKernel* (*create_func_)(OpKernelConstruction*);
  };

  void InitInternal(const KernelDef* kernel_def, StringPiece kernel_class_name,
                    std::unique_ptr<OpKernelFactory> factory);
};

}  // namespace kernel_factory

// -----------------------------------------------------------------------------
// Template and inline method implementations, please ignore

template <class T>
Status OpKernelConstruction::GetAttr(StringPiece attr_name, T* value) const {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_112(mht_112_v, 2045, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelConstruction::GetAttr");

  return GetNodeAttr(def(), attr_name, value);
}

inline DataType OpKernelContext::input_dtype(int index) const {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_113(mht_113_v, 2052, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::input_dtype");

  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs());
  const TensorValue& value((*params_->inputs)[index]);
  return value.dtype();
}

inline MemoryType OpKernelContext::input_memory_type(int index) const {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_114(mht_114_v, 2062, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::input_memory_type");

  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs());
  return op_kernel().input_memory_types()[index];
}

inline DataType OpKernelContext::expected_output_dtype(int index) const {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_115(mht_115_v, 2071, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::expected_output_dtype");

  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_outputs());
  return params_->op_kernel->output_type(index);
}

inline MemoryType OpKernelContext::output_memory_type(int index) const {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_116(mht_116_v, 2080, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::output_memory_type");

  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_outputs());
  return op_kernel().output_memory_types()[index];
}

inline bool OpKernelContext::input_is_ref(int index) const {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_117(mht_117_v, 2089, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::input_is_ref");

  const TensorValue& value((*params_->inputs)[index]);
  return value.is_ref();
}

// no input if tensor == nullptr.
inline bool OpKernelContext::has_input(int index) const {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_118(mht_118_v, 2098, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::has_input");

  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs());
  return (*params_->inputs)[index].tensor != nullptr;
}

inline mutex* OpKernelContext::input_ref_mutex(int index) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_119(mht_119_v, 2107, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::input_ref_mutex");

  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs());
  DCHECK(input_is_ref(index));
  return (*params_->inputs)[index].mutex_if_ref;
}

inline Tensor* OpKernelContext::mutable_output(int index) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_120(mht_120_v, 2117, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::mutable_output");

  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_outputs());
  return outputs_[index].tensor;
}

inline TensorValue OpKernelContext::release_output(int index) {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_121(mht_121_v, 2126, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::release_output");

  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_outputs());
  TensorValue value = outputs_[index];
  outputs_[index] = TensorValue();
  return value;
}

inline Status OpKernelContext::forward_input_or_allocate_output(
    gtl::ArraySlice<int> candidate_input_indices, int output_index,
    const TensorShape& output_shape, Tensor** output, int* forwarded_input) {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_122(mht_122_v, 2139, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::forward_input_or_allocate_output");

  for (int input_index : candidate_input_indices) {
    if (forward_input_to_output_with_shape(input_index, output_index,
                                           output_shape, output)) {
      if (forwarded_input != nullptr) {
        *forwarded_input = input_index;
      }
      return Status::OK();
    }
  }
  if (forwarded_input != nullptr) {
    *forwarded_input = -1;
  }
  return allocate_output(output_index, output_shape, output);
}

inline Status OpKernelContext::forward_input_or_allocate_output(
    gtl::ArraySlice<StringPiece> candidate_input_names, StringPiece output_name,
    const TensorShape& output_shape, Tensor** output) {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_123(mht_123_v, 2160, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::forward_input_or_allocate_output");

  for (const StringPiece& input_name : candidate_input_names) {
    if (forward_input_to_output_with_shape(input_name, output_name,
                                           output_shape, output)
            .ok()) {
      return Status::OK();
    }
  }
  return allocate_output(output_name, output_shape, output);
}

template <typename T>
T* OpKernelContext::op_device_context() {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_124(mht_124_v, 2175, "", "./tensorflow/core/framework/op_kernel.h", "OpKernelContext::op_device_context");

  static_assert(std::is_base_of<DeviceContext, T>::value,
                "T is not a subclass of DeviceContext");
  return static_cast<T*>(op_device_context());
}

inline const Tensor& OpInputList::operator[](int i) const {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_125(mht_125_v, 2184, "", "./tensorflow/core/framework/op_kernel.h", "lambda");

  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->input(start_ + i);
}

inline mutex* OpMutableInputList::ref_mutex(int i) {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_126(mht_126_v, 2193, "", "./tensorflow/core/framework/op_kernel.h", "OpMutableInputList::ref_mutex");

  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->input_ref_mutex(start_ + i);
}

inline Tensor OpMutableInputList::at(int i, bool lock_held) {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_127(mht_127_v, 2202, "", "./tensorflow/core/framework/op_kernel.h", "OpMutableInputList::at");

  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->mutable_input(start_ + i, lock_held);
}

inline Tensor* OpOutputList::operator[](int i) {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_128(mht_128_v, 2211, "", "./tensorflow/core/framework/op_kernel.h", "lambda");

  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->mutable_output(start_ + i);
}

inline bool OpOutputList::required(int i) const {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_129(mht_129_v, 2220, "", "./tensorflow/core/framework/op_kernel.h", "OpOutputList::required");

  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->output_required(start_ + i);
}

inline DataType OpOutputList::expected_output_dtype(int i) const {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_130(mht_130_v, 2229, "", "./tensorflow/core/framework/op_kernel.h", "OpOutputList::expected_output_dtype");

  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->expected_output_dtype(start_ + i);
}

inline Status OpOutputList::allocate(int i, const TensorShape& shape,
                                     Tensor** output) {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_131(mht_131_v, 2239, "", "./tensorflow/core/framework/op_kernel.h", "OpOutputList::allocate");

  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  return ctx_->allocate_output(start_ + i, shape, output);
}

inline void OpOutputList::set(int i, const Tensor& tensor) {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_132(mht_132_v, 2248, "", "./tensorflow/core/framework/op_kernel.h", "OpOutputList::set");

  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  ctx_->set_output(start_ + i, tensor);
}

inline void OpOutputList::set(int i, Tensor&& tensor) {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_133(mht_133_v, 2257, "", "./tensorflow/core/framework/op_kernel.h", "OpOutputList::set");

  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  ctx_->set_output(start_ + i, std::move(tensor));
}

inline void OpOutputList::set_ref(int i, mutex* mu, Tensor* tensor_for_ref) {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_134(mht_134_v, 2266, "", "./tensorflow/core/framework/op_kernel.h", "OpOutputList::set_ref");

  DCHECK_GE(i, 0);
  DCHECK_LT(i, stop_ - start_);
  ctx_->set_output_ref(i, mu, tensor_for_ref);
}

// Generate a fatal error if OP_REQUIRES or OP_REQUIRES_OK are used in
// AsyncOpKernel implementations. If these macros are used and the condition
// does not hold, the `done` callback will never be called and the system will
// deadlock, so a crash failure is preferable. Since the OP_REQUIRES[_OK] macros
// are legal to use in AsyncOpKernel constructors, we use overload resolution
// to distinguish between OpKernelConstruction* and OpKernelContext* context
// types.
class XlaOpKernelContext;
inline void CheckNotInComputeAsync(XlaOpKernelContext*, const char*) {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_135(mht_135_v, 2283, "", "./tensorflow/core/framework/op_kernel.h", "CheckNotInComputeAsync");
}
inline void CheckNotInComputeAsync(OpKernelConstruction*, const char*) {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTh mht_136(mht_136_v, 2287, "", "./tensorflow/core/framework/op_kernel.h", "CheckNotInComputeAsync");
}
void CheckNotInComputeAsync(OpKernelContext* ctx,
                            const char* correct_macro_name);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OP_KERNEL_H_
