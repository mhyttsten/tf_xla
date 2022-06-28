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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc() {
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

#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"

#include <memory>

#include "absl/strings/match.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/profiler/lib/annotated_traceme.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#endif  // !IS_MOBILE_PLATFORM

namespace tensorflow {

Status EagerKernelArgs::GetLocalArg(const FunctionArgIndex& index,
                                    Tensor* val) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_0(mht_0_v, 224, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "EagerKernelArgs::GetLocalArg");

  if (index.sub_index >= 0) {
    return errors::InvalidArgument("Got unexpected sub_index ", index.sub_index,
                                   " for argument ", index.index);
  }
  Tensor* arg = tensor_args_.at(index.index).tensor;
  if (arg) {
    *val = *arg;
    return Status::OK();
  } else {
    return errors::NotFound("Argument ", index.index, " has no local tensor.");
  }
}

std::vector<Tensor> EagerKernelArgs::GetLocalTensors() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "EagerKernelArgs::GetLocalTensors");

  std::vector<Tensor> local_inputs;
  local_inputs.reserve(tensor_args_.size());
  for (const TensorValue& tensor_value : tensor_args_) {
    local_inputs.push_back(*tensor_value.tensor);
  }
  return local_inputs;
}

std::function<void(std::function<void()>)>* KernelAndDevice::get_runner()
    const {
  if (runner_) {
    return runner_;
  } else {
    static auto* default_runner =
        new std::function<void(std::function<void()>)>(
            [](const std::function<void()>& f) { f(); });
    return default_runner;
  }
}

KernelAndDeviceFunc::~KernelAndDeviceFunc() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_2(mht_2_v, 265, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceFunc::~KernelAndDeviceFunc");

  if (handle_ != kInvalidHandle) {
    Status status = pflr_->ReleaseHandle(handle_);
    if (!status.ok()) {
      LOG(INFO) << "Ignoring error status when releasing multi-device function "
                   "handle "
                << status.ToString();
    }
  }
}

Status KernelAndDeviceOp::Init(const bool log_device_placement,
                               const NodeDef& ndef,
                               GraphCollector* graph_collector) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_3(mht_3_v, 281, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceOp::Init");

  OpKernel* k = nullptr;
  if (flr_ == nullptr) {
    return errors::Internal(
        "A valid FunctionLibraryRuntime must be provided when running ops "
        "based on OpKernel.");
  }
  std::shared_ptr<const NodeProperties> props;
  TF_RETURN_IF_ERROR(NodeProperties::CreateFromNodeDef(
      ndef, flr_->GetFunctionLibraryDefinition(), &props));
  TF_RETURN_IF_ERROR(flr_->CreateKernel(props, &k));
  kernel_.reset(k);
  const auto* op_reg_data = OpRegistry::Global()->LookUp(ndef.op());
  if (op_reg_data != nullptr) {
    is_distributed_communication_op_ =
        op_reg_data->op_def.is_distributed_communication();
  }

  input_alloc_attrs_.resize(kernel_->num_inputs());
  input_devices_.resize(kernel_->num_inputs(), device_);
  for (size_t i = 0; i < input_alloc_attrs_.size(); ++i) {
    bool host = kernel_->input_memory_types()[i] == tensorflow::HOST_MEMORY;
    input_alloc_attrs_[i].set_on_host(host);
    if (host && input_devices_[i]->device_type() != DEVICE_CPU) {
      input_devices_[i] = host_cpu_device_;
    }
  }
  output_alloc_attrs_.resize(kernel_->num_outputs());
  for (size_t i = 0; i < output_alloc_attrs_.size(); ++i) {
    output_alloc_attrs_[i].set_on_host(kernel_->output_memory_types()[i] ==
                                       tensorflow::HOST_MEMORY);
  }

  return Status::OK();
}

Status KernelAndDeviceFunc::InstantiateFunc(const bool log_device_placement,
                                            const NodeDef& ndef,
                                            GraphCollector* graph_collector) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_4(mht_4_v, 322, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceFunc::InstantiateFunc");

  const OpDef* op_def = nullptr;
  const FunctionDef* function_def;
  if (flr_ == nullptr) {
    // If function is being executed without an explicit device request,
    // lookup the FunctionDef in the CPU's FLR. All FLRs share the same
    // library.
    function_def = pflr_->GetFLR(host_cpu_device_->name())
                       ->GetFunctionLibraryDefinition()
                       ->Find(ndef.op());
  } else {
    function_def = flr_->GetFunctionLibraryDefinition()->Find(ndef.op());
  }

  if (function_def != nullptr) {
    op_def = &(function_def->signature());
  } else {
    TF_RETURN_IF_ERROR(OpDefForOp(ndef.op(), &op_def));
  }
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(ndef, *op_def, &input_dtypes_, &output_dtypes_));

  FunctionLibraryRuntime::InstantiateOptions options;
  options.target = device_ == nullptr ? "" : device_->name();
  options.is_multi_device_function = true;
  for (const Device* device : input_devices_) {
    options.input_devices.push_back(device->name());
  }
  options.composite_devices = composite_devices_;
  options.input_resource_dtypes_and_shapes = input_resource_dtypes_and_shapes_;
  if (outputs_on_op_device_) {
    const FunctionLibraryDefinition* lib_def =
        pflr_->GetFunctionLibraryDefinition();
    const FunctionDef* fdef = lib_def->Find(ndef.op());
    if (fdef == nullptr) {
      return errors::InvalidArgument("Failed to find function ", ndef.op());
    }
    for (int i = 0; i < fdef->signature().output_arg_size(); ++i) {
      options.output_devices.push_back(options.target);
    }
  }

  const auto& it = ndef.attr().find("executor_type");
  if (it != ndef.attr().end()) {
    options.executor_type = it->second.s();
  }
  const auto& is_component_fn_it = ndef.attr().find("is_component_function");
  if (is_component_fn_it != ndef.attr().end()) {
    options.is_component_function = is_component_fn_it->second.b();
  }
#if !defined(IS_MOBILE_PLATFORM)
  // Android tf library does not include grappler.
  const auto& config_it = ndef.attr().find("config_proto");
  if (config_it != ndef.attr().end()) {
    if (!options.config_proto.ParseFromString(config_it->second.s())) {
      return errors::InvalidArgument(
          "Failed to parse config_proto attribute as tensorflow::ConfigProto "
          "proto.");
    }
    grappler::GrapplerItem::OptimizationOptions optimization_options =
        grappler::CreateOptOptionsForEager();

    options.optimize_graph_fn = std::bind(
        grappler::OptimizeGraph, std::placeholders::_1, std::placeholders::_2,
        std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
        options.config_proto, function_def->signature().name(),
        optimization_options, std::placeholders::_6);
  }
#endif  // !IS_MOBILE_PLATFORM
  options.graph_collector = graph_collector;

  options.allow_small_function_optimizations =
      allow_small_function_optimizations_;

  options.allow_control_flow_sync_execution =
      allow_control_flow_sync_execution_;

  options.shape_inference_on_tfe_dialect_import =
      shape_inference_on_tfe_dialect_import_;

  // In Eager mode we always inline all functions into the top-level
  // function body graph, to get a single executable graph, that could be
  // optimized across function boundaries (e.g. prune unused inputs and
  // outputs in a function call chain). This is required to mimic graph mode
  // execution, with aggressive pruning of nodes not in the transitive fanin
  // of fetches.
  options.config_proto.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);

  options.config_proto.set_log_device_placement(log_device_placement);

  options.int_args_and_retvals_on_device = int_args_and_retvals_on_device_;

  if (xla_compile_device_type_.has_value()) {
    options.xla_compile_device_type = xla_compile_device_type_.value();
  }

  TF_RETURN_IF_ERROR(
      pflr_->Instantiate(ndef.op(), AttrSlice(ndef), options, &handle_));
  return pflr_->IsCrossProcess(handle_, &is_cross_process_);
}

Status KernelAndDeviceFunc::Init(const bool log_device_placement,
                                 const NodeDef& ndef,
                                 GraphCollector* graph_collector) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_5(mht_5_v, 430, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceFunc::Init");

  TF_RETURN_IF_ERROR(
      InstantiateFunc(log_device_placement, ndef, graph_collector));
  return pflr_->GetOutputDevices(handle_, &output_devices_);
}

namespace {
// In certain contexts (e.g. TPU async executions), the CancellationManager is
// used to shut down the device in error scenarios (as opposed to using the
// AsyncCompute's DoneCallback). This is handled through the
// {inc,dec}_num_deferred_ops_function.
struct OpExecutionState : public core::RefCounted {
  // TODO(nareshmodi): consider refcounting the cancellation_manager.
  CancellationManager cancellation_manager;
};
}  // anonymous namespace

Status KernelAndDeviceOp::Run(
    ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
    std::vector<EagerKernelRet>* outputs,
    CancellationManager* cancellation_manager,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    const absl::optional<ManagedStackTrace>& stack_trace,
    CoordinationServiceAgent* coordination_service_agent) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_6(mht_6_v, 456, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceOp::Run");

  OpKernelContext::Params params;
  params.device = device_;
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = inputs.GetTensorValues();
  params.op_kernel = kernel_.get();
  params.resource_manager = device_->resource_manager();
  params.input_alloc_attrs = &input_alloc_attrs_;
  params.output_attr_array = output_alloc_attrs_.data();
  params.function_library = flr_;
  params.slice_reader_cache = &slice_reader_cache_;
  params.rendezvous = rendezvous_;
  params.stack_trace = stack_trace;
  OpExecutionState* op_execution_state = nullptr;

  CancellationManager default_cancellation_manager;
  if (cancellation_manager) {
    params.cancellation_manager = cancellation_manager;
  } else if (kernel_->is_deferred()) {
    op_execution_state = new OpExecutionState;
    params.cancellation_manager = &op_execution_state->cancellation_manager;
    params.inc_num_deferred_ops_function = [op_execution_state]() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_7(mht_7_v, 480, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "lambda");

      op_execution_state->Ref();
    };
    params.dec_num_deferred_ops_function = [op_execution_state]() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_8(mht_8_v, 486, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "lambda");

      op_execution_state->Unref();
    };
  } else {
    params.cancellation_manager = &default_cancellation_manager;
  }

  params.log_memory = log_memory_;

  params.runner = get_runner();

  params.step_container = step_container;

  params.collective_executor =
      collective_executor_ ? collective_executor_->get() : nullptr;

  params.coordination_service_agent = coordination_service_agent;

  OpKernelContext context(&params);

  {
    port::ScopedFlushDenormal flush;
    port::ScopedSetRound round(FE_TONEAREST);
    // 'AnnotatedTraceMe' will trace both scheduling time on host and execution
    // time on device of the OpKernel.
    profiler::AnnotatedTraceMe activity(
        [&] { return kernel_->TraceString(context, /*verbose=*/false); },
        profiler::TraceMeLevel::kInfo);
    device_->Compute(kernel_.get(), &context);
  }

  // Clean up execution op_execution_state if deferred ops aren't running.
  if (op_execution_state != nullptr) {
    op_execution_state->Unref();
  }

  Status s = context.status();
  if (TF_PREDICT_FALSE(!s.ok())) {
    if (errors::IsUnavailable(s) && !is_distributed_communication_op_) {
      s = errors::ReplaceErrorFromNonCommunicationOps(s, kernel_->name());
    }
    return s;
  }

  if (outputs != nullptr) {
    outputs->clear();
    for (int i = 0; i < context.num_outputs(); ++i) {
      const auto* output_tensor = context.mutable_output(i);
      if (output_tensor != nullptr) {
        outputs->push_back(Tensor(*output_tensor));
      } else {
        outputs->push_back(Tensor());
      }
    }
  }
  return Status::OK();
}

std::shared_ptr<FunctionLibraryRuntime::Options>
KernelAndDeviceFunc::PrepareForRun(
    ScopedStepContainer* step_container, std::vector<EagerKernelRet>* outputs,
    CancellationManager* cancellation_manager,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    const absl::optional<ManagedStackTrace>& stack_trace,
    CoordinationServiceAgent* coordination_service_agent) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_9(mht_9_v, 553, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceFunc::PrepareForRun");

  std::shared_ptr<FunctionLibraryRuntime::Options> opts = nullptr;
  if (eager_func_params.has_value()) {
    const EagerFunctionParams& params = eager_func_params.value();
    if (params.step_id.has_value()) {
      // If the function is a remote component of a cross-process function,
      // re-use the step id as its parent function's.
      opts = std::make_shared<FunctionLibraryRuntime::Options>(
          params.step_id.value());
    } else {
      opts = std::make_shared<FunctionLibraryRuntime::Options>();
    }
    // Reuse the op id if it exists.
    if (params.op_id != kInvalidOpId) {
      opts->op_id = params.op_id;
    }
  } else {
    opts = std::make_shared<FunctionLibraryRuntime::Options>();
    if (get_op_id_ && is_cross_process_) {
      // If the function is a cross-process function and the remote execution
      // goes through eager service, create an eager op id for the function.
      opts->op_id = get_op_id_();
    }
  }

  // We don't pass rendezvous from eager context because we can get tensor
  // name collisions in send/recv ops when running multiple instances
  // of the same multi-device function concurrently.
  Rendezvous* rendezvous = rendezvous_creator_(opts->step_id);
  opts->rendezvous = rendezvous;
  opts->create_rendezvous = false;

  // Create a cancellation manager to be used by FLR options if caller does not
  // pass in one. If the caller does provide one, pass it to process FLR and the
  // locally created one will be unused.
  std::shared_ptr<CancellationManager> local_cm;
  if (cancellation_manager) {
    opts->cancellation_manager = cancellation_manager;
  } else {
    opts->cancellation_manager = new CancellationManager;
  }
  opts->allow_dead_tensors = true;
  opts->step_container = step_container;
  opts->collective_executor =
      collective_executor_ ? collective_executor_->get() : nullptr;
  opts->stack_trace = stack_trace;

  opts->stats_collector = nullptr;
  opts->runner = get_runner();
  opts->coordination_service_agent = coordination_service_agent;

  outputs->clear();
  return opts;
}

Status KernelAndDeviceFunc::Run(
    ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
    std::vector<EagerKernelRet>* outputs,
    CancellationManager* cancellation_manager,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    const absl::optional<ManagedStackTrace>& stack_trace,
    CoordinationServiceAgent* coordination_service_agent) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_10(mht_10_v, 617, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceFunc::Run");

  profiler::TraceMe activity("KernelAndDeviceFunc::Run",
                             profiler::TraceMeLevel::kInfo);
  // Don't try to handle packed or remote inputs synchronously.
  if (inputs.HasRemoteOrPackedInputs() || eager_func_params.has_value()) {
    Notification n;
    Status status;
    RunAsync(step_container, inputs, outputs, cancellation_manager,
             eager_func_params, coordination_service_agent,
             [&status, &n](Status s) {
               status = s;
               n.Notify();
             });
    n.WaitForNotification();
    return status;
  }
  std::shared_ptr<FunctionLibraryRuntime::Options> opts =
      PrepareForRun(step_container, outputs, cancellation_manager,
                    eager_func_params, stack_trace, coordination_service_agent);

  std::vector<Tensor> rets;
  Status s;
  {
    port::ScopedFlushDenormal flush;
    port::ScopedSetRound round(FE_TONEAREST);
    s.Update(pflr_->RunSync(*opts, handle_, inputs.GetLocalTensors(), &rets));
  }

  if (cancellation_manager == nullptr) {
    delete opts->cancellation_manager;
  }
  static_cast<Rendezvous*>(opts->rendezvous)->Unref();
  outputs->reserve(rets.size());
  for (auto& v : rets) {
    outputs->push_back(std::move(v));
  }
  return s;
}

void KernelAndDeviceFunc::RunAsync(
    ScopedStepContainer* step_container, const EagerKernelArgs& inputs,
    std::vector<EagerKernelRet>* outputs,
    CancellationManager* cancellation_manager,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    CoordinationServiceAgent* coordination_service_agent,
    std::function<void(const Status&)> done) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_11(mht_11_v, 665, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceFunc::RunAsync");

  profiler::TraceMe activity("KernelAndDeviceFunc::RunAsync",
                             profiler::TraceMeLevel::kInfo);
  std::shared_ptr<FunctionLibraryRuntime::Options> opts = PrepareForRun(
      step_container, outputs, cancellation_manager, eager_func_params,
      absl::nullopt, coordination_service_agent);

  pflr_->Run(
      *opts, handle_, inputs, outputs,
      [opts, cancellation_manager, done = std::move(done)](const Status& s) {
        if (cancellation_manager == nullptr) {
          delete opts->cancellation_manager;
        }
        static_cast<Rendezvous*>(opts->rendezvous)->Unref();
        done(s);
      });
}

tensorflow::Device* KernelAndDeviceOp::OutputDevice(int idx) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_12(mht_12_v, 686, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceOp::OutputDevice");

  if (kernel_->output_memory_types()[idx] == HOST_MEMORY) {
    return nullptr;
  }
  return device_;
}

tensorflow::Device* KernelAndDeviceFunc::OutputDevice(int idx) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_13(mht_13_v, 696, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceFunc::OutputDevice");

  if (output_dtypes_[idx] == DT_RESOURCE) {
    return nullptr;
  }
  return output_devices_[idx];
}

tensorflow::Device* KernelAndDeviceOp::OutputResourceDevice(int idx) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_14(mht_14_v, 706, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceOp::OutputResourceDevice");

  if (kernel_->output_type(idx) == DT_RESOURCE) {
    return device_;
  }
  return nullptr;
}

tensorflow::Device* KernelAndDeviceFunc::OutputResourceDevice(int idx) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_15(mht_15_v, 716, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceFunc::OutputResourceDevice");

  if (output_dtypes_[idx] == DT_RESOURCE) {
    return output_devices_[idx];
  }
  return nullptr;
}

Device* KernelAndDeviceOp::InputDevice(int i) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_16(mht_16_v, 726, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceOp::InputDevice");

  return input_devices_[i];
}

Device* KernelAndDeviceFunc::InputDevice(int i) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSkernel_and_deviceDTcc mht_17(mht_17_v, 733, "", "./tensorflow/core/common_runtime/eager/kernel_and_device.cc", "KernelAndDeviceFunc::InputDevice");

  if ((input_dtypes_[i] == DT_RESOURCE) &&
      (composite_devices_.find(input_devices_[i]->name()) ==
       composite_devices_.end())) {
    return host_cpu_device_;
  } else {
    return input_devices_[i];
  }
}

}  // namespace tensorflow
