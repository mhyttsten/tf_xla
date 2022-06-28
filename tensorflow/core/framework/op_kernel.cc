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
class MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc() {
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

#include "tensorflow/core/framework/op_kernel.h"

#include <cstdlib>
#include <cstring>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/strings/match.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/kernel_def_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform_strings.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

const char* kJitKernelLabel = "JITCompiledKernel";
const char* kDisableJitKernelsEnvVar = "TF_DISABLE_JIT_KERNELS";

namespace {

Status MatchSignatureHelper(const DataTypeSlice expected_inputs,
                            const DataTypeSlice expected_outputs,
                            const DataTypeSlice inputs,
                            const DataTypeSlice outputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_0(mht_0_v, 240, "", "./tensorflow/core/framework/op_kernel.cc", "MatchSignatureHelper");

  bool signature_mismatch = false;

  if (inputs.size() != expected_inputs.size()) signature_mismatch = true;
  for (size_t i = 0; !signature_mismatch && i < inputs.size(); ++i) {
    if (!TypesCompatible(expected_inputs[i], inputs[i])) {
      signature_mismatch = true;
    }
  }

  if (outputs.size() != expected_outputs.size()) signature_mismatch = true;
  for (size_t i = 0; !signature_mismatch && i < outputs.size(); ++i) {
    if (!TypesCompatible(expected_outputs[i], outputs[i])) {
      signature_mismatch = true;
    }
  }

  if (signature_mismatch) {
    return errors::InvalidArgument(
        "Signature mismatch, have: ", DataTypeSliceString(inputs), "->",
        DataTypeSliceString(outputs),
        " expected: ", DataTypeSliceString(expected_inputs), "->",
        DataTypeSliceString(expected_outputs));
  }
  return Status::OK();
}

}  // namespace

// OpKernel ------------------------------------------------------------------

OpKernel::OpKernel(OpKernelConstruction* context) : OpKernel(context, false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_1(mht_1_v, 274, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernel::OpKernel");
}

OpKernel::OpKernel(OpKernelConstruction* context, bool is_deferred)
    : props_(context->props_),
      input_memory_types_(context->input_memory_types().begin(),
                          context->input_memory_types().end()),
      output_memory_types_(context->output_memory_types().begin(),
                           context->output_memory_types().end()),
      input_name_map_(context->num_inputs()),
      output_name_map_(context->num_outputs()),
      name_view_(props_->node_def.name()),
      type_string_view_(props_->node_def.op()),
      graph_def_version_(context->graph_def_version()),
      is_deferred_(is_deferred) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_2(mht_2_v, 290, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernel::OpKernel");

  OP_REQUIRES_OK(context,
                 NameRangesForNode(props_->node_def, *props_->op_def,
                                   &input_name_map_, &output_name_map_));
  OP_REQUIRES_OK(context, CheckOpDeprecation(*props_->op_def,
                                             context->graph_def_version()));

  // Kernels executing on GPU tie very few resources on the CPU where the
  // scheduler runs: we consider them as inexpensive.
  expensive_ = context->device_type() != DeviceType(DEVICE_GPU) &&
               !DeviceFactory::IsPluggableDevice(
                   DeviceTypeString(context->device_type()));
}

OpKernel::OpKernel(OpKernelConstruction* context, NodeDef&& custom_def,
                   bool is_deferred)
    : props_(std::make_shared<const NodeProperties>(
          context->props_->op_def, std::move(custom_def),
          context->props_->input_types, context->props_->output_types)),
      input_memory_types_(context->input_memory_types().begin(),
                          context->input_memory_types().end()),
      output_memory_types_(context->output_memory_types().begin(),
                           context->output_memory_types().end()),
      input_name_map_(context->num_inputs()),
      output_name_map_(context->num_outputs()),
      name_view_(props_->node_def.name()),
      type_string_view_(props_->node_def.op()),
      graph_def_version_(context->graph_def_version()),
      is_deferred_(is_deferred) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_3(mht_3_v, 321, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernel::OpKernel");

  OP_REQUIRES_OK(context,
                 NameRangesForNode(props_->node_def, *props_->op_def,
                                   &input_name_map_, &output_name_map_));
  OP_REQUIRES_OK(context, CheckOpDeprecation(*props_->op_def,
                                             context->graph_def_version()));

  // Kernels executing on GPU tie very few resources on the CPU where the
  // scheduler runs: we consider them as inexpensive.
  expensive_ = context->device_type() != DeviceType(DEVICE_GPU) &&
               !DeviceFactory::IsPluggableDevice(
                   DeviceTypeString(context->device_type()));
}

OpKernel::~OpKernel() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_4(mht_4_v, 338, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernel::~OpKernel");
}

Status OpKernel::InputRange(StringPiece input_name, int* start,
                            int* stop) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_5(mht_5_v, 344, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernel::InputRange");

  const auto result = input_name_map_.find(input_name);
  if (result == input_name_map_.end()) {
    return errors::InvalidArgument("Unknown input name: ", input_name);
  } else {
    *start = result->second.first;
    *stop = result->second.second;
    return Status::OK();
  }
}

Status OpKernel::OutputRange(StringPiece output_name, int* start,
                             int* stop) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_6(mht_6_v, 359, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernel::OutputRange");

  const auto result = output_name_map_.find(output_name);
  if (result == output_name_map_.end()) {
    return errors::InvalidArgument("Unknown output name: ", output_name);
  } else {
    *start = result->second.first;
    *stop = result->second.second;
    return Status::OK();
  }
}

string OpKernel::ShapeTraceString(const OpKernelContext& ctx) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_7(mht_7_v, 373, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernel::ShapeTraceString");

  int num_inputs = ctx.num_inputs();
  if (num_inputs == 0) return "";
  std::vector<string> tensor_shapes;
  tensor_shapes.reserve(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    if (!ctx.has_input(i)) {
      tensor_shapes.emplace_back();  // Placeholder
      continue;
    }
    DataType input_dtype = ctx.input_dtype(i);
    if (input_dtype == DataType::DT_RESOURCE ||
        input_dtype == DataType::DT_VARIANT || IsRefType(input_dtype)) {
      tensor_shapes.emplace_back();  // Placeholder
      continue;
    }
    tensor_shapes.emplace_back(strings::StrCat(
        DataTypeString(input_dtype), ctx.input(i).shape().DebugString()));
  }
  return strings::StrCat("(", absl::StrJoin(tensor_shapes, ";"), ")");
}

string OpKernel::TraceString(const OpKernelContext& ctx, bool verbose) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_8(mht_8_v, 398, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernel::TraceString");

  string trace_string = profiler::TraceMeOp(name_view(), type_string_view());
  if (verbose) {
    string shape = ShapeTraceString(ctx);
    if (!shape.empty()) {
      trace_string =
          profiler::TraceMeEncode(std::move(trace_string), {{"shape", shape}});
    }
  }
  return trace_string;
}

void AsyncOpKernel::Compute(OpKernelContext* context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_9(mht_9_v, 413, "", "./tensorflow/core/framework/op_kernel.cc", "AsyncOpKernel::Compute");

  Notification n;
  ComputeAsync(context, [&n]() { n.Notify(); });
  n.WaitForNotification();
}

// OpKernelConstruction ------------------------------------------------------

OpKernelConstruction::OpKernelConstruction(
    DeviceType device_type, DeviceBase* device, Allocator* allocator,
    FunctionLibraryRuntime* flib, ResourceMgr* resource_mgr,
    const std::shared_ptr<const NodeProperties>& props,
    const MemoryTypeSlice& input_memory_types,
    const MemoryTypeSlice& output_memory_types, int graph_def_version,
    Status* status)
    : device_type_(std::move(device_type)),
      device_(device),
      allocator_(allocator),
      flib_(flib),
      resource_mgr_(resource_mgr),
      props_(props),
      input_memory_types_(input_memory_types),
      output_memory_types_(output_memory_types),
      graph_def_version_(graph_def_version),
      status_(status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_10(mht_10_v, 440, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelConstruction::OpKernelConstruction");
}

bool OpKernelConstruction::HasAttr(StringPiece attr_name) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_11(mht_11_v, 445, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelConstruction::HasAttr");

  return HasNodeAttr(def(), attr_name);
}

void OpKernelConstruction::SetStatus(const Status& status) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_12(mht_12_v, 452, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelConstruction::SetStatus");

  status_->Update(status);
}

Status OpKernelConstruction::MatchSignature(
    const DataTypeSlice expected_inputs, const DataTypeSlice expected_outputs) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_13(mht_13_v, 460, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelConstruction::MatchSignature");

  return MatchSignatureHelper(expected_inputs, expected_outputs,
                              props_->input_types, props_->output_types);
}

Status OpKernelConstruction::allocate_temp(DataType type,
                                           const TensorShape& shape,
                                           Tensor* out_temp) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_14(mht_14_v, 470, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelConstruction::allocate_temp");

  AllocationAttributes attr;
  attr.allocation_will_be_logged = true;
  Tensor new_temp(allocator_, type, shape, attr);

  if (!new_temp.IsInitialized()) {
    return errors::ResourceExhausted(
        "OOM when allocating temporary tensor with shape", shape.DebugString());
  }
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordTensorAllocation(
        def().name(), LogMemory::OP_KERNEL_CONSTRUCTION_STEP_ID, new_temp);
  }
  *out_temp = new_temp;
  return Status::OK();
}

Status OpKernelConstruction::allocate_temp(DataType type,
                                           const TensorShape& shape,
                                           Tensor* out_temp,
                                           AllocatorAttributes allocator_attr) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_15(mht_15_v, 493, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelConstruction::allocate_temp");

  if (allocator_attr.scope_id != 0) {
    return errors::InvalidArgument(
        "ScopedAllocator cannot be used via OpKernelConstruction.");
  }
  Allocator* a = device_->GetAllocator(allocator_attr);
  AllocationAttributes attr;
  attr.allocation_will_be_logged = true;
  Tensor new_temp(a, type, shape, attr);

  if (!new_temp.IsInitialized()) {
    return errors::ResourceExhausted(
        "OOM when allocating temporary tensor with shape", shape.DebugString());
  }
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordTensorAllocation(
        def().name(), LogMemory::OP_KERNEL_CONSTRUCTION_STEP_ID, new_temp);
  }
  *out_temp = new_temp;
  return Status::OK();
}

// OpKernelContext -----------------------------------------------------------

const int OpKernelContext::Params::kNeverForward;
const int OpKernelContext::Params::kNoReservation;

OpKernelContext::OpKernelContext(Params* params)
    : OpKernelContext(
          params, static_cast<int>(params->op_kernel->output_types().size())) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_16(mht_16_v, 525, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::OpKernelContext");
}

OpKernelContext::OpKernelContext(Params* params, int num_outputs)
    : params_(params), outputs_(num_outputs) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_17(mht_17_v, 531, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::OpKernelContext");

  if (params_->track_allocations) {
    tracking_state_ = absl::make_unique<TrackingState>();
  }

  params_->ensure_eigen_gpu_device();
  if (params_->eigen_gpu_device != nullptr) {
    Allocator* eigen_gpu_allocator = get_allocator(AllocatorAttributes());
    Status s = params_->device->ReinitializeGpuDevice(
        this, params_->eigen_gpu_device, params_->op_device_context,
        eigen_gpu_allocator);
    if (!s.ok()) {
      SetStatus(s);
    }
  }
}

OpKernelContext::~OpKernelContext() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_18(mht_18_v, 551, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::~OpKernelContext");

  for (TensorValue& value : outputs_) {
    if (!value.is_ref()) {
      delete value.tensor;
    }
  }
  if (params_->track_allocations &&
      !tracking_state_->wrapped_allocators.empty()) {
    LOG(WARNING) << "OpKernelContext is tracking allocations but they are not "
                 << "being consumed by the StepStatsCollector.";
    for (auto& wrapped_allocator : tracking_state_->wrapped_allocators) {
      wrapped_allocator.second->GetRecordsAndUnRef();
    }
  }
}

Allocator* OpKernelContext::get_allocator(AllocatorAttributes attr) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_19(mht_19_v, 570, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::get_allocator");

  Allocator* allocator = nullptr;
  if (TF_PREDICT_FALSE(attr.scope_id > 0)) {
    allocator = params_->device->GetScopedAllocator(attr, step_id());
    CHECK(allocator);
  } else {
    allocator = params_->device->GetAllocator(attr);
  }
  if (TF_PREDICT_FALSE(track_allocations())) {
    DCHECK(tracking_state_);
    mutex_lock lock(tracking_state_->mu);
    for (const auto& wrapped : tracking_state_->wrapped_allocators) {
      if (wrapped.first == allocator) {
        return wrapped.second;
      }
    }
    TrackingAllocator* wrapped_allocator =
        new TrackingAllocator(allocator, params_->track_allocations);
    tracking_state_->wrapped_allocators.push_back(
        std::make_pair(allocator, wrapped_allocator));
    return wrapped_allocator;
  } else {
    return allocator;
  }
}

void OpKernelContext::SetStatus(const Status& status) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_20(mht_20_v, 599, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::SetStatus");

  status_.Update(status);
}

Status OpKernelContext::input(StringPiece name, const Tensor** tensor) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_21(mht_21_v, 606, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::input");

  int index;
  TF_RETURN_IF_ERROR(get_input_index(name, &index));
  if (input_is_ref(index)) {
    return errors::InvalidArgument("OpKernel used ref input name '", name,
                                   "' when non-ref input was expected");
  }
  *tensor = (*params_->inputs)[index].tensor;
  return Status::OK();
}

Status OpKernelContext::input_dtype(StringPiece name, DataType* dtype) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_22(mht_22_v, 620, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::input_dtype");

  int index;
  TF_RETURN_IF_ERROR(get_input_index(name, &index));
  const TensorValue& value((*params_->inputs)[index]);
  *dtype = value.dtype();
  return Status::OK();
}

Status OpKernelContext::input_ref_mutex(StringPiece name, mutex** out_mutex) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_23(mht_23_v, 631, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::input_ref_mutex");

  int index;
  TF_RETURN_IF_ERROR(get_input_index(name, &index));
  *out_mutex = input_ref_mutex(index);
  return Status::OK();
}

const Tensor& OpKernelContext::input(int index) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_24(mht_24_v, 641, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::input");

  CHECK_GE(index, 0);
  CHECK_LT(index, num_inputs()) << " name: " << op_kernel().name();
  CHECK(!input_is_ref(index));
  const Tensor& tensor = *((*params_->inputs)[index].tensor);
  return tensor;
}

Tensor OpKernelContext::mutable_input(int index, bool lock_held) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_25(mht_25_v, 652, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::mutable_input");

  CHECK_GE(index, 0);
  CHECK_LT(index, num_inputs());
  CHECK(input_is_ref(index));
  // return a copy of the Ref acquired while holding the mutex
  if (lock_held) {
    Tensor& tensor = *((*params_->inputs)[index].tensor);
    return tensor;
  } else {
    tf_shared_lock l(*input_ref_mutex(index));
    Tensor& tensor = *((*params_->inputs)[index].tensor);
    return tensor;
  }
}

void OpKernelContext::replace_ref_input(int index, const Tensor& tensor,
                                        bool lock_held) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_26(mht_26_v, 671, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::replace_ref_input");

  CHECK_GE(index, 0);
  CHECK_LT(index, num_inputs());
  CHECK(input_is_ref(index));
  // should only modify the tensor while holding the mutex
  if (lock_held) {
    *(*params_->inputs)[index].tensor = tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    *(*params_->inputs)[index].tensor = tensor;
  }
}

void OpKernelContext::forward_ref_input_to_ref_output(int input_index,
                                                      int output_index) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_27(mht_27_v, 688, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::forward_ref_input_to_ref_output");

  CHECK_GE(input_index, 0);
  CHECK_LT(input_index, num_inputs());
  CHECK(input_is_ref(input_index));
  set_output_ref(output_index, (*params_->inputs)[input_index].mutex_if_ref,
                 (*params_->inputs)[input_index].tensor);
}

bool OpKernelContext::forward_input_to_output_with_shape(
    int input_index, int output_index, const TensorShape& output_shape,
    Tensor** output) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_28(mht_28_v, 701, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::forward_input_to_output_with_shape");

  const auto output_attr = params_->output_attr_array == nullptr
                               ? AllocatorAttributes()
                               : output_alloc_attr(output_index);
  std::unique_ptr<Tensor> new_tensor = forward_input(
      input_index, output_index, expected_output_dtype(output_index),
      output_shape, output_memory_type(output_index), output_attr);
  if (new_tensor != nullptr) {
    // Transfer ownership to the output slot in OpKernelContext.
    outputs_[output_index] = TensorValue(new_tensor.release());
    *output = outputs_[output_index].tensor;
    return true;
  } else {
    return false;
  }
}

Status OpKernelContext::forward_input_to_output_with_shape(
    StringPiece input_name, StringPiece output_name,
    const TensorShape& output_shape, Tensor** output) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_29(mht_29_v, 723, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::forward_input_to_output_with_shape");

  int input_index, output_index;
  TF_RETURN_IF_ERROR(get_input_index(input_name, &input_index));
  TF_RETURN_IF_ERROR(get_output_index(output_name, &output_index));
  if (!forward_input_to_output_with_shape(input_index, output_index,
                                          output_shape, output)) {
    return errors::FailedPrecondition("OpKernel could not forward input '",
                                      input_name, "' to output '", output_name);
  }
  return Status::OK();
}

std::unique_ptr<Tensor> OpKernelContext::forward_input(
    int input_index, int output_index, DataType output_dtype,
    const TensorShape& output_shape, MemoryType output_memory_type,
    const AllocatorAttributes& output_attr) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_30(mht_30_v, 741, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::forward_input");

  CHECK_GE(input_index, 0);
  CHECK_LT(input_index, num_inputs());
  const TensorValue& input = (*params_->inputs)[input_index];
  // Check whether at graph construction time this output was marked
  // either for no forwarding or with a reservation for this input.
  // If it's reserved for this input we'll skip the refcount and
  // AllocatorAttribute checks.
  // TODO(tucker): Maybe we should skip all of the checks?
  bool never_forward =
      (params_->forward_from_array != nullptr && output_index >= 0 &&
       params_->forward_from_array[output_index] == Params::kNeverForward);
  if (never_forward) return nullptr;
  bool forward_expected =
      (params_->forward_from_array != nullptr && output_index >= 0 &&
       params_->forward_from_array[output_index] == input_index);
  if (!forward_expected && params_->forward_from_array != nullptr) {
    // Check for possibly conflicting forward.
    for (int i = 0; i < num_outputs(); ++i) {
      if (params_->forward_from_array[i] == input_index) {
        // This input is reserved for output i.
        return nullptr;
      }
    }
  }
  // Check that input tensor exists and is not a ref.
  if (input.tensor == nullptr || input.is_ref()) {
    CHECK(!forward_expected);
    return nullptr;
  }
  // Check that input type matches.
  if (input_dtype(input_index) != output_dtype) {
    CHECK(!forward_expected);
    return nullptr;
  }
  // Check that the input and output sizes are compatible.
  if (input.tensor->shape().num_elements() != output_shape.num_elements()) {
    CHECK(!forward_expected);
    return nullptr;
  }
  // Check that input and output memory types match, i.e.
  // that they either both live in host or both live in device memory.
  if (input_memory_type(input_index) != output_memory_type) {
    CHECK(!forward_expected);
    return nullptr;
  }
  if (!forward_expected) {
    if (!input->RefCountIsOne()) {
      return nullptr;
    }
    // Check that output allocator attributes are not more restrictive than
    // input allocator attributes.
    const auto input_attr = params_->input_alloc_attrs == nullptr
                                ? AllocatorAttributes()
                                : input_alloc_attr(input_index);
    if (!output_attr.IsEqualOrLessRestrictiveThan(input_attr)) {
      return nullptr;
    }
  }

  auto output_tensor = MakeUnique<Tensor>();
  CHECK(output_tensor->CopyFrom(*input.tensor, output_shape));
  return output_tensor;
}

Status OpKernelContext::forward_input_or_allocate_temp(
    gtl::ArraySlice<int> candidate_input_indices, DataType type,
    const TensorShape& shape, const AllocatorAttributes& allocator_attr,
    Tensor* out_temp) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_31(mht_31_v, 812, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::forward_input_or_allocate_temp");

  for (int input_index : candidate_input_indices) {
    std::unique_ptr<Tensor> new_tensor =
        forward_input(input_index, Params::kNoReservation /*output_index*/,
                      type, shape, DEVICE_MEMORY, allocator_attr);
    if (new_tensor != nullptr) {
      *out_temp = std::move(*new_tensor);
      return Status::OK();
    }
  }
  return allocate_temp(type, shape, out_temp, allocator_attr);
}

void OpKernelContext::delete_ref_input(int index, bool lock_held) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_32(mht_32_v, 828, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::delete_ref_input");

  CHECK_GE(index, 0);
  CHECK_LT(index, num_inputs());
  CHECK(input_is_ref(index));
  // should only modify the tensor while holding the mutex
  if (lock_held) {
    delete (*params_->inputs)[index].tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    delete (*params_->inputs)[index].tensor;
  }
}

Status OpKernelContext::mutable_input(StringPiece name, Tensor* tensor,
                                      bool lock_held) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_33(mht_33_v, 845, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::mutable_input");

  int index;
  TF_RETURN_IF_ERROR(get_input_index(name, &index));
  if (!input_is_ref(index)) {
    return errors::InvalidArgument("OpKernel used non-ref input name '", name,
                                   "' when ref input was expected");
  }
  // return a copy of the Ref acquired while holding the mutex
  if (lock_held) {
    *tensor = *(*params_->inputs)[index].tensor;
  } else {
    tf_shared_lock l(*input_ref_mutex(index));
    *tensor = *(*params_->inputs)[index].tensor;
  }
  return Status::OK();
}

Status OpKernelContext::replace_ref_input(StringPiece name,
                                          const Tensor& tensor,
                                          bool lock_held) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_34(mht_34_v, 867, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::replace_ref_input");

  int index;
  TF_RETURN_IF_ERROR(get_input_index(name, &index));
  if (!input_is_ref(index)) {
    return errors::InvalidArgument("OpKernel used immutable input name '", name,
                                   "' when ref input was expected");
  }
  replace_ref_input(index, tensor, lock_held);
  return Status::OK();
}

Status OpKernelContext::input_list(StringPiece name, OpInputList* list) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_35(mht_35_v, 881, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::input_list");

  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  *list = OpInputList(this, start, stop);
  return Status::OK();
}

Status OpKernelContext::mutable_input_list(StringPiece name,
                                           OpMutableInputList* list) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_36(mht_36_v, 892, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::mutable_input_list");

  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  *list = OpMutableInputList(this, start, stop);
  return Status::OK();
}

Status OpKernelContext::output_list(StringPiece name, OpOutputList* list) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_37(mht_37_v, 902, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::output_list");

  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  *list = OpOutputList(this, start, stop);
  return Status::OK();
}

void OpKernelContext::maybe_initialize_scope_id_set() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_38(mht_38_v, 912, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::maybe_initialize_scope_id_set");

  if (allocated_scope_ids_ == nullptr) {
    allocated_scope_ids_ = absl::make_unique<std::unordered_set<int32>>();
  }
}

Status OpKernelContext::allocate_output(int index, const TensorShape& shape,
                                        Tensor** tensor) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_39(mht_39_v, 922, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::allocate_output");

  if (index < 0) {
    return errors::Internal("allocate_output with bad index=", index,
                            " kernel=", params_->op_kernel->name());
  }
  if (index >= num_outputs()) {
    return errors::Internal("allocate_output with bad index=", index,
                            " num_outputs=", num_outputs(),
                            " kernel=", params_->op_kernel->name());
  }
  bool forward_expected =
      (params_->forward_from_array != nullptr && index >= 0 &&
       params_->forward_from_array[index] >= 0);
  if (forward_expected) {
    return errors::Internal(
        "Explicit allocate_output call where input forwarding required.  Try "
        "turning off the ScopedAllocator optimizer.");
  }
  AllocatorAttributes attr = output_alloc_attr(index);
  return allocate_output(index, shape, tensor, attr);
}

Status OpKernelContext::allocate_output(StringPiece name,
                                        const TensorShape& shape,
                                        Tensor** tensor) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_40(mht_40_v, 949, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::allocate_output");

  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  return allocate_output(start, shape, tensor);
}

Status OpKernelContext::allocate_output(StringPiece name,
                                        const TensorShape& shape,
                                        Tensor** tensor,
                                        AllocatorAttributes attr) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_41(mht_41_v, 967, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::allocate_output");

  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  return allocate_output(start, shape, tensor, attr);
}

Status OpKernelContext::allocate_tensor(
    DataType type, const TensorShape& shape, Tensor* out_tensor,
    AllocatorAttributes attr, const AllocationAttributes& allocation_attr) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_42(mht_42_v, 984, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::allocate_tensor");

  Allocator* a = get_allocator(attr);
  Tensor new_tensor(
      a, type, shape,
      AllocationAttributes(
          /*retry_on_failure=*/allocation_attr.retry_on_failure,
          /*allocation_will_be_logged=*/true, allocation_attr.freed_by_func));

  if (!new_tensor.IsInitialized()) {
    return errors::ResourceExhausted(
        "OOM when allocating tensor with shape", shape.DebugString(),
        " and type ", DataTypeString(type), " on ", params_->device->name(),
        " by allocator ", a->Name());
  }
  if (params_->log_memory) {
    LogMemory::RecordTensorAllocation(params_->op_kernel->name(),
                                      params_->step_id, new_tensor);
  }
  *out_tensor = std::move(new_tensor);
  return Status::OK();
}

Status OpKernelContext::allocate_output(int index, const TensorShape& shape,
                                        Tensor** output,
                                        AllocatorAttributes attr) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_43(mht_43_v, 1011, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::allocate_output");

  if (index < 0) {
    return errors::Internal("allocate_output with bad index=", index,
                            " kernel=", params_->op_kernel->name());
  }
  if (index >= num_outputs()) {
    return errors::Internal("allocate_output with bad index=", index,
                            " num_outputs=", outputs_.size(),
                            " kernel=", params_->op_kernel->name());
  }
  const DataType type = params_->op_kernel->output_type(index);
  if (IsRefType(type)) {
    return errors::Internal("allocate_output with ref type. index=", index,
                            " type=", type,
                            " kernel=", params_->op_kernel->name());
  }
  if (mutable_output(index) != nullptr) {
    return errors::Internal("allocate_output on same index multiple times.",
                            " index = ", index,
                            " mutable_output(index) = ", mutable_output(index),
                            " kernel=", params_->op_kernel->name());
  }
  if (attr.scope_id > 0) {
    maybe_initialize_scope_id_set();
    if (!allocated_scope_ids_->insert(attr.scope_id).second) {
      return errors::Internal(
          "OpKernel ", params_->op_kernel->name(),
          " called allocate_output at index ", index, " with scope_id ",
          attr.scope_id,
          " more than once.  Try turning off the ScopedAllocator optimizer.");
    }
  }
  profiler::ScopedMemoryDebugAnnotation op_annotation(
      op_kernel().name_view().data(), step_id(), "output", type,
      [&shape]() { return shape.DebugString(); });
  auto output_tensor = MakeUnique<Tensor>();
  Status s = allocate_tensor(type, shape, output_tensor.get(), attr);
  if (s.ok()) {
    outputs_[index] = TensorValue(output_tensor.release());
    *output = outputs_[index].tensor;
  }
  return s;
}

Status OpKernelContext::allocate_temp(
    DataType type, const TensorShape& shape, Tensor* out_temp,
    AllocatorAttributes allocator_attr,
    const AllocationAttributes& allocation_attr) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_44(mht_44_v, 1061, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::allocate_temp");

  if (allocator_attr.scope_id > 0) {
    // We do not allow ScopedAllocator calls from allocate_temp.
    // Here we clear the scope_id and return a temporary buffer.
    // This is because it is legal for a kernel to call allocate_temp
    // and then set_output with the temp tensor.
    //
    // We achieve memory correctness by forcing an allocation in set_output and
    // copying over the tensor from the temp buffer.  Kernels which would like
    // to avoid this performance penalty should switch to calling
    // allocate_output.
    VLOG(2) << "Warning: OpKernel " << params_->op_kernel->name()
            << " called allocate_temp with scope_id " << allocator_attr.scope_id
            << ".  Switch to allocate_output to avoid performance penalty.";
    allocator_attr.scope_id = -1;
  }
  profiler::ScopedMemoryDebugAnnotation op_annotation(
      op_kernel().name_view().data(), step_id(), "temp", type,
      [&shape]() { return shape.DebugString(); });
  Status s =
      allocate_tensor(type, shape, out_temp, allocator_attr, allocation_attr);
  if (track_allocations() && s.ok() && out_temp->TotalBytes() > 0) {
    Allocator* a = get_allocator(allocator_attr);
    if (a->TracksAllocationSizes()) {
      int64_t alloc_size = a->AllocatedSize(out_temp->tensor_data().data());
      record_temp_memory_allocation(alloc_size, *out_temp);
    }
  } else if (record_memory_consumption_) {
    DCHECK(tracking_state_);
    mutex_lock l(tracking_state_->stats_mu);
    tracking_state_->temp_memory_allocated += out_temp->TotalBytes();
  }
  return s;
}

Status OpKernelContext::get_input_index(StringPiece name,
                                        int* out_index) const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_45(mht_45_v, 1100, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::get_input_index");

  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was "
                                   "expected");
  }
  *out_index = start;
  return Status::OK();
}

Status OpKernelContext::get_output_index(StringPiece name,
                                         int* out_index) const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_46(mht_46_v, 1117, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::get_output_index");

  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  *out_index = start;
  return Status::OK();
}

Status OpKernelContext::set_output(StringPiece name, const Tensor& tensor) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_47(mht_47_v, 1133, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::set_output");

  int index;
  TF_RETURN_IF_ERROR(get_output_index(name, &index));
  set_output(index, tensor);
  return Status::OK();
}

Status OpKernelContext::set_output(StringPiece name, Tensor&& tensor) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_48(mht_48_v, 1143, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::set_output");

  int index;
  TF_RETURN_IF_ERROR(get_output_index(name, &index));
  set_output(index, std::move(tensor));
  return Status::OK();
}

bool OpKernelContext::maybe_set_output_by_allocate_and_copy(
    int index, const Tensor& tensor) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_49(mht_49_v, 1154, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::maybe_set_output_by_allocate_and_copy");

  bool allocate_and_copy = false;
  const bool never_forward =
      (params_->forward_from_array != nullptr &&
       params_->forward_from_array[index] == Params::kNeverForward);
  if (TF_PREDICT_FALSE(never_forward)) {
    maybe_initialize_scope_id_set();
    if (allocated_scope_ids_->find(output_alloc_attr(index).scope_id) ==
        allocated_scope_ids_->end()) {
      allocate_and_copy = true;
    } else {
      // The output at `index` must have been previously allocated via a call to
      // `allocate_output(index, ...)`.  That call would ensure that we return
      // the correct slice of the ScopedAllocated buffer, so we do not
      // re-allocate and copy here.
      LOG(WARNING)
          << "OpKernel " << params_->op_kernel->name()
          << " called both allocate_output and set_output with scope_id "
          << output_alloc_attr(index).scope_id;
    }
  }

  if (TF_PREDICT_FALSE(allocate_and_copy)) {
    // This output was marked to not be forwarded either during graph
    // construction or grappler passes.  Force an allocation and copy input to
    // output.
    VLOG(1) << "OpKernelContext set_output index " << index << " tensor "
            << tensor.DebugString() << " never_forward " << never_forward
            << " params_->forward_from_array[index] "
            << params_->forward_from_array[index] << " alloc_attr.scope_id "
            << output_alloc_attr(index).scope_id;
    profiler::ScopedMemoryDebugAnnotation op_annotation(
        op_kernel().name_view().data(), step_id(), "output", tensor.dtype(),
        [&tensor]() { return tensor.shape().DebugString(); });
    auto new_tensor = MakeUnique<Tensor>();
    Status s = allocate_tensor(tensor.dtype(), tensor.shape(), new_tensor.get(),
                               output_alloc_attr(index));
    TF_CHECK_OK(s);
    device()->CopyTensorInSameDevice(&tensor, new_tensor.get(),
                                     op_device_context(), [](const Status&) {});
    outputs_[index] = TensorValue(new_tensor.release());
  }
  return allocate_and_copy;
}

void OpKernelContext::maybe_track_allocations_for_set_output(
    const Tensor& tensor) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_50(mht_50_v, 1203, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::maybe_track_allocations_for_set_output");

  if (TF_PREDICT_FALSE(track_allocations()) && tensor.TotalBytes() > 0) {
    DCHECK(tracking_state_);
    mutex_lock l(tracking_state_->stats_mu);
    const auto it = std::find_if(
        tracking_state_->temp_tensor_buffer_and_size.begin(),
        tracking_state_->temp_tensor_buffer_and_size.end(),
        [&tensor](const std::pair<const void*, int64>& e) {
          return e.first ==
                 static_cast<const void*>(tensor.tensor_data().data());
        });
    if (it != tracking_state_->temp_tensor_buffer_and_size.end()) {
      tracking_state_->temp_memory_allocated -= it->second;
      tracking_state_->temp_tensor_buffer_and_size.erase(it);
    }
  }
}

void OpKernelContext::set_output(int index, const Tensor& tensor) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_51(mht_51_v, 1224, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::set_output");

  CHECK_GE(index, 0);
  CHECK_LT(index, outputs_.size());
  const DataType type = params_->op_kernel->output_type(index);
  CHECK(!IsRefType(type));
  CHECK_EQ(outputs_[index].tensor, nullptr);
  if (TF_PREDICT_TRUE(!maybe_set_output_by_allocate_and_copy(index, tensor))) {
    // Input can be forwarded to output; incref on `tensor` and set output at
    // `index` to this tensor.
    outputs_[index] = TensorValue(new Tensor(tensor));
    maybe_track_allocations_for_set_output(*outputs_[index].tensor);
  }
}

void OpKernelContext::set_output(int index, Tensor&& tensor) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_52(mht_52_v, 1241, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::set_output");

  CHECK_GE(index, 0);
  CHECK_LT(index, outputs_.size());
  const DataType type = params_->op_kernel->output_type(index);
  CHECK(!IsRefType(type));
  CHECK_EQ(outputs_[index].tensor, nullptr);
  if (TF_PREDICT_TRUE(!maybe_set_output_by_allocate_and_copy(index, tensor))) {
    // Input can be forwarded to output; set output at `index` to this tensor.
    outputs_[index] = TensorValue(new Tensor(std::move(tensor)));
    maybe_track_allocations_for_set_output(*outputs_[index].tensor);
  }
}

void OpKernelContext::set_output_ref(int index, mutex* mu,
                                     Tensor* tensor_for_ref) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_53(mht_53_v, 1258, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::set_output_ref");

  CHECK_GE(index, 0);
  CHECK_LT(index, outputs_.size());
  CHECK(IsRefType(params_->op_kernel->output_type(index)));
  outputs_[index] = TensorValue(mu, tensor_for_ref);
}

Status OpKernelContext::set_output_ref(StringPiece name, mutex* mu,
                                       Tensor* tensor_for_ref) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_54(mht_54_v, 1269, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::set_output_ref");

  int index;
  TF_RETURN_IF_ERROR(get_output_index(name, &index));
  set_output_ref(index, mu, tensor_for_ref);
  return Status::OK();
}

Status OpKernelContext::mutable_output(StringPiece name, Tensor** tensor) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_55(mht_55_v, 1279, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::mutable_output");

  int index;
  TF_RETURN_IF_ERROR(get_output_index(name, &index));
  *tensor = mutable_output(index);
  return Status::OK();
}

bool OpKernelContext::ValidateInputsAreSameShape(OpKernel* op) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_56(mht_56_v, 1289, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::ValidateInputsAreSameShape");

  const auto& inputs = *params_->inputs;
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (!inputs[0]->IsSameSize(*(inputs[i].tensor))) {
      SetStatus(errors::InvalidArgument(
          "Inputs to operation ", op->name(), " of type ", op->type_string(),
          " must have the same size and shape.  Input 0: ",
          inputs[0]->shape().DebugString(), " != input ", i, ": ",
          inputs[i]->shape().DebugString()));
      return false;
    }
  }
  return true;
}

Status OpKernelContext::MatchSignature(const DataTypeSlice expected_inputs,
                                       const DataTypeSlice expected_outputs) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_57(mht_57_v, 1308, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::MatchSignature");

  DataTypeVector inputs;
  for (const TensorValue& t : *params_->inputs) {
    inputs.push_back(t.dtype());
  }
  DataTypeVector outputs = params_->op_kernel->output_types();
  return MatchSignatureHelper(expected_inputs, expected_outputs, inputs,
                              outputs);
}

void OpKernelContext::record_temp_memory_allocation(int64_t size,
                                                    const Tensor& t) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_58(mht_58_v, 1322, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::record_temp_memory_allocation");

  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    tracking_state_->temp_memory_allocated += size;
    tracking_state_->temp_tensor_buffer_and_size.emplace_back(
        static_cast<const void*>(t.tensor_data().data()), size);
  }
}

int64_t OpKernelContext::temp_memory_allocated() const {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_59(mht_59_v, 1334, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::temp_memory_allocated");

  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    return tracking_state_->temp_memory_allocated;
  } else {
    return 0;
  }
}

void OpKernelContext::record_persistent_memory_allocation(int64_t size,
                                                          int64_t alloc_id) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_60(mht_60_v, 1347, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::record_persistent_memory_allocation");

  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    tracking_state_->persistent_memory_allocated += size;
    if (alloc_id >= 0) {
      tracking_state_->persistent_alloc_ids.push_back(alloc_id);
    }
  }
}

int64_t OpKernelContext::persistent_memory_allocated() const {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_61(mht_61_v, 1360, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::persistent_memory_allocated");

  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    return tracking_state_->persistent_memory_allocated;
  } else {
    return 0;
  }
}

std::vector<int64_t> OpKernelContext::persistent_alloc_ids() const {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_62(mht_62_v, 1372, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::persistent_alloc_ids");

  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    return std::vector<int64_t>(tracking_state_->persistent_alloc_ids.begin(),
                                tracking_state_->persistent_alloc_ids.end());
  } else {
    return std::vector<int64_t>();
  }
}

void OpKernelContext::clear_recorded_memory() {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_63(mht_63_v, 1385, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::clear_recorded_memory");

  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    tracking_state_->temp_memory_allocated = 0;
    tracking_state_->persistent_memory_allocated = 0;
    tracking_state_->temp_tensor_buffer_and_size.clear();
    tracking_state_->persistent_alloc_ids.clear();
  }
}

void OpKernelContext::set_record_memory_consumption(bool v) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_64(mht_64_v, 1398, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::set_record_memory_consumption");

  record_memory_consumption_ = v;
  if (v && !tracking_state_) {
    tracking_state_ = absl::make_unique<TrackingState>();
  }
}

const string& OpKernelContext::executor_type() const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_65(mht_65_v, 1408, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::executor_type");

  if (params_->executor_type) {
    return *params_->executor_type;
  } else {
    static const string& kEmptyString = *new string("");
    return kEmptyString;
  }
}

// OpKernel registration ------------------------------------------------------

struct KernelRegistration {
  KernelRegistration(const KernelDef& d, StringPiece c,
                     std::unique_ptr<kernel_factory::OpKernelFactory> f)
      : def(d), kernel_class_name(c), factory(std::move(f)) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_66(mht_66_v, 1425, "", "./tensorflow/core/framework/op_kernel.cc", "KernelRegistration");
}

  const KernelDef def;
  const string kernel_class_name;
  std::unique_ptr<kernel_factory::OpKernelFactory> factory;
};

// This maps from 'op_type' + DeviceType to the set of KernelDefs and
// factory functions for instantiating the OpKernel that matches the
// KernelDef.
struct KernelRegistry {
  mutex mu;
  std::unordered_multimap<string, KernelRegistration> registry
      TF_GUARDED_BY(mu);
};

#if defined(_WIN32)
static const char kKernelLibPattern[] = "libtfkernel*.dll";
#elif defined(__APPLE__)
static const char kKernelLibPattern[] = "libtfkernel*.dylib";
#else
static const char kKernelLibPattern[] = "libtfkernel*.so";
#endif

#define FEATURE(x) \
  { x, #x }

// Returns Status::OK if the dynamic library at the given path is safe to
// load with some level of confidence.
static Status IsProbablySafeToLoad(const string& path) {
   std::vector<std::string> mht_67_v;
   mht_67_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_67(mht_67_v, 1458, "", "./tensorflow/core/framework/op_kernel.cc", "IsProbablySafeToLoad");

  // A map of platform string to required CPU feature.
  using port::CPUFeature;
  static const auto* feature_map =
      new std::map<string, std::pair<CPUFeature, string>>{
          {"__AVX512VL__=1", FEATURE(CPUFeature::AVX512VL)},
      };

  std::vector<std::string> platform_strings;
  int result = GetPlatformStrings(path, &platform_strings);
  if (result) {
    return Status(error::Code::UNKNOWN, strerror(result));
  }
  if (platform_strings.empty()) {
    return Status(error::Code::FAILED_PRECONDITION,
                  "Didn't find any platform strings");
  }
  std::vector<std::string> missing_features;
  for (const auto& platform_string : platform_strings) {
    const auto& entry = feature_map->find(platform_string);
    if (entry != feature_map->end() &&
        !port::TestCPUFeature(entry->second.first)) {
      missing_features.emplace_back(entry->second.second);
    }
  }
  if (!missing_features.empty()) {
    string errmsg = "Missing CPU features: ";
    errmsg.append(absl::StrJoin(missing_features, ", "));
    return Status(errors::Code::FAILED_PRECONDITION, errmsg);
  }
  return Status::OK();
}

void LoadDynamicKernelsInternal() {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_68(mht_68_v, 1494, "", "./tensorflow/core/framework/op_kernel.cc", "LoadDynamicKernelsInternal");

  Env* env = Env::Default();

  // Override to allow loading unsafe packages for development.
  // DO NOT USE UNLESS YOU KNOW WHAT ABI ISSUES YOU CAN ENCOUNTER.
  char* _abi_check_env_var = getenv("TF_REALLY_LOAD_UNSAFE_PACKAGES");
  bool override_abi_check = false;
  if (_abi_check_env_var != nullptr) {
    override_abi_check = strcmp(_abi_check_env_var, "1") == 0;
  }

  string bazel_kernel_dir =
      io::JoinPath(env->GetRunfilesDir(), "tensorflow", "core", "kernels");
  std::vector<string> files;
  Status s_kernel_dir = env->GetChildren(bazel_kernel_dir, &files);
  if (s_kernel_dir.ok()) {
    string dll_spec = io::JoinPath(bazel_kernel_dir, kKernelLibPattern);
    for (const auto& file : files) {
      string fullpath = io::JoinPath(bazel_kernel_dir, file);
      if (env->MatchPath(fullpath, dll_spec)) {
        Status s = IsProbablySafeToLoad(fullpath);
        if (!s.ok() && override_abi_check) {
          LOG(WARNING) << "Loading UNSAFE library " << fullpath
                       << " because ABI check override is set: "
                       << s.error_message();
        }
        if (s.ok() || override_abi_check) {
          // TODO(gunan): Store the handles to the opened files.
          void* unused_filehandle;
          TF_CHECK_OK(
              env->LoadDynamicLibrary(fullpath.c_str(), &unused_filehandle));
        } else {
          LOG(WARNING) << "Not loading plugin library " << fullpath << ": "
                       << s.error_message();
        }
      }
    }
  }
}

// Mechanism for loading existing kernel libraries.
void LoadDynamicKernels() {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_69(mht_69_v, 1538, "", "./tensorflow/core/framework/op_kernel.cc", "LoadDynamicKernels");

  // TODO(gunan): As more features are available, add intelligent kernel
  // selection, and dropping unsuitable kernel logic here.
  static absl::once_flag dll_loader_flag;
  absl::call_once(dll_loader_flag, LoadDynamicKernelsInternal);
}

static string Key(StringPiece op_type, const DeviceType& device_type,
                  StringPiece label) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_70(mht_70_v, 1549, "", "./tensorflow/core/framework/op_kernel.cc", "Key");

  return strings::StrCat(op_type, ":", DeviceTypeString(device_type), ":",
                         label);
}

// Provide a way for users to disable JIT kernels for a transitional period.
// Until this is removed, this function also removes the JIT label that is added
// to JIT kernels during the static registration, to allow them to be found
// during lookup as normal kernels.
void SetupOrDisableJit(KernelRegistry* registry) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_71(mht_71_v, 1561, "", "./tensorflow/core/framework/op_kernel.cc", "SetupOrDisableJit");

  std::unordered_multimap<string, KernelRegistration> jit_kernels;
  bool remove_jit_kernels = absl::StrContains(
      absl::NullSafeStringView(getenv(kDisableJitKernelsEnvVar)), "1");

  mutex_lock l(registry->mu);
  std::unordered_multimap<string, KernelRegistration>& all_kernels =
      registry->registry;
  auto it = all_kernels.begin();
  while (it != all_kernels.end()) {
    if (absl::StrContains(it->second.def.label(), kJitKernelLabel)) {
      // Remove all kernels that have the jit label. They will be added back
      // without the label if they are not to be disabled.
      KernelDef def_without_label = it->second.def;
      def_without_label.set_label("");

      if (!remove_jit_kernels) {
        jit_kernels.emplace(
            Key(def_without_label.op(),
                DeviceType(def_without_label.device_type()),
                def_without_label.label()),
            KernelRegistration(def_without_label, it->second.kernel_class_name,
                               std::move(it->second.factory)));
      }

      it = all_kernels.erase(it);
    } else {
      it++;
    }
  }

  // Add back kernels if they are not disabled. This new key-value pair have all
  // references to the label removed.
  for (auto& jit_kernel : jit_kernels) {
    all_kernels.insert(std::move(jit_kernel));
  }
}

void* GlobalKernelRegistry() {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_72(mht_72_v, 1602, "", "./tensorflow/core/framework/op_kernel.cc", "GlobalKernelRegistry");

  static KernelRegistry* global_kernel_registry = []() {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_73(mht_73_v, 1606, "", "./tensorflow/core/framework/op_kernel.cc", "lambda");

    KernelRegistry* registry = new KernelRegistry;
    OpRegistry::Global()->RegisterValidator(ValidateKernelRegistrations);
    return registry;
  }();
  return global_kernel_registry;
}

static KernelRegistry* GlobalKernelRegistryTyped() {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_74(mht_74_v, 1617, "", "./tensorflow/core/framework/op_kernel.cc", "GlobalKernelRegistryTyped");

#ifdef AUTOLOAD_DYNAMIC_KERNELS
  LoadDynamicKernels();
#endif  // AUTOLOAD_DYNAMIC_KERNELS
  auto* registry = reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
  // Update or disable JIT kernels based on user configuration. This is a
  // temporary fallback as part of the initial release of JIT kernels.
  static absl::once_flag setup_or_disable_jit;
  absl::call_once(setup_or_disable_jit, SetupOrDisableJit, registry);
  return registry;
}

namespace kernel_factory {

void OpKernelRegistrar::InitInternal(const KernelDef* kernel_def,
                                     StringPiece kernel_class_name,
                                     std::unique_ptr<OpKernelFactory> factory) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_75(mht_75_v, 1636, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelRegistrar::InitInternal");

  const string key =
      Key(kernel_def->op(), DeviceType(kernel_def->device_type()),
          kernel_def->label());

  // To avoid calling LoadDynamicKernels DO NOT CALL GlobalKernelRegistryTyped
  // here.
  // InitInternal gets called by static initializers, so it ends up executing
  // before main. This causes LoadKernelLibraries function to get called
  // before some file libraries can initialize, which in turn crashes the
  // program flakily. Until we get rid of static initializers in kernel
  // registration mechanism, we have this workaround here.
  auto global_registry =
      reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
  mutex_lock l(global_registry->mu);
  global_registry->registry.emplace(
      key,
      KernelRegistration(*kernel_def, kernel_class_name, std::move(factory)));
  delete kernel_def;
}

OpKernel* OpKernelRegistrar::PtrOpKernelFactory::Create(
    OpKernelConstruction* context) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_76(mht_76_v, 1661, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelRegistrar::PtrOpKernelFactory::Create");

  return (*create_func_)(context);
}

}  // namespace kernel_factory

namespace {

// Label defaults to empty if not found in NodeDef.
const string& GetKernelLabelAttr(const AttrSlice& node_attrs) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_77(mht_77_v, 1673, "", "./tensorflow/core/framework/op_kernel.cc", "GetKernelLabelAttr");

  static const string& kKernelAttr = *new string("_kernel");
  static const string& kEmptyString = *new string("");

  // NOTE: We inline the implementation of `GetNodeAttrString()` here in order
  // to use the `AttrSlice::FindByString()` overload, which does a more
  // efficient map lookup (instead of a linear scan) when the attribute name is
  // already a `const string&`.
  const AttrValue* attr_value = node_attrs.FindByString(kKernelAttr);
  if (attr_value == nullptr || attr_value->value_case() != AttrValue::kS)
    return kEmptyString;
  else
    return attr_value->s();
}

// TODO(irving): Replace with const Node& version below.
Status FindKernelRegistration(
    const DeviceType& device_type, StringPiece node_name,
    bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info,
    StringPiece node_op, AttrSlice node_attrs, const KernelRegistration** reg,
    bool* was_attr_mismatch) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_78(mht_78_v, 1697, "", "./tensorflow/core/framework/op_kernel.cc", "FindKernelRegistration");

  *reg = nullptr;
  *was_attr_mismatch = false;

  const string& label = GetKernelLabelAttr(node_attrs);

  const string key = Key(node_op, device_type, label);
  auto typed_registry = GlobalKernelRegistryTyped();
  tf_shared_lock lock(typed_registry->mu);
  auto regs = typed_registry->registry.equal_range(key);
  for (auto iter = regs.first; iter != regs.second; ++iter) {
    // If there is a kernel registered for the op and device_type,
    // check that the attrs match.
    bool match;
    TF_RETURN_IF_ERROR(KernelAttrsMatch(iter->second.def, node_attrs, &match));
    if (match) {
      if (*reg != nullptr) {
        if ((*reg)->def.priority() == iter->second.def.priority()) {
          return errors::InvalidArgument(
              "Multiple OpKernel registrations match NodeDef at the same "
              "priority '",
              FormatNodeDefForError(node_name, has_experimental_debug_info,
                                    experimental_debug_info),
              "': '", (*reg)->def.ShortDebugString(), "' and '",
              iter->second.def.ShortDebugString(), "'");
        } else if ((*reg)->def.priority() > iter->second.def.priority()) {
          continue;
        }
        // iter->second's priority is higher than *reg.
      }
      *reg = &iter->second;
    } else {
      *was_attr_mismatch = true;
    }
  }
  // Check if no device specific registrations found. If not, try finding a
  // default kernel.
  if (*reg == nullptr &&
      !IsSymbolicExecutionDevice(device_type.type_string())) {
    const string default_key = Key(node_op, DEVICE_DEFAULT, label);
    auto regs = typed_registry->registry.equal_range(default_key);
    for (auto iter = regs.first; iter != regs.second; ++iter) {
      // If there is a kernel registered for the op and device_type,
      // check that the attrs match.
      bool match;
      TF_RETURN_IF_ERROR(
          KernelAttrsMatch(iter->second.def, node_attrs, &match));
      if (match) {
        if (*reg != nullptr) {
          return errors::InvalidArgument(
              "Multiple Default OpKernel registrations match NodeDef '",
              FormatNodeDefForError(node_name, has_experimental_debug_info,
                                    experimental_debug_info),
              "': '", (*reg)->def.ShortDebugString(), "' and '",
              iter->second.def.ShortDebugString(), "'");
        }
        *reg = &iter->second;
      } else {
        *was_attr_mismatch = true;
      }
    }

    if (*reg != nullptr) {
      VLOG(1) << "No device-specific kernels found for NodeDef '"
              << FormatNodeDefForError(node_name, has_experimental_debug_info,
                                       experimental_debug_info)
              << "'"
              << "Will fall back to a default kernel." << std::endl;
    }
  }

  return Status::OK();
}

Status FindKernelRegistration(const DeviceType& device_type,
                              const NodeDef& node_def,
                              const KernelRegistration** reg,
                              bool* was_attr_mismatch) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_79(mht_79_v, 1777, "", "./tensorflow/core/framework/op_kernel.cc", "FindKernelRegistration");

  return FindKernelRegistration(
      device_type, node_def.name(), node_def.has_experimental_debug_info(),
      node_def.experimental_debug_info(), node_def.op(),
      AttrSlice(&node_def.attr()), reg, was_attr_mismatch);
}

}  // namespace

bool KernelDefAvailable(const DeviceType& device_type,
                        const NodeDef& node_def) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_80(mht_80_v, 1790, "", "./tensorflow/core/framework/op_kernel.cc", "KernelDefAvailable");

  const KernelRegistration* reg = nullptr;
  bool was_attr_mismatch;
  Status result =
      FindKernelRegistration(device_type, node_def, &reg, &was_attr_mismatch);
  return result.ok() && reg != nullptr;
}

// TODO(irving): Change const NodeDef& to const Node&
Status FindKernelDef(
    const DeviceType& device_type, StringPiece node_name,
    bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info,
    StringPiece node_op, StringPiece node_device, AttrSlice node_attrs,
    const KernelDef** def, string* kernel_class_name) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_81(mht_81_v, 1807, "", "./tensorflow/core/framework/op_kernel.cc", "FindKernelDef");

  const KernelRegistration* reg = nullptr;
  bool was_attr_mismatch;
  TF_RETURN_IF_ERROR(FindKernelRegistration(
      device_type, node_name, has_experimental_debug_info,
      experimental_debug_info, node_op, node_attrs, &reg, &was_attr_mismatch));
  if (reg == nullptr) {
    const std::string device_str = DeviceTypeString(device_type);
    Status s = errors::NotFound(
        "No registered '", node_op, "' OpKernel for ", device_str,
        " devices compatible with node ",
        FormatNodeDefForError(node_name, has_experimental_debug_info,
                              experimental_debug_info));
    if (was_attr_mismatch) {
      errors::AppendToMessage(
          &s, " (OpKernel was found, but attributes didn't match) ",
          "Requested Attributes: ",
          SummarizeAttrsHelper(node_attrs, node_device));
    }

    // Do not print kernel registrations for other devices when using _JIT
    // devices for compilation.
    if (!absl::StrContains(device_str, "JIT")) {
      errors::AppendToMessage(
          &s, ".  Registered:", KernelsRegisteredForOp(node_op));
    }

    return s;
  }
  if (def != nullptr) *def = &reg->def;
  if (kernel_class_name != nullptr) *kernel_class_name = reg->kernel_class_name;
  return Status::OK();
}

Status FindKernelDef(const DeviceType& device_type, const NodeDef& node_def,
                     const KernelDef** def, string* kernel_class_name) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_82(mht_82_v, 1845, "", "./tensorflow/core/framework/op_kernel.cc", "FindKernelDef");

  return FindKernelDef(
      device_type, node_def.name(), node_def.has_experimental_debug_info(),
      node_def.experimental_debug_info(), node_def.op(), node_def.device(),
      AttrSlice(&node_def.attr()), def, kernel_class_name);
}

Status SupportedDeviceTypesForNode(
    const std::vector<DeviceType>& prioritized_types, const NodeDef& def,
    PrioritizedDeviceTypeVector* prioritized_device_types,
    const DeviceNameUtils::ParsedName* local_address_spec) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_83(mht_83_v, 1858, "", "./tensorflow/core/framework/op_kernel.cc", "SupportedDeviceTypesForNode");

  // TODO(zhifengc): Changes the callers (SimplePlacer and
  // DynamicPlacer) to consider the possibility that 'def' is call to
  // a user-defined function and only calls this
  // SupportedDeviceTypesForNode for primitive ops.
  const OpRegistrationData* op_reg_data;
  const Status s = OpRegistry::Global()->LookUp(def.op(), &op_reg_data);
  if (s.ok()) {
    bool exists_attr_mismatch = false;
    for (const DeviceType& device_type : prioritized_types) {
      const KernelRegistration* reg = nullptr;
      bool was_attr_mismatch = false;
      TF_RETURN_IF_ERROR(
          FindKernelRegistration(device_type, def, &reg, &was_attr_mismatch));
      exists_attr_mismatch = exists_attr_mismatch || was_attr_mismatch;
      if (reg != nullptr) {
        int32_t priority = reg->def.priority();
        prioritized_device_types->emplace_back(device_type, priority);
      }
    }
    // Add extra supported device types if the following conditions are
    // satisfied:
    // 1) No kernel is defined for the given op (e.g. PyFunc on worker process)
    // 2) A device is requested for this node which specifies job/replica/task
    // 3) A local device is provided which specifies job/replica/task
    // 4) The local device does not have the same (job, replica, task) as the
    //    requested device
    //
    // The goal is to address the issue where a graph includes op (e.g. PyFunc)
    // whose kernel is known to a remote process but not to the current process.
    if (prioritized_device_types->empty() && !exists_attr_mismatch &&
        local_address_spec != nullptr) {
      DeviceNameUtils::ParsedName requested_device_name;
      DeviceNameUtils::ParseFullName(def.device(), &requested_device_name);
      if (DeviceNameUtils::IsDifferentAddressSpace(*local_address_spec,
                                                   requested_device_name)) {
        if (requested_device_name.has_type) {
          prioritized_device_types->push_back(
              std::make_pair(DeviceType(requested_device_name.type), 0));
        } else {
          for (const DeviceType& device_type : prioritized_types) {
            prioritized_device_types->push_back(std::make_pair(device_type, 0));
          }
        }
      }
    }

    // If we were unable to find any valid devices let's validate if the node is
    // even valid.
    if (prioritized_device_types->empty()) {
      TF_RETURN_IF_ERROR(ValidateNodeDef(def, op_reg_data->op_def));
    }

    std::stable_sort(prioritized_device_types->begin(),
                     prioritized_device_types->end(),
                     [](const std::pair<DeviceType, int32>& a,
                        const std::pair<DeviceType, int32>& b) {
                       return a.second > b.second;
                     });
  } else {
    // Assumes that all device types support this node.
    for (const DeviceType& device_type : prioritized_types) {
      prioritized_device_types->push_back(std::make_pair(device_type, 0));
    }
  }
  return Status::OK();
}

void LogAllRegisteredKernels() {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_84(mht_84_v, 1929, "", "./tensorflow/core/framework/op_kernel.cc", "LogAllRegisteredKernels");

  KernelList kernel_list = GetAllRegisteredKernels();
  for (const auto& kernel_def : kernel_list.kernel()) {
    LOG(INFO) << "OpKernel ('" << kernel_def.ShortDebugString() << "')";
  }
}

KernelList GetAllRegisteredKernels() {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_85(mht_85_v, 1939, "", "./tensorflow/core/framework/op_kernel.cc", "GetAllRegisteredKernels");

  return GetFilteredRegisteredKernels([](const KernelDef& k) { return true; });
}

KernelList GetFilteredRegisteredKernels(
    const std::function<bool(const KernelDef&)>& predicate) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_86(mht_86_v, 1947, "", "./tensorflow/core/framework/op_kernel.cc", "GetFilteredRegisteredKernels");

  KernelRegistry* const typed_registry = GlobalKernelRegistryTyped();
  KernelList kernel_list;
  tf_shared_lock lock(typed_registry->mu);
  kernel_list.mutable_kernel()->Reserve(typed_registry->registry.size());
  for (const auto& p : typed_registry->registry) {
    const KernelDef& kernel_def = p.second.def;
    if (predicate(kernel_def)) {
      *kernel_list.add_kernel() = kernel_def;
    }
  }
  return kernel_list;
}

KernelList GetRegisteredKernelsForOp(StringPiece op_name) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_87(mht_87_v, 1964, "", "./tensorflow/core/framework/op_kernel.cc", "GetRegisteredKernelsForOp");

  auto op_pred = [op_name](const KernelDef& k) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_88(mht_88_v, 1968, "", "./tensorflow/core/framework/op_kernel.cc", "lambda");
 return k.op() == op_name; };
  return GetFilteredRegisteredKernels(op_pred);
}

string KernelsRegisteredForOp(StringPiece op_name) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_89(mht_89_v, 1975, "", "./tensorflow/core/framework/op_kernel.cc", "KernelsRegisteredForOp");

  KernelList kernel_list = GetRegisteredKernelsForOp(op_name);
  if (kernel_list.kernel_size() == 0) return "  <no registered kernels>\n";
  string ret;
  for (const auto& kernel_def : kernel_list.kernel()) {
    strings::StrAppend(&ret, "  device='", kernel_def.device_type(), "'");
    if (!kernel_def.label().empty()) {
      strings::StrAppend(&ret, "; label='", kernel_def.label(), "'");
    }
    for (int i = 0; i < kernel_def.constraint_size(); ++i) {
      strings::StrAppend(
          &ret, "; ", kernel_def.constraint(i).name(), " in ",
          SummarizeAttrValue(kernel_def.constraint(i).allowed_values()));
    }
    strings::StrAppend(&ret, "\n");
  }
  return ret;
}

/* TODO(rmlarsen): This API is deprecated. Remove it if possible to avoid
 * copying the NodeDef. */
std::unique_ptr<OpKernel> CreateOpKernel(
    DeviceType device_type, DeviceBase* device, Allocator* allocator,
    const NodeDef& node_def, int graph_def_version, Status* status) {
  // Look up the Op registered for this op name.
  std::shared_ptr<const NodeProperties> props;
  status->Update(NodeProperties::CreateFromNodeDef(
      node_def, OpRegistry::Global(), &props));
  if (!status->ok()) {
    errors::AppendToMessage(status,
                            " for node: ", FormatNodeDefForError(node_def));
    return nullptr;
  }
  return CreateOpKernel(device_type, device, allocator, props,
                        graph_def_version, status);
}

std::unique_ptr<OpKernel> CreateOpKernel(
    DeviceType device_type, DeviceBase* device, Allocator* allocator,
    const std::shared_ptr<const NodeProperties>& props, int graph_def_version,
    Status* status) {
  OpKernel* kernel = nullptr;
  *status = CreateOpKernel(std::move(device_type), device, allocator,
                           /*flib=*/nullptr, props, graph_def_version, &kernel);
  return std::unique_ptr<OpKernel>(kernel);
}

Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
                      Allocator* allocator, FunctionLibraryRuntime* flib,
                      const std::shared_ptr<const NodeProperties>& props,
                      int graph_def_version, OpKernel** kernel) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_90(mht_90_v, 2028, "", "./tensorflow/core/framework/op_kernel.cc", "CreateOpKernel");

  return CreateOpKernel(std::move(device_type), device, allocator, flib,
                        /* resource_mgr= */ nullptr, props, graph_def_version,
                        kernel);
}

Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
                      Allocator* allocator, FunctionLibraryRuntime* flib,
                      ResourceMgr* resource_mgr,
                      const std::shared_ptr<const NodeProperties>& props,
                      int graph_def_version, OpKernel** kernel) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_91(mht_91_v, 2041, "", "./tensorflow/core/framework/op_kernel.cc", "CreateOpKernel");

  const NodeDef& node_def = props->node_def;
  bool was_attr_mismatch;
  const KernelRegistration* registration = nullptr;
  Status s;
  if (props != nullptr) {
    VLOG(1) << "Instantiating kernel for node: " << SummarizeNodeDef(node_def);

    // Validate node_def against OpDef.
    TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, *props->op_def));

    // Look up kernel registration.
    s = FindKernelRegistration(device_type, node_def, &registration,
                               &was_attr_mismatch);
    if (!s.ok()) {
      errors::AppendToMessage(&s, " when instantiating ", node_def.op());
      return s;
    }
  }
  if (registration == nullptr) {
    s.Update(errors::NotFound("No registered '", node_def.op(),
                              "' OpKernel for '", DeviceTypeString(device_type),
                              "' devices compatible with node ",
                              FormatNodeDefForError(node_def)));
    if (was_attr_mismatch) {
      errors::AppendToMessage(
          &s, " (OpKernel was found, but attributes didn't match) ",
          "Requested Attributes: ", SummarizeAttrs(node_def));
    }
    errors::AppendToMessage(
        &s, ".  Registered:", KernelsRegisteredForOp(node_def.op()));
    return s;
  }

  // We are creating a kernel for an op registered in
  // OpRegistry::Global(), we consult the kernel registry to decide
  // the kernel's input and output memory types.
  MemoryTypeVector input_memory_types;
  MemoryTypeVector output_memory_types;
  TF_RETURN_IF_ERROR(MemoryTypesForNode(OpRegistry::Global(), device_type,
                                        node_def, &input_memory_types,
                                        &output_memory_types));

  // Everything needed for OpKernel construction.
  OpKernelConstruction context(std::move(device_type), device, allocator, flib,
                               resource_mgr, props, input_memory_types,
                               output_memory_types, graph_def_version, &s);
  *kernel = registration->factory->Create(&context);
  if (!s.ok()) {
    delete *kernel;
    *kernel = nullptr;
  }
  return s;
}

namespace {

bool FindArgInOp(StringPiece arg_name,
                 const protobuf::RepeatedPtrField<OpDef::ArgDef>& args) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_92(mht_92_v, 2102, "", "./tensorflow/core/framework/op_kernel.cc", "FindArgInOp");

  for (const auto& arg : args) {
    if (arg_name == arg.name()) {
      return true;
    }
  }
  return false;
}

}  // namespace

Status ValidateKernelRegistrations(const OpRegistryInterface& op_registry) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_93(mht_93_v, 2116, "", "./tensorflow/core/framework/op_kernel.cc", "ValidateKernelRegistrations");

  auto typed_registry = GlobalKernelRegistryTyped();
  tf_shared_lock lock(typed_registry->mu);
  for (const auto& key_registration : typed_registry->registry) {
    const KernelDef& kernel_def(key_registration.second.def);
    const OpRegistrationData* op_reg_data;
    const Status status = op_registry.LookUp(kernel_def.op(), &op_reg_data);
    if (!status.ok()) {
      // TODO(josh11b): Make this a hard error.
      LOG(ERROR) << "OpKernel ('" << kernel_def.ShortDebugString()
                 << "') for unknown op: " << kernel_def.op();
      continue;
    }
    const OpDef& op_def = op_reg_data->op_def;
    for (const auto& host_memory_arg : kernel_def.host_memory_arg()) {
      if (!FindArgInOp(host_memory_arg, op_def.input_arg()) &&
          !FindArgInOp(host_memory_arg, op_def.output_arg())) {
        return errors::InvalidArgument(
            "HostMemory arg '", host_memory_arg,
            "' not found in OpDef: ", SummarizeOpDef(op_def));
      }
    }
  }
  return Status::OK();
}

template <>
const Eigen::ThreadPoolDevice& OpKernelContext::eigen_device() const {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_94(mht_94_v, 2146, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::eigen_device");

  return eigen_cpu_device();
}

template <>
const Eigen::GpuDevice& OpKernelContext::eigen_device() const {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_95(mht_95_v, 2154, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::eigen_device");

  return eigen_gpu_device();
}

void OpKernelConstruction::CtxFailure(const Status& s) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_96(mht_96_v, 2161, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelConstruction::CtxFailure");

  VLOG(1) << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailureWithWarning(const Status& s) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_97(mht_97_v, 2169, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelConstruction::CtxFailureWithWarning");

  LOG(WARNING) << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailure(const char* file, int line,
                                      const Status& s) {
   std::vector<std::string> mht_98_v;
   mht_98_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_98(mht_98_v, 2179, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelConstruction::CtxFailure");

  VLOG(1) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
          << " : " << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
                                                 const Status& s) {
   std::vector<std::string> mht_99_v;
   mht_99_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_99(mht_99_v, 2190, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelConstruction::CtxFailureWithWarning");

  LOG(WARNING) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
               << " : " << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailure(const Status& s) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_100(mht_100_v, 2199, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::CtxFailure");

  VLOG(1) << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailureWithWarning(const Status& s) {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_101(mht_101_v, 2207, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::CtxFailureWithWarning");

  LOG(WARNING) << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailure(const char* file, int line, const Status& s) {
   std::vector<std::string> mht_102_v;
   mht_102_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_102(mht_102_v, 2216, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::CtxFailure");

  VLOG(1) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
          << " : " << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailureWithWarning(const char* file, int line,
                                            const Status& s) {
   std::vector<std::string> mht_103_v;
   mht_103_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_103(mht_103_v, 2227, "", "./tensorflow/core/framework/op_kernel.cc", "OpKernelContext::CtxFailureWithWarning");

  LOG(WARNING) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
               << " : " << s;
  SetStatus(s);
}

void CheckNotInComputeAsync(OpKernelContext* ctx,
                            const char* correct_macro_name) {
   std::vector<std::string> mht_104_v;
   mht_104_v.push_back("correct_macro_name: \"" + (correct_macro_name == nullptr ? std::string("nullptr") : std::string((char*)correct_macro_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_kernelDTcc mht_104(mht_104_v, 2238, "", "./tensorflow/core/framework/op_kernel.cc", "CheckNotInComputeAsync");

  CHECK_EQ(nullptr, ctx->params_->op_kernel->AsAsync())
      << "Use " << correct_macro_name << " in AsyncOpKernel implementations.";
}

}  // namespace tensorflow
