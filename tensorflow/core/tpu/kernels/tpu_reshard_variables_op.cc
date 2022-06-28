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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_opDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_opDTcc() {
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

#include "tensorflow/core/tpu/kernels/tpu_reshard_variables_op.h"

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group.h"
#include "tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/tpu/tpu_execute.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tpu/tpu_executor.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_node_context.h"

namespace tensorflow {

namespace reshard_util = ::tensorflow::tpu::reshard_variables;

TPUReshardVariablesOpKernel::TPUReshardVariablesOpKernel(
    OpKernelConstruction* context)
    : AsyncOpKernel(context, /* is_deferred = */ true) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_opDTcc mht_0(mht_0_v, 222, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op.cc", "TPUReshardVariablesOpKernel::TPUReshardVariablesOpKernel");

  OP_REQUIRES_OK(context, context->GetAttr("N", &num_vars_));
}

void TPUReshardVariablesOpKernel::ComputeAsync(OpKernelContext* context,
                                               DoneCallback done) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_opDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op.cc", "TPUReshardVariablesOpKernel::ComputeAsync");

  // If TPU launches are asynchronous, then perform the launch on this thread
  // to avoid a thread hop, which has an observable latency cost.
  OP_REQUIRES_OK_ASYNC(context, DoWork(context), done);
  done();
}

Status TPUReshardVariablesOpKernel::DoWork(OpKernelContext* context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_opDTcc mht_2(mht_2_v, 240, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op.cc", "TPUReshardVariablesOpKernel::DoWork");

  VLOG(1) << "Cloud TPU: TPUReshardVariablesOpKernel::DoWork";
  TF_RET_CHECK(context->input_dtype(num_vars_) == DT_STRING);
  const Tensor* new_format_key;
  TF_RETURN_IF_ERROR(context->input("new_format_key", &new_format_key));
  TF_RETURN_IF_ERROR(reshard_util::CheckIsValidKey(*new_format_key));

  TF_RET_CHECK(context->input_dtype(num_vars_ + 1) == DT_RESOURCE);
  const ResourceHandle& handle = HandleFromInput(context, num_vars_ + 1);
  core::RefCountPtr<Var> format_state_var;
  TF_RETURN_IF_ERROR(LookupOrCreateResource<Var>(
      context, handle, &format_state_var, [new_format_key](Var** ptr) {
        *ptr = new Var(new_format_key->dtype());
        return Status::OK();
      }));
  mutex_lock ml(*format_state_var->mu());
  const bool initialized = format_state_var->is_initialized;
  if (initialized) {
    TF_RETURN_IF_ERROR(
        reshard_util::CheckIsValidKey(*format_state_var->tensor()));
  }

  const bool state_is_default =
      !initialized || reshard_util::IsDefaultKey(*format_state_var->tensor());
  const bool new_format_is_default =
      reshard_util::IsDefaultKey(*new_format_key);

  if ((state_is_default && new_format_is_default) ||
      (initialized && format_state_var->tensor()->vec<tstring>()(2) ==
                          new_format_key->vec<tstring>()(2))) {
    VLOG(1) << "Sharding unchanged, nothing to do.";
    return Status::OK();
  }

  if (!state_is_default) {
    // Convert the current format to default (unsharded).
    VLOG(1) << "Unsharding with key: "
            << format_state_var->tensor()->vec<tstring>()(2);
    TF_RETURN_IF_ERROR(
        DoTpuExecute(context, *format_state_var->tensor(),
                     tpu::CompilationCacheFetchTarget::UNSHARDING));
  }

  if (!new_format_is_default) {
    // Convert the new format.
    VLOG(1) << "Sharding with key: " << new_format_key->vec<tstring>()(2);
    TF_RETURN_IF_ERROR(DoTpuExecute(
        context, *new_format_key, tpu::CompilationCacheFetchTarget::SHARDING));
  }

  // Change the state.
  *format_state_var->tensor() = *new_format_key;
  format_state_var->is_initialized = true;
  return Status::OK();
}

Status TPUReshardVariablesOpKernel::DoTpuExecute(
    OpKernelContext* context, const Tensor& format_key,
    tpu::CompilationCacheFetchTarget fetch_target) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_opDTcc mht_3(mht_3_v, 301, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op.cc", "TPUReshardVariablesOpKernel::DoTpuExecute");

  const XlaDevice::Metadata* metadata;
  TF_RETURN_IF_ERROR(XlaDevice::GetMetadata(context, &metadata));
  const int device_ordinal = metadata->device_ordinal();

  // We are guaranteed that the underlying object won't be deleted out from
  // under us
  TF_ASSIGN_OR_RETURN(std::unique_ptr<tpu::TpuNodeContext> node_interfaces,
                      tpu::TpuNodeContext::Create(device_ordinal));

  profiler::TraceMe trace_me(
      [device_ordinal] {
        return profiler::TraceMeEncode("TPUReshardVariablesOpKernel",
                                       {{"device_ordinal", device_ordinal}});
      },
      /*level=*/2);
  profiler::TraceMe trace_me_init("TPUReshardVariablesOpKernel::Init",
                                  /*level=*/2);

  string rendezvous_key_base;
  std::unique_ptr<tpu::CompilationCacheEntryRef> entry_ref;
  TF_RETURN_IF_ERROR(reshard_util::GetComputationCacheEntry(
      format_key, &rendezvous_key_base, &entry_ref, fetch_target));
  tpu::TpuCompilationCacheEntry entry = entry_ref->get();
  if (entry.tpu_program_group() == nullptr) {
    VLOG(2) << "Sharding/unsharding program does not exist, so this is default "
               "sharding.";
    return Status::OK();
  }

  const tpu::TpuProgramGroupInterface* tpu_program_group =
      entry.tpu_program_group();
  const int core_index = entry.core_index();
  const TPUExecutableInfoProto& executable_info_proto =
      tpu_program_group->executable_info(core_index);
  const TPUExecutableInfoProto* executable = &executable_info_proto;

  xla::Backend* const backend = node_interfaces->backend();
  xla::TransferManager* const transfer_manager = backend->transfer_manager();

  CHECK(context->op_device_context());
  se::Stream* stream = context->op_device_context()->stream();

  TF_RET_CHECK(executable->input_shapes_size() == 1);
  xla::Shape host_shape(executable->input_shapes(0));
  std::vector<VariableInfo> variables;
  for (int i = 0; i < num_vars_; ++i) {
    TF_RET_CHECK(context->input_dtype(i) == DT_RESOURCE);
    const ResourceHandle& handle = HandleFromInput(context, i);
    Var* variable;
    TF_RETURN_IF_ERROR(LookupResource(context, handle, &variable));
    variables.push_back(VariableInfo(i, handle.name(), variable));
  }

  // Block for previous TPUExecute ops so that the memory used for them could be
  // freed.
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  // Lock variables to prevent concurrent access.
  TF_RETURN_IF_ERROR(LockVariables(absl::MakeSpan(variables)));

  // Build input buffers.
  TF_ASSIGN_OR_RETURN(auto input_buffers, reshard_util::BuildInputBuffers(
                                              context, variables, host_shape,
                                              backend, device_ordinal, stream));
  xla::ShapedBuffer shaped_buffer(std::move(host_shape), input_buffers.shape(),
                                  device_ordinal);
  shaped_buffer.set_buffers(input_buffers.Map<se::DeviceMemoryBase>(
      [](const xla::MaybeOwningDeviceMemory& buffer) {
        return buffer.AsDeviceMemoryBase();
      }));

  // Write input root tuple.
  TF_ASSIGN_OR_RETURN(auto transfer_stream_ptr,
                      backend->BorrowStream(device_ordinal));
  if (transfer_manager->CanShapedBufferBeAccessedNow(stream->parent(),
                                                     shaped_buffer)) {
    TF_RETURN_IF_ERROR(transfer_manager->WriteRootTupleIndexTable(
        transfer_stream_ptr.get(), shaped_buffer));
    stream->ThenWaitFor(transfer_stream_ptr.get());
  } else {
    TF_RETURN_IF_ERROR(
        transfer_manager->WriteRootTupleIndexTable(stream, shaped_buffer));
  }
  VLOG(4) << "Input buffers: " << shaped_buffer.ToString();

  TF_RET_CHECK(!executable->has_session_module())
      << "session module not supported in sharding/unsharding program.";

  auto definition_event = std::make_shared<se::Event>(stream->parent());
  TF_RET_CHECK(definition_event->Init())
      << "TPU definition event initialization failed";

  trace_me_init.Stop();

  // Execute the program.
  std::unique_ptr<xla::DeviceAssignment> device_assignment;
  if (executable->has_device_assignment()) {
    TF_ASSIGN_OR_RETURN(
        device_assignment,
        xla::DeviceAssignment::Deserialize(executable->device_assignment()));
  }
  std::vector<xla::ExecutionInput> input;
  input.emplace_back(xla::ExecutionInput(std::move(input_buffers),
                                         shaped_buffer.on_host_shape()));

  const TPUHostTransferInfoProto& host_transfer_info =
      tpu_program_group->host_transfer_info(core_index);

  TF_ASSIGN_OR_RETURN(
      xla::ExecutionOutput output,
      TPUExecute(*executable, host_transfer_info,
                 *tpu_program_group->hlo_metadatas()[core_index],
                 std::move(input), rendezvous_key_base, GetXLARandomSeed(),
                 node_interfaces.get(), device_assignment.get(),
                 context->cancellation_manager(), context, stream,
                 transfer_stream_ptr.get(),
                 tpu_program_group->tpu_program(core_index)));

  stream->ThenRecordEvent(definition_event.get());

  // Assign the new buffers to the variables.
  xla::ScopedShapedBuffer result = output.ConsumeResult();

  // Only perform compaction when sharding.
  // NOTE: Compaction is not supported on some TPUs, see b/168322060 for details
  if (node_interfaces->CompactionSupported(device_ordinal) &&
      fetch_target == tpu::CompilationCacheFetchTarget::SHARDING) {
    // Block until program execution is done so that input, output, and program
    // cache memory can be actually released.
    TF_RETURN_IF_ERROR(transfer_stream_ptr->BlockHostUntilDone());
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    {
      // Explicitly release any RAII objects owning on-device allocations.
      auto unused = output.ConsumeToBeReleased();
    }
    // Release variables holding inputs.
    for (int i = 0; i < variables.size(); ++i) {
      *variables[i].var()->tensor() =
          Tensor(variables[i].var()->tensor()->dtype());
    }
    // Flush on-device program memory cache.
    TF_RETURN_IF_ERROR(
        reshard_util::FlushProgramMemory(backend->platform(), device_ordinal));
    TF_RETURN_IF_ERROR(reshard_util::PerformCompaction(stream));
  }
  return reshard_util::UpdateOutputVariables(
      context, std::move(result), executable->output_tensor_shapes(), backend,
      stream, device_ordinal, variables, definition_event);
}

TPUReshardVariablesOpKernel::~TPUReshardVariablesOpKernel() = default;

#if !defined(PLATFORM_GOOGLE)
REGISTER_KERNEL_BUILDER(Name("TPUReshardVariables")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("format_state_var")
                            .HostMemory("new_format_key"),
                        TPUReshardVariablesOpKernel);
#endif

}  // namespace tensorflow
