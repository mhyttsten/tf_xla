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
class MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc() {
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

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_memory_manager.h"
#include "tensorflow/compiler/xrt/xrt_metrics.h"
#include "tensorflow/compiler/xrt/xrt_state.h"
#include "tensorflow/compiler/xrt/xrt_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/timed.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/tpu/tpu_execute.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace tensorflow {
namespace {

using tensorflow::tpu::CompilationCacheEntryRef;
using tensorflow::tpu::TpuCompilationCacheEntry;
using tensorflow::tpu::TpuCompilationCacheLookup;
using GetBufferFunction =
    std::function<xla::StatusOr<std::vector<xla::ExecutionInput>>()>;

// Looks up the input `key` in the compilation cache.
Status GetComputationCacheEntry(
    ResourceMgr* rm, int64_t key, int core_index_in_replica,
    std::unique_ptr<CompilationCacheEntryRef>* entry) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc mht_0(mht_0_v, 232, "", "./tensorflow/compiler/xrt/kernels/tpu_execute_op.cc", "GetComputationCacheEntry");

  profiler::TraceMe trace_me("XRTExecuteOp::LookupProto", /*level=*/2);
  TpuCompilationCacheLookup* proto_lookup;
  TF_RETURN_IF_ERROR(rm->Lookup(rm->default_container(),
                                tpu::kCompiledProtoCacheResourceName,
                                &proto_lookup));
  core::ScopedUnref lookup_unref(proto_lookup);
  TF_RETURN_IF_ERROR(proto_lookup->Lookup(key, core_index_in_replica, entry));
  return Status::OK();
}

std::vector<bool> GetDynamicInputInfo(
    const TPUExecutableInfoProto& executable_proto) {
  std::vector<bool> input_is_dynamic;
  input_is_dynamic.reserve(executable_proto.input_shapes().size());
  for (int64_t i = 0; i < executable_proto.input_shapes().size(); ++i) {
    input_is_dynamic.push_back(
        !xla::Shape(executable_proto.input_shapes(i)).is_static());
  }
  return input_is_dynamic;
}

xla::StatusOr<std::vector<RefPtr<XRTTupleAllocation>>> GetChainedOpInputs(
    const xrt::XRTChainedExecuteOp& op,
    absl::Span<const RefPtr<XRTTupleAllocation>> op_inputs,
    const TPUExecutableInfoProto& executable_proto) {
  if (op.inputs_size() != executable_proto.input_shapes_size()) {
    return errors::InvalidArgument(
        "Number of inputs does not match executable proto input shapes: ",
        op.inputs_size(), " vs. ", executable_proto.input_shapes_size());
  }

  std::vector<RefPtr<XRTTupleAllocation>> input_tuples;
  input_tuples.reserve(op.inputs_size());
  for (int i = 0; i < op.inputs_size(); ++i) {
    auto& input = op.inputs(i);
    const RefPtr<XRTTupleAllocation>& tuple = op_inputs[i];
    // Thanks to the greatness of proto3, there is no way to query for
    // explicitly set fields, so the default for output_index (zero) means no
    // sub-index. As consequence, the real index is output_index - 1.
    if (input.output_index() == 0) {
      input_tuples.push_back(tuple);
    } else {
      XRTTupleAllocation* sub_tuple;
      TF_RETURN_IF_ERROR(XRTTupleAllocation::MakeSubBuffer(
          tuple.get(), {input.output_index() - 1}, &sub_tuple,
          /*alias_parent_allocation=*/true));
      input_tuples.emplace_back(sub_tuple);
    }
    if (!InputShapeMatches(xla::Shape(executable_proto.input_shapes(i)),
                           input_tuples.back()->on_host_shape())) {
      return errors::InvalidArgument(
          "Run-time shape mismatch for XRTExecute argument[", i, "] (",
          op.computation_handle(), "). Expected ",
          executable_proto.input_shapes(i).DebugString(), "; got ",
          tuple->on_host_shape().DebugString());
    }
  }
  return std::move(input_tuples);
}

xla::StatusOr<xla::HloInputOutputAliasConfig> GetExecutableAliasConfig(
    const tpu::TpuProgramGroup* tpu_program_group, xla::Backend* const backend,
    int core_index) {
  const TPUExecutableInfoProto& executable =
      tpu_program_group->executable_info(core_index);
  return xla::HloInputOutputAliasConfig::CreateFromProto(
      backend->transfer_manager()->HostShapeToDeviceShape(
          xla::Shape(executable.output_shape())),
      tpu_program_group->hlo_metadata(core_index)
          ->hlo_module()
          .input_output_alias());
}

xla::StatusOr<RefPtr<XRTTupleAllocation>> AllocateOutputTuple(
    tpu::TpuNodeContext* node_context, se::Stream* stream,
    absl::Span<const RefPtr<XRTTupleAllocation>> input_tuples,
    const xla::HloInputOutputAliasConfig& input_output_alias,
    xla::ScopedShapedBuffer output_scoped_buffer, int device_ordinal) {
  auto output_shaped_buffer = output_scoped_buffer.release();

  xla::Shape output_device_shape = output_shaped_buffer.on_device_shape();
  if (!output_device_shape.is_static()) {
    TF_RETURN_IF_ERROR(
        node_context->backend()->transfer_manager()->ReadDynamicShapes(
            stream, &output_shaped_buffer, &output_device_shape));
  }

  XRTTupleAllocation* output_tuple;
  xla::Shape output_host_shape =
      xla::ShapeUtil::DeviceShapeToHostShape(output_device_shape);

  TF_RETURN_IF_ERROR(XRTTupleAllocation::CreateFromBuffer(
      output_shaped_buffer, output_host_shape, output_device_shape,
      node_context->backend(), device_ordinal, &output_tuple,
      node_context->backend()->memory_allocator()));
  RefPtr<XRTTupleAllocation> output_tuple_ptr(output_tuple);

  // If the input tuples had to release some buffers in order to provide the
  // proper temporary ownership transfer, we patch the holes here by alising the
  // buffers from the result tuple. The device address we patch back here, will
  // essentially be the same one we carved out in the DoWork() function.
  TF_RETURN_IF_ERROR(
      RebuildOutputAliases(output_tuple_ptr, input_tuples, input_output_alias));

  return std::move(output_tuple_ptr);
}

Status AllocateOutputTensors(
    OpKernelContext* context, XRTMemoryManager* memory_manager,
    tpu::TpuNodeContext* node_context, se::Stream* stream,
    const xrt::XRTExecutionConfig& config_proto,
    const TPUExecutableInfoProto& executable_proto,
    absl::Span<const RefPtr<XRTTupleAllocation>> input_tuples,
    const xla::HloInputOutputAliasConfig& input_output_alias,
    xla::ScopedShapedBuffer output_scoped_buffer, int device_ordinal) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc mht_1(mht_1_v, 350, "", "./tensorflow/compiler/xrt/kernels/tpu_execute_op.cc", "AllocateOutputTensors");

  TF_ASSIGN_OR_RETURN(
      RefPtr<XRTTupleAllocation> output_tuple,
      AllocateOutputTuple(node_context, stream, input_tuples,
                          input_output_alias, std::move(output_scoped_buffer),
                          device_ordinal));
  return CreateExecuteOutput(context, memory_manager, std::move(output_tuple),
                             config_proto.return_exploded_tuple());
}

xla::StatusOr<xla::ExecutionOutput> RunExecutable(
    OpKernelContext* context, tpu::TpuNodeContext* node_context,
    const TPUExecutableInfoProto& executable,
    std::vector<xla::ExecutionInput> arguments, const string& execution_id,
    const uint32 rng_seed, const tpu::TpuProgramGroup* tpu_program_group,
    xla::Backend* const backend, se::Stream* stream, int core_index,
    int device_ordinal, string rendezvous_key_base) {
  profiler::TraceMe trace_me("RunExecutable", /*level=*/2);

  // se::StreamExecutor* executor = node->stream_executor();

  std::unique_ptr<xla::DeviceAssignment> device_assignment;
  if (executable.has_device_assignment()) {
    TF_ASSIGN_OR_RETURN(device_assignment, xla::DeviceAssignment::Deserialize(
                                               executable.device_assignment()));
  }
  // Ideally this should be the host-to-device stream from XlaDeviceContext.
  // The particular anti-dependency this is avoiding (why we need a separate
  // transfer stream) is between the executable writing tuple tables and
  // TPUExecute()'s deregister_stream; if they come from the same stream pool
  // antidependencies will occur. XlaBackend has a different pool of streams
  // to the stream->GetOrCreateSubStream() that TPUExecute() uses, so these
  // will never refer to the same stream.
  TF_ASSIGN_OR_RETURN(auto transfer_stream_ptr,
                      backend->BorrowStream(device_ordinal));
  const TPUHostTransferInfoProto& host_transfer_info =
      tpu_program_group->host_transfer_info(core_index);
  TF_ASSIGN_OR_RETURN(
      xla::ExecutionOutput output,
      TPUExecute(executable, host_transfer_info,
                 *tpu_program_group->hlo_metadata(core_index),
                 std::move(arguments), rendezvous_key_base, rng_seed,
                 node_context, device_assignment.get(),
                 context->cancellation_manager(), context, stream,
                 transfer_stream_ptr.get(),
                 tpu_program_group->tpu_program(core_index)));

  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  return output;
}

xla::StatusOr<xla::ExecutionOutput> ExecuteTPUProgram(
    OpKernelContext* context, tpu::TpuNodeContext* node_context,
    XRTMemoryManager* memory_manager, const TPUExecutableInfoProto& executable,
    const GetBufferFunction& get_buffers_fn, const string& execution_id,
    const uint32 rng_seed, const tpu::TpuProgramGroup* tpu_program_group,
    xla::Backend* const backend, se::Stream* stream, int core_index,
    int device_ordinal, string rendezvous_key_base) {
  auto runfn = [&]() -> xla::StatusOr<xla::ExecutionOutput> {
    TF_ASSIGN_OR_RETURN(auto arguments, get_buffers_fn());
    return RunExecutable(context, node_context, executable,
                         std::move(arguments), execution_id, rng_seed,
                         tpu_program_group, backend, stream, core_index,
                         device_ordinal, rendezvous_key_base);
  };
  return memory_manager->Run<xla::ExecutionOutput>(
      runfn, backend, device_ordinal, /*requested_free_size=*/0,
      backend->memory_allocator());
}

// XRTExecuteOp

class XRTExecuteOp : public AsyncOpKernel {
 public:
  explicit XRTExecuteOp(OpKernelConstruction* context);

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override;

 private:
  Status DoWork(OpKernelContext* context);
};

XRTExecuteOp::XRTExecuteOp(OpKernelConstruction* context)
    : AsyncOpKernel(context, /* is_deferred = */ true) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc mht_2(mht_2_v, 437, "", "./tensorflow/compiler/xrt/kernels/tpu_execute_op.cc", "XRTExecuteOp::XRTExecuteOp");
}

void XRTExecuteOp::ComputeAsync(OpKernelContext* context, DoneCallback done) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc mht_3(mht_3_v, 442, "", "./tensorflow/compiler/xrt/kernels/tpu_execute_op.cc", "XRTExecuteOp::ComputeAsync");

  // Schedule onto the default queue, for unbounded concurrency. See b/73520706
  OP_REQUIRES_OK_ASYNC(context, DoWork(context), done);
  done();
}

Status XRTExecuteOp::DoWork(OpKernelContext* context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc mht_4(mht_4_v, 451, "", "./tensorflow/compiler/xrt/kernels/tpu_execute_op.cc", "XRTExecuteOp::DoWork");

  VLOG(1) << "XRTExecuteOp::Compute";

  const XlaDevice::Metadata* metadata;
  TF_RETURN_IF_ERROR(XlaDevice::GetMetadata(context, &metadata));
  const int device_ordinal = metadata->device_ordinal();
  // We are guaranteed that the object underlying TpuNodeContext won't be
  // deleted out from under us, while node_context is alive.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<tpu::TpuNodeContext> node_context,
                      tpu::TpuNodeContext::Create(device_ordinal));
  xla::Backend* const backend = node_context->backend();
  se::Stream* stream = context->op_device_context()->stream();

  auto timed = monitoring::MakeTimed(xrt_metrics::GetExecuteCell());
  profiler::TraceMe trace_me(
      [context] {
        return profiler::TraceMeEncode("TpuExecuteOp",
                                       {{"step_id", context->step_id()}});
      },
      /*level=*/2);
  profiler::TraceMe trace_me_init("XRTExecuteOp::Init", /*level=*/2);

  auto* rm = GetTPUConfigResourceMgr();
  TF_RET_CHECK(rm != nullptr);

  const Tensor& execution_input = context->input(0);
  TF_RET_CHECK(TensorShapeUtils::IsScalar(execution_input.shape()));
  int64_t compilation_handle = execution_input.scalar<int64_t>()();

  const Tensor& execution_config = context->input(1);
  TF_RET_CHECK(TensorShapeUtils::IsScalar(execution_config.shape()));
  xrt::XRTExecutionConfig config_proto;
  TF_RET_CHECK(
      config_proto.ParseFromString(execution_config.scalar<tstring>()()));

  int core_index_in_replica = config_proto.core_index_in_replica();
  bool release_inputs = config_proto.release_input_handles();
  bool release_compilation = config_proto.release_compilation_handle();

  string rendezvous_key_base = std::to_string(compilation_handle);
  std::unique_ptr<CompilationCacheEntryRef> entry;
  TF_RETURN_IF_ERROR(GetComputationCacheEntry(rm, compilation_handle,
                                              core_index_in_replica, &entry));

  TpuCompilationCacheEntry centry = entry->get();
  const tpu::TpuProgramGroup* tpu_program_group =
      tensorflow::down_cast<const tpu::TpuProgramGroup*>(
          centry.tpu_program_group());
  CHECK_NE(tpu_program_group, nullptr);

  if (release_compilation) {
    // Process-wide cache of Tpu executables.
    tpu::TpuCompilationCacheInterface* cache;
    TF_RETURN_IF_ERROR(rm->Lookup<tpu::TpuCompilationCacheInterface>(
        rm->default_container(), tpu::kCompilationCacheResourceName, &cache));
    core::ScopedUnref cache_unref(cache);
    TF_RETURN_IF_ERROR(cache->Release(compilation_handle));
    VLOG(2) << "Released compilation handle " << compilation_handle;
  }

  const int core_index = centry.core_index();
  const TPUExecutableInfoProto& executable =
      tpu_program_group->executable_info(core_index);

  std::vector<bool> input_is_dynamic = GetDynamicInputInfo(executable);

  TF_ASSIGN_OR_RETURN(
      xla::HloInputOutputAliasConfig input_output_alias,
      GetExecutableAliasConfig(tpu_program_group, backend, core_index));
  TF_ASSIGN_OR_RETURN(std::vector<InputCoords> input_coords,
                      GetComputationInputs(context, "input_handles"));

  RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
  XRTMemoryManager::WorkingSet working_set(memory_manager);
  TF_ASSIGN_OR_RETURN(
      std::vector<RefPtr<XRTTupleAllocation>> input_tuples,
      GetInputTupleAllocations(
          input_coords, &working_set, backend, executable.input_shapes_size(),
          [&](int64_t i) { return xla::Shape(executable.input_shapes(i)); },
          release_inputs, backend->memory_allocator()));
  auto get_buffers_fn = [&]() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc mht_5(mht_5_v, 534, "", "./tensorflow/compiler/xrt/kernels/tpu_execute_op.cc", "lambda");

    return GetArgumentsBuffers(input_output_alias, input_tuples,
                               input_is_dynamic, release_inputs);
  };
  trace_me_init.Stop();

  TF_ASSIGN_OR_RETURN(
      xla::ExecutionOutput output,
      ExecuteTPUProgram(
          context, node_context.get(), memory_manager.get(), executable,
          get_buffers_fn, config_proto.execution_instance_key(),
          config_proto.rng_seed(), tpu_program_group, backend, stream,
          core_index, device_ordinal, rendezvous_key_base));

  // AllocateComputationOutput writes the output tuple handle to the output
  // tensor return value from the Op.
  TF_RETURN_IF_ERROR(AllocateOutputTensors(
      context, memory_manager.get(), node_context.get(), stream, config_proto,
      executable, input_tuples, input_output_alias, output.ConsumeResult(),
      device_ordinal));
  return Status::OK();
}

class XRTExecuteChainedOp : public AsyncOpKernel {
 public:
  explicit XRTExecuteChainedOp(OpKernelConstruction* context);

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override;

 private:
  Status DoWork(OpKernelContext* context);
};

XRTExecuteChainedOp::XRTExecuteChainedOp(OpKernelConstruction* context)
    : AsyncOpKernel(context, /* is_deferred = */ true) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc mht_6(mht_6_v, 571, "", "./tensorflow/compiler/xrt/kernels/tpu_execute_op.cc", "XRTExecuteChainedOp::XRTExecuteChainedOp");
}

void XRTExecuteChainedOp::ComputeAsync(OpKernelContext* context,
                                       DoneCallback done) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc mht_7(mht_7_v, 577, "", "./tensorflow/compiler/xrt/kernels/tpu_execute_op.cc", "XRTExecuteChainedOp::ComputeAsync");

  // Schedule onto the default queue, for unbounded concurrency. See b/73520706
  OP_REQUIRES_OK_ASYNC(context, DoWork(context), done);
  done();
}

Status XRTExecuteChainedOp::DoWork(OpKernelContext* context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc mht_8(mht_8_v, 586, "", "./tensorflow/compiler/xrt/kernels/tpu_execute_op.cc", "XRTExecuteChainedOp::DoWork");

  VLOG(1) << "XRTExecuteChainedOp::Compute";
  const XlaDevice::Metadata* metadata;
  TF_RETURN_IF_ERROR(XlaDevice::GetMetadata(context, &metadata));
  const int device_ordinal = metadata->device_ordinal();
  // We are guaranteed that the object underlying TpuNodeContext won't be
  // deleted out from under us, while node_context is alive.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<tpu::TpuNodeContext> node_context,
                      tpu::TpuNodeContext::Create(device_ordinal));
  xla::Backend* const backend = node_context->backend();
  se::Stream* stream = context->op_device_context()->stream();
  auto timed = monitoring::MakeTimed(xrt_metrics::GetExecuteChainedCell());
  profiler::TraceMe trace_me(
      [context] {
        return profiler::TraceMeEncode("TpuExecuteChainedOp",
                                       {{"step_id", context->step_id()}});
      },
      /*level=*/2);
  ResourceMgr* rm = GetTPUConfigResourceMgr();
  TF_RET_CHECK(rm != nullptr);

  const Tensor& execution_plan = context->input(0);
  TF_RET_CHECK(TensorShapeUtils::IsScalar(execution_plan.shape()));
  xrt::XRTChainedExecutePlan plan;
  TF_RET_CHECK(plan.ParseFromString(execution_plan.scalar<tstring>()()));

  const Tensor& execution_config = context->input(1);
  TF_RET_CHECK(TensorShapeUtils::IsScalar(execution_config.shape()));
  xrt::XRTChainedExecuteConfig config;
  TF_RET_CHECK(config.ParseFromString(execution_config.scalar<tstring>()()));

  TpuCompilationCacheLookup* proto_lookup;
  TF_RETURN_IF_ERROR(rm->Lookup(rm->default_container(),
                                tpu::kCompiledProtoCacheResourceName,
                                &proto_lookup));
  core::ScopedUnref lookup_unref(proto_lookup);
  RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
  auto execute_op = [&](const xrt::XRTChainedExecuteOp& op,
                        absl::Span<const RefPtr<XRTTupleAllocation>> op_inputs)
      -> xla::StatusOr<RefPtr<XRTTupleAllocation>> {
    std::unique_ptr<CompilationCacheEntryRef> entry;
    TF_RETURN_IF_ERROR(proto_lookup->Lookup(
        op.computation_handle(), config.core_index_in_replica(), &entry));
    string rendezvous_key_base = std::to_string(op.computation_handle());
    TpuCompilationCacheEntry centry = entry->get();
    const tpu::TpuProgramGroup* tpu_program_group =
        tensorflow::down_cast<const tpu::TpuProgramGroup*>(
            centry.tpu_program_group());
    CHECK_NE(tpu_program_group, nullptr);
    const int core_index = centry.core_index();
    const TPUExecutableInfoProto& executable =
        tpu_program_group->executable_info(core_index);
    std::vector<bool> input_is_dynamic = GetDynamicInputInfo(executable);

    TF_ASSIGN_OR_RETURN(
        xla::HloInputOutputAliasConfig input_output_alias,
        GetExecutableAliasConfig(tpu_program_group, backend, core_index));
    TF_ASSIGN_OR_RETURN(std::vector<RefPtr<XRTTupleAllocation>> input_tuples,
                        GetChainedOpInputs(op, op_inputs, executable));
    auto get_buffers_fn = [&]() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSkernelsPStpu_execute_opDTcc mht_9(mht_9_v, 648, "", "./tensorflow/compiler/xrt/kernels/tpu_execute_op.cc", "lambda");

      return GetArgumentsBuffers(input_output_alias, input_tuples,
                                 input_is_dynamic,
                                 /*release_inputs=*/false);
    };
    TF_ASSIGN_OR_RETURN(
        xla::ExecutionOutput output,
        ExecuteTPUProgram(context, node_context.get(), memory_manager.get(),
                          executable, get_buffers_fn,
                          config.execution_instance_key(), config.rng_seed(),
                          tpu_program_group, backend, stream, core_index,
                          device_ordinal, rendezvous_key_base));
    return AllocateOutputTuple(node_context.get(), stream, input_tuples,
                               input_output_alias, output.ConsumeResult(),
                               device_ordinal);
  };

  return ExecuteChained(context, memory_manager, backend, device_ordinal, plan,
                        config, execute_op, backend->memory_allocator());
}

}  // namespace

REGISTER_KERNEL_BUILDER(Name("XRTExecute")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("computation_handle")
                            .HostMemory("execution_config")
                            .HostMemory("input_handles")
                            .HostMemory("output_handle"),
                        XRTExecuteOp);

REGISTER_KERNEL_BUILDER(Name("XRTExecuteChained")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("execution_plan")
                            .HostMemory("execution_config")
                            .HostMemory("output_handle"),
                        XRTExecuteChainedOp);

}  // namespace tensorflow
