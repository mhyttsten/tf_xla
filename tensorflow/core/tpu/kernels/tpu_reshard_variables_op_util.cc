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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc() {
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

#include "tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.h"

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
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/tpu/tpu_execute.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tpu/tpu_executor.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_node_context.h"

namespace tensorflow {
namespace tpu {
namespace reshard_variables {

Status FlushProgramMemory(se::Platform* platform, int device_ordinal) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.cc", "FlushProgramMemory");

  TF_ASSIGN_OR_RETURN(std::unique_ptr<tpu::TpuNodeContext> node_interfaces,
                      tpu::TpuNodeContext::Create(device_ordinal));

  auto* executor = tensorflow::down_cast<tpu::TpuExecutorInterface*>(
      node_interfaces->stream_executor()->implementation());
  return executor->UnloadAllPrograms();
}

Status CheckIsValidKey(const Tensor& key) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc mht_1(mht_1_v, 231, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.cc", "CheckIsValidKey");

  if (!TensorShapeUtils::IsVector(key.shape()) ||
      key.shape().dim_size(0) != 3) {
    return errors::InvalidArgument(
        "new_format_key argument to TPUReshardVariables  must be a 3-element "
        "vector");
  }
  if (key.dtype() != DT_STRING) {
    return errors::InvalidArgument(
        "new_format_key argument to TPUReshardVariables must be DT_STRING "
        "type");
  }
  return Status::OK();
}

bool IsDefaultKey(const Tensor& key) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.cc", "IsDefaultKey");
 return key.vec<tstring>()(0).empty(); }

// Looks up the input `key` in the compilation cache, populating
// `*rendezvous_key_base` and `*entry`.
Status GetComputationCacheEntry(
    const Tensor& key, string* rendezvous_key_base,
    std::unique_ptr<tpu::CompilationCacheEntryRef>* entry,
    tpu::CompilationCacheFetchTarget fetch_target) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc mht_3(mht_3_v, 259, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.cc", "GetComputationCacheEntry");

  profiler::TraceMe trace_me("TPUReshardVariablesOpKernel::LookupProto",
                             /*level=*/2);
  TF_RETURN_IF_ERROR(CheckIsValidKey(key));
  auto* rmgr = GetTPUConfigResourceMgr();
  tpu::TpuCompilationCacheLookup* proto_lookup;
  TF_RETURN_IF_ERROR(rmgr->Lookup(rmgr->default_container(),
                                  tpu::kCompiledProtoCacheResourceName,
                                  &proto_lookup));
  core::ScopedUnref lookup_unref(proto_lookup);
  TF_RETURN_IF_ERROR(
      proto_lookup->Lookup(key.vec<tstring>()(0), entry, fetch_target));
  *rendezvous_key_base = key.vec<tstring>()(1);
  return Status::OK();
}

// Builds an InputBuffers object that describes the inputs to the computation.
xla::StatusOr<xla::ShapeTree<xla::MaybeOwningDeviceMemory>> BuildInputBuffers(
    OpKernelContext* context, const std::vector<VariableInfo>& variables,
    const xla::Shape& input_host_shape, xla::Backend* backend,
    int device_ordinal, se::Stream* stream) {
  profiler::TraceMe trace_me("BuildComputationInputs", /*level=*/2);
  OpInputList var_list;
  TF_RETURN_IF_ERROR(context->input_list("vars", &var_list));

  if (var_list.size() != xla::ShapeUtil::TupleElementCount(input_host_shape)) {
    return errors::InvalidArgument(
        "Number of variables (", var_list.size(),
        ") does not match input shape: ",
        xla::ShapeUtil::TupleElementCount(input_host_shape));
  }

  auto validate_shape = [&](int i, const Tensor& tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc mht_4(mht_4_v, 294, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.cc", "lambda");

    const xla::Shape& expected =
        xla::ShapeUtil::GetTupleElementShape(input_host_shape, i);
    VLOG(4) << "Input " << i << " TF shape " << tensor.shape().DebugString();
    XlaTensor* xla_tensor = XlaTensor::FromTensor(&tensor);

    if (xla_tensor == nullptr) {
      // FromTensor failed; tensor must be empty.
      if (!xla::ShapeUtil::IsZeroElementArray(expected)) {
        return errors::InvalidArgument(
            "Run-time shape mismatch for TPUExecute argument[", i, "] (",
            context->op_kernel().requested_input(i), "). Expected ",
            expected.DebugString(),
            "; got empty tensor. If you are running "
            "with TF2 TPU, make sure you set `drop_remainder=False` when "
            "calling `dataset.batch` on the `tf.data.Dataset` so dynamic batch "
            "size can be handled");
      }
    } else {
      const xla::Shape& xla_shape = xla_tensor->shaped_buffer().on_host_shape();
      if (!xla::ShapeUtil::Compatible(expected, xla_shape)) {
        return errors::InvalidArgument(
            "Run-time shape mismatch for TPUReshardVariables argument[", i,
            "] (", context->op_kernel().requested_input(i), "). Expected ",
            expected.DebugString(), "; got ", xla_shape.DebugString());
      }
    }

    return Status::OK();
  };

  for (int i = 0; i < variables.size(); ++i) {
    TF_RETURN_IF_ERROR(
        validate_shape(variables[i].index(), *variables[i].var()->tensor()));
  }

  se::DeviceMemoryAllocator* const allocator = backend->memory_allocator();
  xla::TransferManager* const transfer_manager = backend->transfer_manager();

  xla::ShapeTree<xla::MaybeOwningDeviceMemory> input_buffers(
      transfer_manager->HostShapeToDeviceShape(input_host_shape));

  // Allocates a buffer for the root tuple.
  const int64_t root_size =
      transfer_manager->GetByteSizeRequirement(input_buffers.shape());
  TF_ASSIGN_OR_RETURN(*input_buffers.mutable_element({}),
                      allocator->Allocate(device_ordinal, root_size));

  auto set_input_buffers_helper = [&](int arg_index, xla::ShapedBuffer* buffers,
                                      bool owning = false) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc mht_5(mht_5_v, 346, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.cc", "lambda");

    buffers->buffers().ForEachMutableElement(
        [&](const xla::ShapeIndex& index, se::DeviceMemoryBase* buffer) {
          xla::ShapeIndex in_index = {arg_index};
          for (int64_t j : index) {
            in_index.push_back(j);
          }
          if (owning) {
            *input_buffers.mutable_element(in_index) =
                se::OwningDeviceMemory(*buffer, device_ordinal, allocator);
            *buffer = se::DeviceMemoryBase();
          } else {
            *input_buffers.mutable_element(in_index) = *buffer;
          }
        });
  };

  // Assigns the buffers of 'tensor' as computation input 'i'. Allocates fresh
  // buffers for zero-element tensors where required.
  auto assign_input = [&](int i, const Tensor& tensor) -> xla::Status {
    XlaTensor* xla_tensor = XlaTensor::FromTensor(&tensor);

    // Size 0 tensors have no backing XlaTensor, but may still need to have
    // tuple buffers allocated.
    if (xla_tensor == nullptr) {
      CHECK_EQ(tensor.NumElements(), 0);
      const xla::Shape& host_shape =
          xla::ShapeUtil::GetSubshape(input_host_shape, {i});
      TF_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer buffers,
                          transfer_manager->AllocateScopedShapedBuffer(
                              host_shape, allocator, device_ordinal));
      set_input_buffers_helper(/*arg_index=*/i, &buffers);
    } else {
      set_input_buffers_helper(/*arg_index=*/i, &xla_tensor->shaped_buffer(),
                               tensor.RefCountIsOne());
      xla_tensor->WaitForDefinitionEventOnStream(stream);
    }
    return Status::OK();
  };

  for (int i = 0; i < var_list.size(); ++i) {
    TF_RET_CHECK(var_list[i].dtype() == DT_RESOURCE);
    TF_RETURN_IF_ERROR(assign_input(i, *variables[i].var()->tensor()));
  }

  return std::move(input_buffers);
}

// Perform a compaction to reduce fragmentation.
Status PerformCompaction(stream_executor::Stream* stream) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc mht_6(mht_6_v, 398, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.cc", "PerformCompaction");

  profiler::TraceMe trace_me("PerformCompaction", /*level=*/2);
  auto* ds_executor =
      down_cast<tpu::TpuExecutorInterface*>(stream->parent()->implementation());
  TF_RETURN_IF_ERROR(ds_executor->EnqueueCompactionOnStreamForHbm(stream));
  // LoadProgram and GetOrCreateConstantHandle are not managed by stream
  // dependencies but they write to shared memory, so we need to block here to
  // prevent those operations from racing.
  return stream->BlockHostUntilDone();
}

// Updates the variables to the execution result's buffers, and deallocates the
// root tuple buffer.
Status UpdateOutputVariables(
    OpKernelContext* context, xla::ScopedShapedBuffer result_buffers,
    absl::Span<const TensorShapeProto* const> output_tensor_shape_protos,
    xla::Backend* backend, se::Stream* stream, int device_ordinal,
    const std::vector<VariableInfo>& variables,
    const std::shared_ptr<se::Event>& definition_event) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc mht_7(mht_7_v, 419, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.cc", "UpdateOutputVariables");

  profiler::TraceMe trace_me("UpdateOutputVariables", /*level=*/2);
  // Shapes of the outputs, in TensorShape form.
  const int64_t sub_elements =
      xla::ShapeUtil::TupleElementCount(result_buffers.on_host_shape());
  if (sub_elements != output_tensor_shape_protos.size()) {
    return errors::InvalidArgument(
        "Mismatched numbers of output shapes: ", sub_elements, " vs. ",
        output_tensor_shape_protos.size());
  }

  if (sub_elements != variables.size()) {
    return errors::InvalidArgument(
        "Output count does not equal varaible count: ", sub_elements, " vs. ",
        variables.size());
  }

  std::vector<TensorShape> output_tensor_shapes;
  output_tensor_shapes.reserve(sub_elements);
  for (int64_t i = 0; i < sub_elements; ++i) {
    TF_RETURN_IF_ERROR(
        TensorShape::IsValidShape(*output_tensor_shape_protos[i]));
    TensorShape shape(*output_tensor_shape_protos[i]);
    const xla::Shape& xla_shape =
        xla::ShapeUtil::GetSubshape(result_buffers.on_host_shape(), {i});
    if (!xla_shape.IsArray() ||
        xla::ShapeUtil::ElementsIn(xla_shape) != shape.num_elements()) {
      return errors::InvalidArgument(
          "Mismatched number of elements in output shape: ",
          xla::ShapeUtil::HumanString(xla_shape), " vs ", shape.DebugString());
    }
    output_tensor_shapes.push_back(shape);
    VLOG(2) << "Output " << i << " shape " << shape.DebugString();
  }

  // Build a shaped buffer for the outputs.
  TF_RET_CHECK(result_buffers.on_host_shape().IsTuple());
  TF_RET_CHECK(!xla::ShapeUtil::IsNestedTuple(result_buffers.on_host_shape()));

  se::DeviceMemoryAllocator* const allocator = backend->memory_allocator();

  auto output_buffers = result_buffers.release();
  const xla::Shape& output_host_shape = output_buffers.on_host_shape();
  const xla::Shape& output_device_shape = output_buffers.on_device_shape();

  // Transfers ownership of the buffers that back XLA computation output 'i'
  // to 'output_tensor'.
  auto transfer_buffers = [&](int i, Tensor* output_tensor) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_reshard_variables_op_utilDTcc mht_8(mht_8_v, 469, "", "./tensorflow/core/tpu/kernels/tpu_reshard_variables_op_util.cc", "lambda");

    const xla::Shape& host_shape =
        xla::ShapeUtil::GetTupleElementShape(output_host_shape, i);
    const xla::Shape& device_shape =
        xla::ShapeUtil::GetTupleElementShape(output_device_shape, i);
    if (output_tensor->NumElements() > 0) {
      xla::ScopedShapedBuffer shaped_buffer(host_shape, device_shape, allocator,
                                            device_ordinal);
      shaped_buffer.buffers().ForEachMutableElement(
          [&](const xla::ShapeIndex& index, se::DeviceMemoryBase* buffer) {
            xla::ShapeIndex out_index = {i};
            for (int64_t j : index) {
              out_index.push_back(j);
            }
            *buffer = output_buffers.buffers().element(out_index);
          });

      XlaTensor* xla_tensor = XlaTensor::FromTensor(output_tensor);
      xla_tensor->set_shaped_buffer(std::move(shaped_buffer));
      xla_tensor->ResetDefinitionEvent(definition_event, stream);
    }
  };

  for (int i = 0; i < variables.size(); ++i) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        variables[i].var()->tensor()->dtype(), output_tensor_shapes[i],
        variables[i].var()->tensor()));
    transfer_buffers(i, variables[i].var()->tensor());
  }
  return allocator->Deallocate(output_buffers.device_ordinal(),
                               output_buffers.buffer({}));
}

}  // namespace reshard_variables
}  // namespace tpu
}  // namespace tensorflow
