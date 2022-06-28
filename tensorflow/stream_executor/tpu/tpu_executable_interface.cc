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
class MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executable_interfaceDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executable_interfaceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executable_interfaceDTcc() {
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

#include "tensorflow/stream_executor/tpu/tpu_executable_interface.h"

#include <utility>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

namespace {

// Write the tuple index buffers (arrays of pointers).
static Status PopulateResultTupleBuffers(const ShapedBuffer& result,
                                         se::Stream* stream,
                                         se::Stream* transfer_stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executable_interfaceDTcc mht_0(mht_0_v, 204, "", "./tensorflow/stream_executor/tpu/tpu_executable_interface.cc", "PopulateResultTupleBuffers");

  TF_ASSIGN_OR_RETURN(auto transfer_manager, TransferManager::GetForPlatform(
                                                 stream->parent()->platform()));
  if (transfer_manager->CanShapedBufferBeAccessedNow(stream->parent(),
                                                     result)) {
    TF_RETURN_IF_ERROR(transfer_manager->WriteTupleIndexTablesAsync(
        transfer_stream ? transfer_stream : stream, result));
    if (transfer_stream && transfer_stream != stream) {
      stream->ThenWaitFor(transfer_stream);
    }
    return Status::OK();
  } else {
    return transfer_manager->WriteTupleIndexTablesAsync(stream, result);
  }
}

}  // namespace

StatusOr<ExecutionOutput>
TpuExecutableInterface::AllocateOutputMemoryWithInputReuse(
    const Shape& shape, const HloInputOutputAliasConfig& alias_config,
    se::DeviceMemoryAllocator* allocator,
    std::vector<ExecutionInput>* arguments, se::Stream* stream,
    se::Stream* transfer_stream) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executable_interfaceDTcc mht_1(mht_1_v, 230, "", "./tensorflow/stream_executor/tpu/tpu_executable_interface.cc", "TpuExecutableInterface::AllocateOutputMemoryWithInputReuse");

  auto stream_exec = stream->parent();
  auto device_ordinal = stream_exec->device_ordinal();
  VLOG(3) << "AllocateOutputMemoryWithInputReuse, device = " << device_ordinal
          << " shape = " << ShapeUtil::HumanStringWithLayout(shape);
  auto update_layout = [this](xla::Shape* subshape,
                              const xla::ShapeIndex& index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executable_interfaceDTcc mht_2(mht_2_v, 239, "", "./tensorflow/stream_executor/tpu/tpu_executable_interface.cc", "lambda");

    if (subshape->IsArray()) {
      CHECK(subshape->has_layout());
      if (!subshape->layout().tiles().empty()) {
        // Already in device shape.
        return;
      }
      *subshape = HostShapeToDeviceShape(*subshape);
    }
  };
  Shape device_shape = shape;
  xla::ShapeUtil::ForEachMutableSubshape(&device_shape, update_layout);

  TF_RETURN_IF_ERROR(alias_config.ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          absl::optional<HloInputOutputAliasConfig::Alias> alias) {
        if (alias && alias->must_alias()) {
          VLOG(1) << alias->ToString();
          const MaybeOwningDeviceMemory& original_input =
              (*arguments)[alias->parameter_number].Buffers().element(
                  alias->parameter_index);
          if (!original_input.HasOwnership()) {
            return InvalidArgument(
                "An input was configured to be must-alias at "
                "compile time but not donated at runtime: %s",
                alias->ToString());
          }
        }
        return Status::OK();
      }));

  if (VLOG_IS_ON(3)) {
    VLOG(3) << "AllocateOutputMemoryWithInputReuse, device = " << device_ordinal
            << " shape = " << ShapeUtil::HumanStringWithLayout(shape);
    if (!Shape::Equal().MinorToMajorOnlyInLayout()(shape, device_shape)) {
      VLOG(3) << "Rewrote shape to device_shape: "
              << ShapeUtil::HumanStringWithLayout(shape) << " -> "
              << ShapeUtil::HumanStringWithLayout(device_shape);
    }
  }

  ExecutionOutput result(std::move(device_shape), allocator, device_ordinal);
  // Iterate through and allocate a buffer for each shape index, checking for
  // possible input buffer reuse.
  int64_t reused_buffer_bytes = 0;
  int64_t total_result_buffer_bytes = 0;
  for (auto& pair : result.MutableResult()->buffers()) {
    const ShapeIndex& result_index = pair.first;
    se::DeviceMemoryBase& result_buffer = pair.second;
    int64_t allocation_bytes = ShapeSize(ShapeUtil::GetSubshape(
        result.Result().on_device_shape(), result_index));
    total_result_buffer_bytes += allocation_bytes;

    // Return an InternalError if result_index is invalid. This avoids failing
    // the CHECK when calling GetAliasedParameter
    if (!ShapeUtil::IndexIsValid(alias_config.shape(), result_index)) {
      return InternalError("result_index is invalid: %s",
                           result_index.ToString());
    }

    absl::optional<HloInputOutputAliasConfig::Alias> alias =
        alias_config.GetAliasedParameter(result_index);
    if (alias) {
      TF_RET_CHECK(alias->parameter_number < arguments->size());
      ExecutionInput& input = (*arguments)[alias->parameter_number];
      MaybeOwningDeviceMemory* device_memory =
          input.MutableBuffer(alias->parameter_index);
      if (auto owning = device_memory->Release()) {
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceMemoryBase device_memory_base = owning->Release();
        *device_memory = device_memory_base;
        result_buffer = device_memory_base;
        reused_buffer_bytes += allocation_bytes;
        // The caller is giving us the input buffer, but in case of error of the
        // execute call, we should not be releasing it as it contains valid data
        // (for example, it is a parameter which the user wants us to alias, in
        // a gradient update computation). So we store the index into the result
        // in the aliased vactor, which will be fed to the ExecutionOutput,
        // which will be using the indices to drop the addresses from its own
        // ScopedShapedBuffer result, if the ExecutionOutput is not committed.
        result.AddAliasedIndex(result_index);
      } else {
        VLOG(2) << "An input was not reused since it is not donated "
                << alias->ToString();
      }
    }

    // We need to allocate a new output buffer for two cases:
    // - There is no alias between this output and any input.
    // - There is an alias, but the xla doesn't own the input memory so it can't
    // donate buffer to the computation.
    if (result_buffer.is_null()) {
      const Shape& on_device_shape = result.Result().on_device_shape();
      const Shape& on_device_subshape =
          ShapeUtil::GetSubshape(on_device_shape, result_index);
      TF_ASSIGN_OR_RETURN(
          auto allocated_buffer,
          allocator->Allocate(device_ordinal, allocation_bytes,
                              /*retry_on_failure=*/true,
                              on_device_subshape.layout().memory_space()));
      // Store the allocated buffer in our ScopedShapedBuffer, which takes
      // ownership.
      result_buffer = allocated_buffer.Release();
    }
    TF_RET_CHECK(allocation_bytes == 0 || result_buffer != nullptr);
  }

  VLOG(1) << "Reused " << reused_buffer_bytes
          << " parameter buffers (total result buffer size: "
          << total_result_buffer_bytes << ")";

  TF_RETURN_IF_ERROR(
      PopulateResultTupleBuffers(result.Result(), stream, transfer_stream));
  return std::move(result);
}

StatusOr<ExecutionOutput> TpuExecutableInterface::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* /*hlo_execution_profile*/) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executable_interfaceDTcc mht_3(mht_3_v, 364, "", "./tensorflow/stream_executor/tpu/tpu_executable_interface.cc", "TpuExecutableInterface::ExecuteAsyncOnStream");

  std::vector<se::DeviceMemoryBase> memory_bases;
  memory_bases.reserve(arguments.size());
  for (auto& argument : arguments) {
    memory_bases.push_back(argument.Buffer({}).AsDeviceMemoryBase());
  }
  se::Stream* stream = run_options->stream();

  CHECK_NE(run_options->allocator(), nullptr);
  const Shape& shape =
      hlo_module_ == nullptr ? ShapeUtil::MakeNil() : result_shape();
  const HloInputOutputAliasConfig& alias_config =
      hlo_module_ == nullptr ? HloInputOutputAliasConfig()
                             : hlo_module_->input_output_alias_config();
  TF_ASSIGN_OR_RETURN(
      ExecutionOutput result,
      AllocateOutputMemoryWithInputReuse(
          shape, alias_config, run_options->allocator(), &arguments, stream,
          run_options->run_options().host_to_device_stream()));

  // Address of the buffer in TPU memory that is being speculated.
  absl::optional<se::DeviceMemoryBase> cross_program_prefetch_addr;
  if (hlo_module_) {
    for (const auto& prefetch : hlo_module_->CrossProgramPrefetches()) {
      const auto& parameter = prefetch.first;
      const auto& index = prefetch.second;
      CHECK_LT(parameter, arguments.size());
      // Ensure the cross program prefetched buffer doesn't alias with any
      // program outputs. If the input and output aliased, the buffer could be
      // invalidated during program execution and the program could read stale
      // data from fast memory instead of fresh data in large memory.
      auto it = arguments[parameter].MutableBuffers()->find({index});
      CHECK(it != arguments[parameter].MutableBuffers()->end());
      CHECK(!it->second.AsDeviceMemoryBase().is_null());
      if (absl::c_none_of(result.Result().buffers(), [&](auto index_addr_pair) {
            return index_addr_pair.second.IsSameAs(
                it->second.AsDeviceMemoryBase());
          })) {
        // Supports only one cross-program prefetch address.
        cross_program_prefetch_addr = it->second.AsDeviceMemoryBase();
      }
    }
  }

  // MarkToBeReleasedArguments may std::move some elements of arguments, so it
  // must run after the cross program prefetch address is calculated from the
  // arguments.
  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);

  TF_RETURN_IF_ERROR(LoadProgramAndEnqueueToStream(
      *run_options, memory_bases, result.Result().root_buffer(),
      cross_program_prefetch_addr));
  return std::move(result);
}

}  // namespace xla
