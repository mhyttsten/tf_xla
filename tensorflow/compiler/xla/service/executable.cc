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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc() {
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

#include "tensorflow/compiler/xla/service/executable.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/stream_executor/device_description.h"

namespace xla {

ExecutionInput::~ExecutionInput() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/executable.cc", "ExecutionInput::~ExecutionInput");

  for (auto& index : unowned_indices_) {
    auto buffer = buffers_.mutable_element(index)->Release();
    if (buffer) {
      buffer->Release();
    }
  }
}

Status ExecutionInput::SetDynamicShape(Shape dynamic_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/service/executable.cc", "ExecutionInput::SetDynamicShape");

  const Shape& input_shape = shape();
  if (!ShapeUtil::DynamicShapeIsCompatible(input_shape, dynamic_shape)) {
    return tensorflow::errors::InvalidArgument(
        "Cannot set dynamic shape: ", input_shape.DebugString(), " vs. ",
        dynamic_shape.DebugString());
  }
  dynamic_shape_ = absl::make_unique<Shape>(std::move(dynamic_shape));
  return Status::OK();
}

void ExecutionInput::SetUnownedBuffer(const ShapeIndex& index,
                                      MaybeOwningDeviceMemory buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_2(mht_2_v, 231, "", "./tensorflow/compiler/xla/service/executable.cc", "ExecutionInput::SetUnownedBuffer");

  *buffers_.mutable_element(index) = std::move(buffer);
  unowned_indices_.insert(index);
}

StatusOr<ShapedBuffer> ExecutionInput::ToShapedBuffer(
    se::DeviceMemoryAllocator* allocator, int device_ordinal) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_3(mht_3_v, 240, "", "./tensorflow/compiler/xla/service/executable.cc", "ExecutionInput::ToShapedBuffer");

  const Shape& input_shape = shape();
  ShapedBuffer shaped_buffer(input_shape, device_ordinal);
  for (const auto& index_buffer : Buffers()) {
    const tensorflow::se::OwningDeviceMemory* mem =
        index_buffer.second.AsOwningDeviceMemory();
    if (mem != nullptr && (mem->allocator() != allocator ||
                           mem->device_ordinal() != device_ordinal)) {
      return tensorflow::errors::InvalidArgument(
          "Device buffer at index ", index_buffer.first.ToString(),
          " has mismatching allocator/device");
    }
    shaped_buffer.set_buffer(index_buffer.second.AsDeviceMemoryBase(),
                             index_buffer.first);
  }
  return std::move(shaped_buffer);
}

StatusOr<ScopedShapedBuffer> Executable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_4(mht_4_v, 264, "", "./tensorflow/compiler/xla/service/executable.cc", "Executable::ExecuteOnStream");

  StatusOr<ScopedShapedBuffer> result =
      ExecuteAsyncOnStream(run_options, arguments, hlo_execution_profile);
  Status blocking_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(blocking_status);
  return result;
}

static ExecutionInput MakeMaybeOwningDeviceMemoryTree(
    const ShapedBuffer& shaped_buffer) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_5(mht_5_v, 277, "", "./tensorflow/compiler/xla/service/executable.cc", "MakeMaybeOwningDeviceMemoryTree");

  ExecutionInput result(shaped_buffer.on_device_shape());
  shaped_buffer.buffers().ForEachElement(
      [&](const ShapeIndex& index, const se::DeviceMemoryBase& mem) {
        result.SetBuffer(index, MaybeOwningDeviceMemory(mem));
      });
  return result;
}

StatusOr<ScopedShapedBuffer> Executable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_6(mht_6_v, 292, "", "./tensorflow/compiler/xla/service/executable.cc", "Executable::ExecuteAsyncOnStream");

  std::vector<ExecutionInput> args;
  args.reserve(arguments.size());
  for (const ShapedBuffer* arg : arguments) {
    args.emplace_back(MakeMaybeOwningDeviceMemoryTree(*arg));
  }
  TF_ASSIGN_OR_RETURN(ExecutionOutput out,
                      ExecuteAsyncOnStream(run_options, std::move(args),
                                           hlo_execution_profile));
  return out.ConsumeResult();
}

StatusOr<ExecutionOutput> Executable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_7(mht_7_v, 310, "", "./tensorflow/compiler/xla/service/executable.cc", "Executable::ExecuteOnStream");

  StatusOr<ExecutionOutput> result = ExecuteAsyncOnStream(
      run_options, std::move(arguments), hlo_execution_profile);
  Status blocking_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(blocking_status);
  return result;
}

StatusOr<std::vector<ScopedShapedBuffer>> Executable::ExecuteOnStreams(
    absl::Span<const ServiceExecutableRunOptions> run_options,
    absl::Span<const absl::Span<const ShapedBuffer* const>> arguments) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_8(mht_8_v, 324, "", "./tensorflow/compiler/xla/service/executable.cc", "Executable::ExecuteOnStreams");

  TF_RET_CHECK(run_options.size() == arguments.size());

  std::vector<ScopedShapedBuffer> return_values;
  return_values.reserve(run_options.size());

  if (run_options.size() == 1) {
    TF_ASSIGN_OR_RETURN(auto rv,
                        ExecuteOnStream(&run_options[0], arguments[0],
                                        /*hlo_execution_profile=*/nullptr));
    return_values.push_back(std::move(rv));
    return std::move(return_values);
  }

  for (size_t i = 0; i < run_options.size(); ++i) {
    // We cannot BlockHostUntilDone() on the already-launched executions in case
    // of error, since if the executions communicate, the initially launched
    // executions may never complete if not all executions are running.
    TF_ASSIGN_OR_RETURN(
        auto rv, ExecuteAsyncOnStream(&run_options[i], arguments[i],
                                      /*hlo_execution_profile=*/nullptr));
    return_values.push_back(std::move(rv));
  }
  for (const auto& options : run_options) {
    TF_RET_CHECK(options.stream() != nullptr);
    TF_RETURN_IF_ERROR(options.stream()->BlockHostUntilDone());
  }
  return std::move(return_values);
}

StatusOr<ScopedShapedBuffer> Executable::ExecuteOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_9(mht_9_v, 359, "", "./tensorflow/compiler/xla/service/executable.cc", "Executable::ExecuteOnStreamWrapper");

  StatusOr<ScopedShapedBuffer> result =
      ExecuteAsyncOnStreamWrapper(run_options, arguments);
  Status block_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(block_status);
  return result;
}

StatusOr<ExecutionOutput> Executable::ExecuteOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_10(mht_10_v, 373, "", "./tensorflow/compiler/xla/service/executable.cc", "Executable::ExecuteOnStreamWrapper");

  StatusOr<ExecutionOutput> result =
      ExecuteAsyncOnStreamWrapper(run_options, std::move(arguments));
  Status block_status = run_options->stream()->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(result.status());
  TF_RETURN_IF_ERROR(block_status);
  return result;
}

struct ExecuteAsyncOnStreamWrapperState {
  ExecutionProfile* profile;
  std::shared_ptr<se::Timer> timer;
  std::shared_ptr<HloExecutionProfile> profile_ptr;
};

static ExecuteAsyncOnStreamWrapperState ExecuteWrapperBeforeExecution(
    const Executable& executable,
    const ServiceExecutableRunOptions* run_options) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_11(mht_11_v, 393, "", "./tensorflow/compiler/xla/service/executable.cc", "ExecuteWrapperBeforeExecution");

  ExecuteAsyncOnStreamWrapperState state;
  se::Stream* stream = run_options->stream();
  state.profile = run_options->run_options().execution_profile();
  if (state.profile != nullptr) {
    state.timer = std::make_shared<se::Timer>(stream->parent());
    stream->InitTimer(state.timer.get()).ThenStartTimer(state.timer.get());
  }

  VLOG(1) << "enqueueing executable on stream...";
  // If the profiling flag isn't enabled, we pass nullptr as the profile to
  // indicate profiling is not requested.
  state.profile_ptr =
      executable.module_config().debug_options().xla_hlo_profile() &&
              executable.hlo_profiling_enabled()
          ? std::make_shared<HloExecutionProfile>(
                &executable.hlo_profile_printer_data(),
                &executable.hlo_profile_index_map())
          : nullptr;
  return state;
}

Status ExecuteWrapperAfterExecution(
    Executable* executable, const ExecuteAsyncOnStreamWrapperState& state,
    Status return_status, se::Stream* stream) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_12(mht_12_v, 420, "", "./tensorflow/compiler/xla/service/executable.cc", "ExecuteWrapperAfterExecution");

  if (!return_status.ok()) {
    if (state.profile != nullptr) {
      // Ensure the ThenStartTimer call has completed before we destroy timer.
      // We already have a failure status to return, so just log this if it
      // fails.
      Status status = stream->BlockHostUntilDone();
      if (!status.ok()) {
        LOG(ERROR) << "Failed to BlockHostUntilDone: " << status;
      }
    }
    return return_status;
  }

  if (state.profile != nullptr) {
    VLOG(1) << "enqueueing 'stop timer' and profiling callback...";
    stream->ThenStopTimer(state.timer.get());

    // We block instead of using an async callback because reading the timer
    // value may call back into the driver on GPU, which is not allowed.
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

    const int64_t executable_size_in_bytes =
        executable->SizeOfGeneratedCodeInBytes();
    // Merge in run-time profile information from execution_profile.

    // Overall execution time (in nanoseconds) from the executor timer.
    state.profile->set_compute_and_transfer_time_ns(state.timer->Nanoseconds());

    // TODO(b/28447609): The value in compute_and_transfer_time_ns is actually
    // the compute time without the transfer time, so this way we get the
    // correct compute time. We should instead have the correct value for
    // compute_and_transfer_time and set compute_time to the compute time.
    if (state.profile->compute_time_ns() == 0) {
      state.profile->set_compute_time_ns(
          state.profile->compute_and_transfer_time_ns());
    }

    if (executable_size_in_bytes != 0) {
      state.profile->set_executable_size_in_bytes(executable_size_in_bytes);
    }
  }

  if (executable->module_config().debug_options().xla_hlo_profile() &&
      state.profile_ptr != nullptr) {
    DumpToFileInDir(executable->module(), /*file_prefix=*/"",
                    /*file_suffix=*/"hlo_execution_profile_data",
                    state.profile_ptr->ToProto().SerializeAsString());
  }

  if (state.profile_ptr != nullptr) {
    const se::DeviceDescription* device_description =
        &stream->parent()->GetDeviceDescription();
    std::shared_ptr<HloExecutionProfile> profile = state.profile_ptr;
    stream->ThenDoHostCallback([profile, device_description]() {
      XLA_LOG_LINES(tensorflow::INFO,
                    profile->ToString(device_description->clock_rate_ghz()));
    });
  }

  return return_status;
}

StatusOr<ScopedShapedBuffer> Executable::ExecuteAsyncOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_13(mht_13_v, 488, "", "./tensorflow/compiler/xla/service/executable.cc", "Executable::ExecuteAsyncOnStreamWrapper");

  auto state = ExecuteWrapperBeforeExecution(*this, run_options);
  StatusOr<ScopedShapedBuffer> return_value =
      ExecuteAsyncOnStream(run_options, arguments, state.profile_ptr.get());
  TF_RETURN_IF_ERROR(ExecuteWrapperAfterExecution(
      this, state, return_value.status(), run_options->stream()));
  return return_value;
}

StatusOr<ExecutionOutput> Executable::ExecuteAsyncOnStreamWrapper(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_14(mht_14_v, 502, "", "./tensorflow/compiler/xla/service/executable.cc", "Executable::ExecuteAsyncOnStreamWrapper");

  auto state = ExecuteWrapperBeforeExecution(*this, run_options);
  StatusOr<ExecutionOutput> return_value = ExecuteAsyncOnStream(
      run_options, std::move(arguments), state.profile_ptr.get());
  TF_RETURN_IF_ERROR(ExecuteWrapperAfterExecution(
      this, state, return_value.status(), run_options->stream()));
  return return_value;
}

int64_t Executable::SizeOfGeneratedCodeInBytes() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_15(mht_15_v, 514, "", "./tensorflow/compiler/xla/service/executable.cc", "Executable::SizeOfGeneratedCodeInBytes");
 return -1; }

void Executable::MarkToBeReleasedArguments(absl::Span<ExecutionInput> arguments,
                                           ExecutionOutput& result) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTcc mht_16(mht_16_v, 520, "", "./tensorflow/compiler/xla/service/executable.cc", "Executable::MarkToBeReleasedArguments");

  for (ExecutionInput& argument : arguments) {
    for (auto& index_buffer : *argument.MutableBuffers()) {
      if (absl::optional<se::OwningDeviceMemory> maybe_owning_buffer =
              index_buffer.second.Release()) {
        result.AddToBeReleased(std::move(*maybe_owning_buffer));
      }
    }
  }
}

}  // namespace xla
