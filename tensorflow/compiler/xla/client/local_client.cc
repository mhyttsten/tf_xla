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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc() {
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

#include "tensorflow/compiler/xla/client/local_client.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "llvm/ADT/Triple.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/source_map_util.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/compiler/xla/status_macros.h"

using xla::source_map_util::InvalidParameterArgument;

namespace xla {

namespace {
StatusOr<StreamPool::Ptr> BorrowStreamForDevice(int device_ordinal,
                                                Backend* backend) {
  if (device_ordinal < 0) {
    device_ordinal = backend->default_device_ordinal();
  }
  return backend->BorrowStream(device_ordinal);
}
}  // namespace

LocalExecutable::LocalExecutable(std::unique_ptr<Executable> executable,
                                 Backend* backend,
                                 ExecutableBuildOptions build_options)
    : executable_(std::move(executable)),
      backend_(backend),
      build_options_(std::move(build_options)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalExecutable::LocalExecutable");

  CHECK_GE(build_options_.device_ordinal(), 0)
      << "Must have a valid device ordinal that the executable was built for.";
}

Status LocalExecutable::ValidateExecutionOptions(
    const ExecutableRunOptions& run_options, const Backend& backend) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_1(mht_1_v, 228, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalExecutable::ValidateExecutionOptions");

  if (run_options.stream() != nullptr) {
    if (!run_options.stream()->ok()) {
      return InvalidArgument("stream is uninitialized or in an error state");
    }

    // Check stream matches service platform.
    const se::Platform* stream_platform =
        run_options.stream()->parent()->platform();
    if (stream_platform != backend_->platform()) {
      return InvalidArgument(
          "stream is for platform %s, but service targets platform %s",
          stream_platform->Name(), backend_->platform()->Name());
    }

    // Cannot specify device_ordinal with a stream. The stream determines these
    // values.
    if (run_options.device_ordinal() != -1) {
      return InvalidArgument(
          "cannot set both device ordinal and stream options in "
          "ExecutableRunOptions; the stream determines the device ordinal");
    }
  }

  // Verify that the device the executable was built for is equivalent
  // to the device it will run on.
  int run_device_ordinal = run_options.device_ordinal();
  if (run_device_ordinal == -1) {
    run_device_ordinal = run_options.stream() != nullptr
                             ? run_options.stream()->parent()->device_ordinal()
                             : backend_->default_device_ordinal();
  }
  TF_ASSIGN_OR_RETURN(bool devices_equivalent,
                      backend_->devices_equivalent(
                          run_device_ordinal, build_options_.device_ordinal()));
  if (!devices_equivalent) {
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * run_executor,
                        backend_->stream_executor(run_device_ordinal));
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * build_executor,
                        backend_->stream_executor(build_device_ordinal()));
    return InvalidArgument(
        "executable is built for device %s of type \"%s\"; cannot run it on "
        "device %s of type \"%s\"",
        backend_->device_name(build_device_ordinal()),
        build_executor->GetDeviceDescription().name(),
        backend_->device_name(run_device_ordinal),
        run_executor->GetDeviceDescription().name());
  }

  if (!run_options.allocator()) {
    return InvalidArgument("an allocator must be provided to ExecuteLocally");
  }

  if (run_options.allocator()->platform() != backend.platform()) {
    return InvalidArgument(
        "allocator platform (%s) does not match service platform (%s)",
        run_options.allocator()->platform()->Name(),
        backend.platform()->Name());
  }

  return Status::OK();
}

StatusOr<std::pair<ServiceExecutableRunOptions, StreamPool::Ptr>>
LocalExecutable::RunHelper(const absl::Span<const Shape* const> argument_shapes,
                           ExecutableRunOptions run_options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_2(mht_2_v, 296, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalExecutable::RunHelper");

  const ComputationLayout& computation_layout =
      executable_->module_config().entry_computation_layout();

  // Check argument number, shapes, and layouts.
  const int argument_shapes_size = argument_shapes.size();
  if (argument_shapes_size != computation_layout.parameter_count()) {
    return InvalidArgument(
        "invalid number of arguments for computation: expected %d, got %u",
        computation_layout.parameter_count(), argument_shapes.size());
  }
  for (int i = 0, end = argument_shapes.size(); i < end; ++i) {
    // TODO(b/187081154): Compare tiling info also.
    if (!computation_layout.parameter_layout(i).MatchesLayoutInShape(
            *argument_shapes[i], /*minor_to_major_only=*/false,
            /*ignore_fully_empty_tiling=*/true)) {
      return InvalidParameterArgument(
          executable_.get(), i,
          "Argument does not match host shape or layout of computation "
          "parameter "
          "%d: want %s, got %s",
          i,
          ShapeUtil::HumanStringWithLayout(
              computation_layout.parameter_layout(i).shape()),
          ShapeUtil::HumanStringWithLayout(*argument_shapes[i]));
    }
  }

  TF_RETURN_IF_ERROR(ValidateExecutionOptions(run_options, *backend_));

  StreamPool::Ptr stream;
  if (run_options.stream() == nullptr) {
    // NB!  The lifetime of `stream` needs to match the lifetime of
    // `service_options` (otherwise we will end up using a returned stream in
    // ExecuteOnStreamWrapper), which is why it isn't declared in the inner "if"
    // scope.
    TF_ASSIGN_OR_RETURN(
        stream, BorrowStreamForDevice(run_options.device_ordinal(), backend_));
    run_options.set_stream(stream.get());
  }
  if (run_options.allocator() == nullptr) {
    run_options.set_allocator(backend_->memory_allocator());
  }

  // For local client execution on CPU backends:
  // *) The thread pool used for eigen CPU ops is from
  //    ExecutableRunOptions.eigen_intra_op_thread_pool.
  // *) The thread pool used for XLA CPU ops is from
  //    backend_->eigen_intra_op_thread_pool().
  ServiceExecutableRunOptions service_options(run_options,
                                              backend_->StreamBorrower());
  return std::make_pair(service_options, std::move(stream));
}

StatusOr<ScopedShapedBuffer> LocalExecutable::Run(
    const absl::Span<const ShapedBuffer* const> arguments,
    ExecutableRunOptions run_options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_3(mht_3_v, 355, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalExecutable::Run");

  std::vector<const Shape*> argument_shapes;
  argument_shapes.reserve(arguments.size());
  for (const ShapedBuffer* const arg : arguments) {
    argument_shapes.push_back(&arg->on_device_shape());
  }
  return AsyncCallAndBlockHostUntilDone<xla::ScopedShapedBuffer>(
      argument_shapes, run_options, [&](const ExecutableRunOptions& options) {
        return RunAsync(arguments, options);
      });
}

StatusOr<ExecutionOutput> LocalExecutable::Run(
    std::vector<ExecutionInput> arguments, ExecutableRunOptions run_options) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_4(mht_4_v, 371, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalExecutable::Run");

  std::vector<const Shape*> argument_shapes;
  argument_shapes.reserve(arguments.size());
  for (const ExecutionInput& arg : arguments) {
    argument_shapes.push_back(&arg.shape());
  }
  return AsyncCallAndBlockHostUntilDone<ExecutionOutput>(
      argument_shapes, run_options, [&](const ExecutableRunOptions& options) {
        return RunAsync(argument_shapes, std::move(arguments), options);
      });
}

static std::shared_ptr<HloSnapshot> DumpArguments(
    const Backend* backend, const Executable* executable,
    const absl::Span<const ShapedBuffer* const> arguments, se::Stream* stream) {
  auto snapshot = std::make_shared<HloSnapshot>();
  snapshot->set_execution_platform(backend->platform()->Name());
  *snapshot->mutable_hlo() = *executable->hlo_proto();
  for (const ShapedBuffer* arg : arguments) {
    auto literal = std::make_shared<Literal>(arg->on_host_shape());
    backend->transfer_manager()->TransferLiteralFromDevice(
        stream, *arg, literal.get(), [snapshot, literal](Status status) {
          if (!status.ok()) {
            LOG(ERROR) << "TransferLiteralFromDevice for HLO snapshot inputs "
                          "failed: "
                       << status;
            return;
          }
          *snapshot->add_arguments() = literal->ToProto();
        });
  }
  return snapshot;
}

static void DumpOutputsAndSaveSnapshot(const Backend* backend,
                                       const ShapedBuffer& outputs,
                                       std::shared_ptr<HloSnapshot> snapshot,
                                       se::Stream* stream) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_5(mht_5_v, 411, "", "./tensorflow/compiler/xla/client/local_client.cc", "DumpOutputsAndSaveSnapshot");

  auto literal = std::make_shared<Literal>(outputs.on_host_shape());
  backend->transfer_manager()->TransferLiteralFromDevice(
      stream, outputs, literal.get(),
      [snapshot{std::move(snapshot)}, literal](Status status) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_6(mht_6_v, 418, "", "./tensorflow/compiler/xla/client/local_client.cc", "lambda");

        if (status.ok()) {
          *snapshot->mutable_result() = literal->ToProto();
        } else {
          LOG(ERROR)
              << "TransferLiteralFromDevice for HLO snapshot outputs failed: "
              << status;
        }
        DumpHloSnapshotIfEnabled(*snapshot, GetDebugOptionsFromFlags());
      });
}

StatusOr<ScopedShapedBuffer> LocalExecutable::RunAsync(
    const absl::Span<const ShapedBuffer* const> arguments,
    ExecutableRunOptions run_options) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_7(mht_7_v, 435, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalExecutable::RunAsync");

  std::vector<const Shape*> argument_shapes;
  argument_shapes.reserve(arguments.size());
  for (const ShapedBuffer* const arg : arguments) {
    argument_shapes.push_back(&arg->on_device_shape());
  }
  TF_ASSIGN_OR_RETURN(auto options_and_stream,
                      RunHelper(argument_shapes, run_options));
  se::Stream* stream = run_options.stream();

  std::shared_ptr<HloSnapshot> snapshot;
  if (executable_->dumping_snapshot()) {
    snapshot = DumpArguments(backend_, executable_.get(), arguments, stream);
  }

  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer outputs,
                      executable_->ExecuteAsyncOnStreamWrapper(
                          &options_and_stream.first, arguments));

  // Transfer the outputs and save the snapshot to disk.
  if (snapshot) {
    DumpOutputsAndSaveSnapshot(backend_, outputs, std::move(snapshot), stream);
  }

  return std::move(outputs);
}

static ShapedBuffer MaybeOwningShapeTreeToShapedBuffer(
    const ShapeTree<MaybeOwningDeviceMemory>& tree, int device_ordinal) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_8(mht_8_v, 466, "", "./tensorflow/compiler/xla/client/local_client.cc", "MaybeOwningShapeTreeToShapedBuffer");

  ShapedBuffer result(tree.shape(), device_ordinal);
  auto it = tree.begin();
  auto out_it = result.buffers().begin();
  for (; it != tree.end(); ++it, ++out_it) {
    out_it->second = it->second.AsDeviceMemoryBase();
  }
  return result;
}

StatusOr<ExecutionOutput> LocalExecutable::RunAsync(
    absl::Span<Shape const* const> argument_host_shapes,
    std::vector<ExecutionInput> arguments, ExecutableRunOptions run_options) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_9(mht_9_v, 481, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalExecutable::RunAsync");

  if (argument_host_shapes.size() != arguments.size()) {
    return InvalidArgument(
        "Number of argument host shapes not equal to number of arguments (%d "
        "vs %d)",
        argument_host_shapes.size(), arguments.size());
  }
  TF_ASSIGN_OR_RETURN(auto options_and_stream,
                      RunHelper(argument_host_shapes, run_options));
  se::Stream* stream = run_options.stream();

  std::shared_ptr<HloSnapshot> snapshot;
  if (executable_->dumping_snapshot()) {
    std::vector<ShapedBuffer> shaped_buffers;
    std::vector<const ShapedBuffer*> shaped_buffer_ptrs;
    shaped_buffers.reserve(arguments.size());
    shaped_buffer_ptrs.reserve(arguments.size());
    for (size_t i = 0; i < arguments.size(); ++i) {
      shaped_buffers.push_back(MaybeOwningShapeTreeToShapedBuffer(
          arguments[i].Buffers(), stream->parent()->device_ordinal()));
      shaped_buffer_ptrs.push_back(&shaped_buffers.back());
    }

    snapshot =
        DumpArguments(backend_, executable_.get(), shaped_buffer_ptrs, stream);
  }

  TF_ASSIGN_OR_RETURN(ExecutionOutput outputs,
                      executable_->ExecuteAsyncOnStreamWrapper(
                          &options_and_stream.first, std::move(arguments)));

  // Transfer the outputs and save the snapshot to disk.
  if (snapshot) {
    DumpOutputsAndSaveSnapshot(backend_, outputs.Result(), std::move(snapshot),
                               stream);
  }

  return std::move(outputs);
}

StatusOr<ExecutionOutput> LocalExecutable::RunAsync(
    std::vector<ExecutionInput> arguments, ExecutableRunOptions run_options) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_10(mht_10_v, 525, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalExecutable::RunAsync");

  std::vector<const Shape*> argument_shapes;
  argument_shapes.reserve(arguments.size());
  for (const ExecutionInput& arg : arguments) {
    argument_shapes.push_back(&arg.shape());
  }
  return RunAsync(argument_shapes, std::move(arguments), run_options);
}

se::Platform* LocalClient::platform() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_11(mht_11_v, 537, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::platform");

  return local_service_->backend().platform();
}

int LocalClient::device_count() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_12(mht_12_v, 544, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::device_count");

  return local_service_->backend().device_count();
}

bool LocalClient::device_ordinal_supported(int device_ordinal) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_13(mht_13_v, 551, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::device_ordinal_supported");

  return local_service_->backend().device_ordinal_supported(device_ordinal);
}

int LocalClient::default_device_ordinal() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_14(mht_14_v, 558, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::default_device_ordinal");

  return local_service_->backend().default_device_ordinal();
}

const Backend& LocalClient::backend() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_15(mht_15_v, 565, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::backend");

  return local_service_->backend();
}

Backend* LocalClient::mutable_backend() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_16(mht_16_v, 572, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::mutable_backend");

  return local_service_->mutable_backend();
}

static StatusOr<ExecutableBuildOptions> UpdateBuildOptions(
    const ExecutableBuildOptions& options, int default_device_ordinal) {
  ExecutableBuildOptions updated_options = options;
  if (options.device_ordinal() == -1) {
    updated_options.set_device_ordinal(default_device_ordinal);
    VLOG(3) << "Set device ordinal to default value of: "
            << updated_options.device_ordinal();
  }
  if (options.has_device_assignment()) {
    if (options.device_assignment().replica_count() != options.num_replicas()) {
      return InvalidArgument(
          "Mismatched number of replicas for device "
          "assignment and computation (%d vs %d).\n%s",
          options.device_assignment().replica_count(), options.num_replicas(),
          options.device_assignment().ToString());
    }
    if (options.device_assignment().computation_count() !=
        options.num_partitions()) {
      return InvalidArgument(
          "Mismatched number of partitions for device "
          "assignment and computation (%d vs %d).\n%s",
          options.device_assignment().computation_count(),
          options.num_partitions(), options.device_assignment().ToString());
    }
  }
  return updated_options;
}

StatusOr<std::vector<std::unique_ptr<LocalExecutable>>> LocalClient::Compile(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& options) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_17(mht_17_v, 610, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::Compile");

  TF_ASSIGN_OR_RETURN(ExecutableBuildOptions updated_options,
                      UpdateBuildOptions(options, default_device_ordinal()));
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Executable>> executables,
                      local_service_->CompileExecutables(
                          computation, argument_layouts, updated_options));

  std::vector<std::unique_ptr<LocalExecutable>> local_executables;
  local_executables.reserve(executables.size());

  for (auto& executable : executables) {
    local_executables.push_back(absl::make_unique<LocalExecutable>(
        std::move(executable), local_service_->mutable_backend(),
        updated_options));
  }

  return std::move(local_executables);
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
LocalClient::CompileAheadOfTime(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& options) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_18(mht_18_v, 636, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::CompileAheadOfTime");

  TF_ASSIGN_OR_RETURN(ExecutableBuildOptions updated_options,
                      UpdateBuildOptions(options, default_device_ordinal()));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      local_service_->CompileAotResults(computation, argument_layouts,
                                        updated_options));

  return std::move(aot_results);
}

StatusOr<std::unique_ptr<LocalExecutable>> LocalClient::Load(
    const std::string& serialized_aot_result,
    const ExecutableBuildOptions& options) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("serialized_aot_result: \"" + serialized_aot_result + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_19(mht_19_v, 653, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::Load");

  TF_ASSIGN_OR_RETURN(ExecutableBuildOptions updated_options,
                      UpdateBuildOptions(options, default_device_ordinal()));
  TF_ASSIGN_OR_RETURN(
      se::StreamExecutor * executor,
      backend().stream_executor(updated_options.device_ordinal()));

  TF_ASSIGN_OR_RETURN(Compiler * compiler,
                      Compiler::GetForPlatform(platform()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::AotCompilationResult> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      aot_result->LoadExecutable(compiler, executor));
  return absl::make_unique<LocalExecutable>(std::move(executable),
                                            local_service_->mutable_backend(),
                                            updated_options);
}

StatusOr<ScopedShapedBuffer> LocalClient::LiteralToShapedBuffer(
    const LiteralSlice& literal, int device_ordinal,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_20(mht_20_v, 678, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::LiteralToShapedBuffer");

  if (allocator == nullptr) {
    allocator = backend().memory_allocator();
  }
  TF_ASSIGN_OR_RETURN(auto scoped_buffer,
                      backend().transfer_manager()->AllocateScopedShapedBuffer(
                          literal.shape(), allocator, device_ordinal));
  TF_ASSIGN_OR_RETURN(auto stream,
                      mutable_backend()->BorrowStream(device_ordinal));
  TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
      stream.get(), literal, scoped_buffer));
  return std::move(scoped_buffer);
}

StatusOr<Literal> LocalClient::ShapedBufferToLiteral(
    const ShapedBuffer& shaped_buffer) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_21(mht_21_v, 696, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::ShapedBufferToLiteral");

  TF_ASSIGN_OR_RETURN(auto stream, mutable_backend()->BorrowStream(
                                       shaped_buffer.device_ordinal()));
  return backend().transfer_manager()->TransferLiteralFromDevice(stream.get(),
                                                                 shaped_buffer);
}

StatusOr<const ShapedBuffer*> LocalClient::GlobalDataToShapedBuffer(
    const GlobalDataHandle& data, int replica_number) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_22(mht_22_v, 707, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::GlobalDataToShapedBuffer");

  return local_service_->GlobalDataToShapedBuffer(data, replica_number);
}

Status LocalClient::TransferToInfeedLocal(const LiteralSlice& literal,
                                          int device_ordinal) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_23(mht_23_v, 715, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::TransferToInfeedLocal");

  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      backend().stream_executor(device_ordinal));
  return backend().transfer_manager()->TransferLiteralToInfeed(executor,
                                                               literal);
}

Status LocalClient::TransferFromOutfeedLocal(int device_ordinal,
                                             MutableBorrowingLiteral literal) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_24(mht_24_v, 726, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::TransferFromOutfeedLocal");

  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      backend().stream_executor(device_ordinal));
  return backend().transfer_manager()->TransferLiteralFromOutfeed(executor,
                                                                  literal);
}

StatusOr<int> LocalClient::ReplicaNumberToDeviceOrdinal(int replica_number) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_25(mht_25_v, 736, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::ReplicaNumberToDeviceOrdinal");

  return local_service_->ReplicaNumberToDeviceOrdinal(replica_number);
}

StatusOr<TransferToServerResponse> LocalClient::TransferToLocalServer(
    const ::xla::BorrowingLiteral& literal, int device_ordinal) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTcc mht_26(mht_26_v, 744, "", "./tensorflow/compiler/xla/client/local_client.cc", "LocalClient::TransferToLocalServer");

  const ::xla::Shape& shape = literal.shape();

  TF_ASSIGN_OR_RETURN(::xla::ScopedShapedBuffer shaped_buffer,
                      backend().transfer_manager()->AllocateScopedShapedBuffer(
                          shape, backend().memory_allocator(), device_ordinal));
  TF_ASSIGN_OR_RETURN(auto stream,
                      mutable_backend()->BorrowStream(device_ordinal));
  TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
      stream.get(), literal, shaped_buffer));
  std::vector<::xla::ScopedShapedBuffer> replicated_buffer;
  replicated_buffer.emplace_back(std::move(shaped_buffer));
  ::xla::TransferToServerResponse result;
  TF_ASSIGN_OR_RETURN(*result.mutable_data(),
                      local_service_->RegisterReplicatedBuffers(
                          std::move(replicated_buffer),
                          absl::StrCat("TransferToServer literal of shape ",
                                       ::xla::ShapeUtil::HumanString(shape))));

  return result;
}

}  // namespace xla
