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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc() {
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
#define EIGEN_USE_THREADS

#include "tensorflow/compiler/xla/service/hlo_runner.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module_group.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

HloRunner::HloRunner(se::Platform* platform, int intra_op_parallelism_threads) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::HloRunner");

  BackendOptions backend_options;
  backend_options.set_platform(platform);
  backend_options.set_intra_op_parallelism_threads(
      intra_op_parallelism_threads);
  backend_ = Backend::CreateBackend(backend_options).ConsumeValueOrDie();
  device_shape_representation_fn_ = [this](const Shape& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "lambda");

    return backend_->compiler()->DeviceShapeRepresentation(shape);
  };
  VLOG(1) << "Created HloRunner for platform: " << platform->Name();
}

HloRunner::~HloRunner() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_2(mht_2_v, 223, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::~HloRunner");
}

StatusOr<ScopedShapedBuffer> HloRunner::TransferLiteralToDevice(
    const Literal& literal) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_3(mht_3_v, 229, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::TransferLiteralToDevice");

  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer buffer,
      backend().transfer_manager()->AllocateScopedShapedBuffer(
          literal.shape(), backend().memory_allocator(),
          backend().default_device_ordinal(), device_shape_representation_fn_));
  TF_ASSIGN_OR_RETURN(
      auto stream, backend().BorrowStream(backend().default_stream_executor()));
  TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
      stream.get(), literal, buffer));
  return std::move(buffer);
}

StatusOr<std::vector<ScopedShapedBuffer>> HloRunner::TransferLiteralsToDevice(
    absl::Span<const Literal* const> literals) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_4(mht_4_v, 246, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::TransferLiteralsToDevice");

  std::vector<ScopedShapedBuffer> buffers;
  buffers.reserve(literals.size());
  for (const Literal* literal : literals) {
    CHECK(literal != nullptr);
    TF_ASSIGN_OR_RETURN(ScopedShapedBuffer buffer,
                        TransferLiteralToDevice(*literal));
    buffers.push_back(std::move(buffer));
  }
  return std::move(buffers);
}

StatusOr<std::vector<ScopedShapedBuffer>> HloRunner::TransferLiteralsToDevice(
    absl::Span<const Literal> literals) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_5(mht_5_v, 262, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::TransferLiteralsToDevice");

  std::vector<const Literal*> literal_pointers;
  literal_pointers.reserve(literals.size());
  for (const auto& literal : literals) {
    literal_pointers.push_back(&literal);
  }
  return TransferLiteralsToDevice(literal_pointers);
}

StatusOr<Literal> HloRunner::TransferLiteralFromDevice(
    const ShapedBuffer& buffer) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_6(mht_6_v, 275, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::TransferLiteralFromDevice");

  TF_ASSIGN_OR_RETURN(
      auto stream, backend().BorrowStream(backend().default_stream_executor()));
  return backend().transfer_manager()->TransferLiteralFromDevice(stream.get(),
                                                                 buffer);
}

StatusOr<Literal> HloRunner::Execute(std::unique_ptr<HloModule> module,
                                     absl::Span<const Literal* const> arguments,
                                     bool run_hlo_passes,
                                     ExecutionProfile* profile) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_7(mht_7_v, 288, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::Execute");

  UpdateEntryComputationLayout(module.get(), device_shape_representation_fn_);

  TF_ASSIGN_OR_RETURN(std::vector<ScopedShapedBuffer> argument_buffers,
                      TransferLiteralsToDevice(arguments));
  TF_ASSIGN_OR_RETURN(ExecutionOutput result,
                      ExecuteWithDeviceBuffers(
                          /*module=*/std::move(module),
                          /*arguments=*/argument_buffers,
                          /*run_hlo_passes=*/run_hlo_passes,
                          /*profile=*/profile));
  return TransferLiteralFromDevice(result.Result());
}

StatusOr<Literal> HloRunner::ExecuteWithExecutable(
    Executable* executable, absl::Span<const Literal* const> arguments,
    ExecutionProfile* profile) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_8(mht_8_v, 307, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::ExecuteWithExecutable");

  TF_ASSIGN_OR_RETURN(std::vector<ScopedShapedBuffer> argument_buffers,
                      TransferLiteralsToDevice(arguments));
  TF_ASSIGN_OR_RETURN(ExecutionOutput result,
                      ExecuteWithDeviceBuffers(
                          /*executable=*/executable,
                          /*arguments=*/argument_buffers,
                          /*profile=*/profile));
  return TransferLiteralFromDevice(result.Result());
}

// Convert the owning buffer of inputs into a (partially) owning vector of
// ExecutionInputs, and an owning vector of `OwningDeviceMemory`'s.
static std::vector<ExecutionInput> ExecutionInputsFromScopedShapedBuffers(
    absl::Span<ScopedShapedBuffer const> inputs,
    HloInputOutputAliasConfig alias_config, int device_ordinal,
    se::DeviceMemoryAllocator* allocator) {
  std::vector<ExecutionInput> execution_inputs;
  std::vector<se::OwningDeviceMemory> owned_args;

  for (int param_num = 0; param_num < inputs.size(); param_num++) {
    const ScopedShapedBuffer& input_buffer = inputs[param_num];
    ShapeTree<MaybeOwningDeviceMemory> buffer_tree(
        input_buffer.on_device_shape());

    input_buffer.buffers().ForEachElement(
        [&](const ShapeIndex& index,
            const se::DeviceMemoryBase& execution_input_buffer) {
          if (alias_config.ParameterHasAlias(param_num, index)) {
            // Store owned.
            *buffer_tree.mutable_element(index) = se::OwningDeviceMemory{
                execution_input_buffer, device_ordinal, allocator};
          } else {
            // Store unowned.
            *buffer_tree.mutable_element(index) = execution_input_buffer;
          }
        });
    execution_inputs.emplace_back(std::move(buffer_tree));
  }
  return execution_inputs;
}

StatusOr<ExecutionOutput> HloRunner::ExecuteWithDeviceBuffers(
    std::unique_ptr<HloModule> module,
    absl::Span<ScopedShapedBuffer const> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_9(mht_9_v, 355, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::ExecuteWithDeviceBuffers");

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      CreateExecutable(std::move(module), run_hlo_passes));
  return ExecuteWithDeviceBuffers(executable.get(), arguments, profile);
}

StatusOr<ExecutionOutput> HloRunner::ExecuteWithDeviceBuffers(
    Executable* executable, absl::Span<ScopedShapedBuffer const> arguments,
    ExecutionProfile* profile) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_10(mht_10_v, 366, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::ExecuteWithDeviceBuffers");

  UpdateEntryComputationLayout(&executable->module(),
                               device_shape_representation_fn_);

  // Get service run options.
  se::Stream stream(backend().default_stream_executor());
  stream.Init();
  ServiceExecutableRunOptions service_run_options =
      GetServiceRunOptionsForDevice(backend().default_device_ordinal(), &stream,
                                    nullptr, RunId());
  service_run_options.mutable_run_options()->set_execution_profile(profile);

  std::vector<ExecutionInput> execution_arguments =
      ExecutionInputsFromScopedShapedBuffers(
          arguments, executable->module().input_output_alias_config(),
          stream.parent()->device_ordinal(), stream.parent()->GetAllocator());

  TF_ASSIGN_OR_RETURN(
      ExecutionOutput retval,
      executable->ExecuteOnStreamWrapper(&service_run_options,
                                         std::move(execution_arguments)));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return std::move(retval);
}

StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    std::unique_ptr<HloModule> module, const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_11(mht_11_v, 396, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::ExecuteReplicated");

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      CreateExecutable(std::move(module), options.run_hlo_passes));
  return ExecuteReplicated(executable.get(), options, device_assignment);
}

StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicatedImpl(
    std::function<StatusOr<std::vector<ScopedShapedBuffer>>(
        const std::vector<ServiceExecutableRunOptions>&,
        const std::vector<absl::Span<const ShapedBuffer* const>>&)>
        execution_helper,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_12(mht_12_v, 414, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::ExecuteReplicatedImpl");

  std::vector<std::unique_ptr<se::Stream>> streams;
  std::vector<ServiceExecutableRunOptions> service_run_options;
  int64_t num_partitions = device_assignment->computation_count();

  std::vector<ScopedShapedBuffer> argument_buffers;
  // This reserve() call is necessary for correctness, because
  // argument_buffer_ptrs contains pointers into the elements of
  // argument_buffers.
  const int64_t total_argument_count = [&]() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_13(mht_13_v, 426, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "lambda");

    int64_t total = 0;
    for (int64_t i = 0; i < options.num_replicas; ++i) {
      total += argument_count_provider(i);
    }
    return total;
  }();
  argument_buffers.reserve(total_argument_count);

  // Plus one so we can safely get &argument_buffer_ptrs[0] in case there are
  // no arguments.
  std::vector<const ShapedBuffer*> argument_buffer_ptrs(total_argument_count +
                                                        1);
  std::vector<absl::Span<const ShapedBuffer* const>> argument_buffer_slices;
  int64_t index = 0;
  RunId run_id;
  for (int64_t i = 0; i < options.num_replicas; ++i) {
    int64_t device =
        (*device_assignment)(i / num_partitions, i % num_partitions);
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                        backend().stream_executor(device));
    streams.push_back(absl::make_unique<se::Stream>(executor));
    streams.back()->Init();
    service_run_options.emplace_back(GetServiceRunOptionsForDevice(
        device, streams.back().get(), device_assignment, run_id));

    // Copy arguments to device.
    const int64_t argument_count = argument_count_provider(i);
    for (int64_t arg_index = 0; arg_index < argument_count; arg_index++) {
      const Literal* const argument = argument_provider(i, arg_index);
      TF_RET_CHECK(argument != nullptr);
      TF_ASSIGN_OR_RETURN(
          ScopedShapedBuffer argument_buffer,
          backend().transfer_manager()->AllocateScopedShapedBuffer(
              argument->shape(), backend().memory_allocator(), device,
              device_shape_representation_fn_));
      TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
          streams.back().get(), *argument, argument_buffer));
      argument_buffers.push_back(std::move(argument_buffer));
      argument_buffer_ptrs[index++] = &argument_buffers.back();
    }
    argument_buffer_slices.emplace_back(
        &argument_buffer_ptrs[index - argument_count], argument_count);
  }

  std::unique_ptr<tensorflow::thread::ThreadPool> pool;
  TF_RET_CHECK(options.infeed_values.empty() ||
               options.infeed_values.size() == options.num_replicas);
  int64_t num_threads = options.infeed_values.size();
  if (ShapeUtil::IsInitialized(options.outfeed_shape)) {
    num_threads += options.num_replicas;
  }
  if (num_threads > 0) {
    pool = absl::make_unique<tensorflow::thread::ThreadPool>(
        tensorflow::Env::Default(), "infeed_outfeed",
        /*num_threads=*/num_threads);
  }
  if (!options.infeed_values.empty()) {
    for (int64_t i = 0; i < options.num_replicas; ++i) {
      int64_t device =
          (*device_assignment)(i / num_partitions, i % num_partitions);
      pool->Schedule([this, device, &options, i]() {
        se::StreamExecutor* executor =
            backend().stream_executor(device).ValueOrDie();
        VLOG(1) << "Starting infeed on device " << device;
        for (int64_t step = 1;
             options.infeed_steps < 0 || step <= options.infeed_steps; ++step) {
          TF_CHECK_OK(backend().transfer_manager()->TransferLiteralToInfeed(
              executor, *options.infeed_values[i]));
          if (step % 100 == 0) {
            VLOG(1) << "Infeed step " << step;
          }
        }
      });
    }
  }
  if (ShapeUtil::IsInitialized(options.outfeed_shape)) {
    if (options.outfeed_values) {
      options.outfeed_values->resize(options.num_replicas);
    }
    for (int64_t i = 0; i < options.num_replicas; ++i) {
      int64_t device =
          (*device_assignment)(i / num_partitions, i % num_partitions);
      pool->Schedule([this, device, &options, i]() {
        se::StreamExecutor* executor =
            backend().stream_executor(device).ValueOrDie();
        VLOG(1) << "Starting outfeed on device " << device;
        for (int64_t step = 1;
             options.infeed_steps < 0 || step <= options.infeed_steps; ++step) {
          Literal literal(options.outfeed_shape);
          TF_CHECK_OK(backend().transfer_manager()->TransferLiteralFromOutfeed(
              executor, &literal));
          if (options.outfeed_values) {
            options.outfeed_values->at(i) = std::move(literal);
          }
          if (step % 100 == 0) {
            VLOG(1) << "Outfeed step " << step;
          }
        }
      });
    }
  }

  LOG(INFO) << "Replicated execution started";
  TF_ASSIGN_OR_RETURN(
      std::vector<ScopedShapedBuffer> results,
      execution_helper(service_run_options, argument_buffer_slices));
  LOG(INFO) << "Replicated execution terminated";

  std::vector<Literal> exec_results;
  exec_results.reserve(options.num_replicas);
  for (int64_t i = 0; i < options.num_replicas; ++i) {
    TF_RETURN_IF_ERROR(streams[i]->BlockHostUntilDone());
    TF_ASSIGN_OR_RETURN(Literal literal,
                        backend().transfer_manager()->TransferLiteralFromDevice(
                            streams[i].get(), results[i]));
    exec_results.push_back(std::move(literal));
  }
  return std::move(exec_results);
}

StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    Executable* executable, const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment, ExecutionProfile* profile) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_14(mht_14_v, 552, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::ExecuteReplicated");

  return ExecuteReplicatedImpl(
      [&](const std::vector<ServiceExecutableRunOptions>& service_run_options,
          const std::vector<absl::Span<const ShapedBuffer* const>>&
              argument_buffer_slices)
          -> StatusOr<std::vector<ScopedShapedBuffer>> {
        std::vector<ScopedShapedBuffer> results;
        if (!options.use_threads) {
          TF_ASSIGN_OR_RETURN(
              results, executable->ExecuteOnStreams(service_run_options,
                                                    argument_buffer_slices));
        } else {
          absl::Mutex mutex;
          std::vector<StatusOr<ScopedShapedBuffer>> thread_results(
              options.num_replicas);
          {
            LOG(INFO) << "Creating thread pool for " << options.num_replicas
                      << " replicas";
            tensorflow::thread::ThreadPool pool(
                tensorflow::Env::Default(), "replicas", options.num_replicas);
            for (int64_t i = 0; i < options.num_replicas; ++i) {
              pool.Schedule([&, i] {
                auto result = executable->ExecuteOnStream(
                    &service_run_options[i], argument_buffer_slices[i],
                    nullptr);
                absl::MutexLock lock(&mutex);
                thread_results[i] = std::move(result);
              });
            }

            // Note: the thread pool destructor guarantees it completes all work
            // before we leave this scope.
          }
          for (auto& thread_result : thread_results) {
            if (!thread_result.ok()) {
              return thread_result.status();
            }
            results.push_back(std::move(thread_result).ValueOrDie());
          }
        }
        return results;
      },
      [&](int64_t replica) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_15(mht_15_v, 597, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "lambda");
 return options.arguments.size(); },
      [&](int64_t replica, int64_t index) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_16(mht_16_v, 601, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "lambda");
 return options.arguments[index]; },
      options, device_assignment);
}

StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    std::function<Executable*(int64_t)> executable_provider,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_17(mht_17_v, 613, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::ExecuteReplicated");

  DeviceAssignment computation_device_assignment;
  if (device_assignment == nullptr) {
    TF_ASSIGN_OR_RETURN(
        computation_device_assignment,
        backend().computation_placer()->AssignDevices(options.num_replicas, 1));
    device_assignment = &computation_device_assignment;
  }
  CHECK_NE(device_assignment, nullptr);
  return ExecuteReplicatedImpl(
      [&](const std::vector<ServiceExecutableRunOptions>& service_run_options,
          const std::vector<absl::Span<const ShapedBuffer* const>>&
              argument_buffer_slices)
          -> StatusOr<std::vector<ScopedShapedBuffer>> {
        TF_RET_CHECK(options.use_threads);
        std::vector<ScopedShapedBuffer> results;
        absl::Mutex mutex;
        std::vector<StatusOr<ScopedShapedBuffer>> thread_results(
            options.num_replicas);
        {
          LOG(INFO) << "Creating thread pool for " << options.num_replicas
                    << " replicas";
          tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(),
                                              "replicas", options.num_replicas);
          for (int64_t i = 0; i < options.num_replicas; ++i) {
            for (const auto& arg : argument_buffer_slices[i]) {
              TF_RET_CHECK(arg != nullptr);
            }
            pool.Schedule([&, i] {
              auto result = executable_provider(i)->ExecuteOnStream(
                  &service_run_options[i], argument_buffer_slices[i], nullptr);
              absl::MutexLock lock(&mutex);
              thread_results[i] = std::move(result);
            });
          }

          // Note: the thread pool destructor guarantees it completes all work
          // before we leave this scope.
        }
        for (auto& thread_result : thread_results) {
          if (!thread_result.ok()) {
            return thread_result.status();
          }
          results.push_back(std::move(thread_result).ValueOrDie());
        }
        return results;
      },
      argument_count_provider, argument_provider, options, device_assignment);
}

StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const ReplicatedExecuteOptions& options) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_18(mht_18_v, 668, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::ExecuteReplicated");

  TF_ASSIGN_OR_RETURN(
      DeviceAssignment device_assignment,
      backend().computation_placer()->AssignDevices(options.num_replicas, 1));
  return ExecuteReplicated(std::move(module), options, &device_assignment);
}

StatusOr<std::unique_ptr<Executable>> HloRunner::CreateExecutable(
    std::unique_ptr<HloModule> module, bool run_hlo_passes) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_19(mht_19_v, 679, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::CreateExecutable");

  if (run_hlo_passes) {
    auto module_group = absl::make_unique<HloModuleGroup>(std::move(module));
    TF_ASSIGN_OR_RETURN(
        auto executables,
        backend().compiler()->Compile(std::move(module_group),
                                      {{backend().default_stream_executor()}},
                                      backend().memory_allocator()));
    return std::move(executables[0]);
  }
  return backend().compiler()->RunBackend(std::move(module),
                                          backend().default_stream_executor(),
                                          backend().memory_allocator());
}

ServiceExecutableRunOptions HloRunner::GetServiceRunOptionsForDevice(
    int64_t device, se::Stream* stream, DeviceAssignment* device_assignment,
    RunId run_id) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_20(mht_20_v, 699, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::GetServiceRunOptionsForDevice");

  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(device);
  run_options.set_stream(stream);
  run_options.set_allocator(backend().memory_allocator());
  run_options.set_intra_op_thread_pool(
      backend().eigen_intra_op_thread_pool_device());
  if (device_assignment != nullptr) {
    run_options.set_device_assignment(device_assignment);
  }
  run_options.set_run_id(run_id);
  return ServiceExecutableRunOptions(run_options, backend().StreamBorrower());
}

Backend& HloRunner::backend() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_21(mht_21_v, 716, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::backend");

  if (!backend_) {
    backend_ = Backend::CreateDefaultBackend().ConsumeValueOrDie();
    VLOG(1) << "Executing on platform " << backend().platform()->Name();
  }
  return *backend_;
}

const Backend& HloRunner::backend() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_22(mht_22_v, 727, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::backend");

  return const_cast<HloRunner*>(this)->backend();
}

absl::string_view HloRunner::Name() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runnerDTcc mht_23(mht_23_v, 734, "", "./tensorflow/compiler/xla/service/hlo_runner.cc", "HloRunner::Name");

  return backend_->platform()->Name();
}

}  // namespace xla
