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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSserviceDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSserviceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSserviceDTh() {
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
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/allocation_tracker.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/channel_tracker.h"
#include "tensorflow/compiler/xla/service/compilation_cache.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/execution_tracker.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {

// Options to configure the service when it is created.
class ServiceOptions {
 public:
  // Set the platform backing the service, or nullptr for the default platform.
  ServiceOptions& set_platform(se::Platform* platform);
  se::Platform* platform() const;

  // Set the default number of replicas to use when compiling replicated
  // programs.
  ServiceOptions& set_number_of_replicas(int number_of_replicas);
  int number_of_replicas() const;

  // Sets the thread pool size for parallel execution of an individual operator.
  ServiceOptions& set_intra_op_parallelism_threads(int num_threads);
  int intra_op_parallelism_threads() const;

  // Sets the allowed_devices set for selectively constructing stream executors
  // on the platform.
  ServiceOptions& set_allowed_devices(
      const absl::optional<std::set<int>>& allowed_devices);
  const absl::optional<std::set<int>>& allowed_devices() const;

 private:
  se::Platform* platform_ = nullptr;
  int number_of_replicas_ = 1;
  int intra_op_parallelism_threads_ = -1;
  absl::optional<std::set<int>> allowed_devices_;
};

// The XLA service object, which is the same across all platforms. It maintains
// the service state of computations and allocations, and delegates
// target-specific requests to the target-specific infrastructure
// (target-specific compiler, StreamExecutor).
class Service : public ServiceInterface {
 public:
  // Factory method for creating a new Service.
  static StatusOr<std::unique_ptr<Service>> NewService(
      se::Platform* platform = nullptr);
  static StatusOr<std::unique_ptr<Service>> NewService(
      const ServiceOptions& options);

  // Unregisters a previously-allocated global handle.
  //
  // If the handle given is not currently allocated, a NOT_FOUND status is
  // returned.
  Status Unregister(const UnregisterRequest* arg,
                    UnregisterResponse* result) override;

  // Deconstructs a tuple. Returns a newly created GlobalDataHandle for each
  // element in the tuple.
  Status DeconstructTuple(const DeconstructTupleRequest* arg,
                          DeconstructTupleResponse* result) override;

  // Compiles a computation into an executable. The request contains the whole
  // computation graph. Returns the handle to the executable.
  Status Compile(const CompileRequest* arg, CompileResponse* result) override;

  // Executes an executable with the provided global data passes as immutable
  // arguments. The request contains the handle to the executable. Returns
  // global data output and execution timing.
  Status Execute(const ExecuteRequest* arg, ExecuteResponse* result) override;

  // Executes one or more computations in parallel with the provided global data
  // passed as immutable arguments. Returns global data output for each
  // computation.
  Status ExecuteGraphParallel(const ExecuteGraphParallelRequest* arg,
                              ExecuteParallelResponse* result) override;

  // Requests one or more device handles from the target.
  //
  // When N device handles are requested and the number of replicas is R, at
  // least N * R devices must be available. The devices are assigned based on
  // the device ordinals such that the first R available devices are assigned to
  // the first set of replicas, and the next R devices to the second set of
  // replicas, etc. Each returned device handle represents the device with the
  // replica id 0.
  Status GetDeviceHandles(const GetDeviceHandlesRequest* arg,
                          GetDeviceHandlesResponse* result) override;

  // Waits until the specified execution is complete and returns the result.
  // Calling this API multiple times with the same execution handle returns the
  // method with an error since the execution handle is destroyed after the
  // first call.
  Status WaitForExecution(const WaitForExecutionRequest* arg,
                          WaitForExecutionResponse* result) override;

  // Requests that global data be transferred to the client in literal form.
  Status TransferToClient(const TransferToClientRequest* arg,
                          TransferToClientResponse* result) override;

  // Transfers data from a literal provided by the client, into device memory.
  Status TransferToServer(const TransferToServerRequest* arg,
                          TransferToServerResponse* result) override;

  // Transfers data from a literal provided by the client, into the Infeed
  // buffer of the device.
  Status TransferToInfeed(const TransferToInfeedRequest* arg,
                          TransferToInfeedResponse* result) override;

  // Transfers data from the Outfeed othe device to the literal provided by the
  // client.
  Status TransferFromOutfeed(const TransferFromOutfeedRequest* arg,
                             TransferFromOutfeedResponse* result) override;

  // Resets devices, clearing all existing state on all the devices associated
  // with this service (including memory allocated on the devices).
  //
  // ResetDevice may only be called where no previous Execution state on the
  // device is used by the next Execution.
  //
  // ResetDevice should be called before an Execution that expect the device to
  // be in the reset state. For example, if the prior Execution modifies device
  // state (e.g., architectural state) that the next Execution depends on.
  Status ResetDevice(const ResetDeviceRequest* arg,
                     ResetDeviceResponse* result) override;

  Status ComputeConstantGraph(const ComputeConstantGraphRequest* arg,
                              ComputeConstantResponse* result) override;

  // Returns the shape (with layout) of an array associated with a given data
  // handle.
  Status GetShape(const GetShapeRequest* arg,
                  GetShapeResponse* result) override;

  // Retrieves the statistics of a computation.
  Status GetComputationGraphStats(const ComputationGraphStatsRequest* arg,
                                  ComputationStatsResponse* result) override;

  // Creates a unique channel handle that can be used for Send/Recv
  // instructions.
  Status CreateChannelHandle(const CreateChannelHandleRequest* arg,
                             CreateChannelHandleResponse* result) override;

  // Returns the backend used to execute computations.
  const Backend& backend() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSserviceDTh mht_0(mht_0_v, 351, "", "./tensorflow/compiler/xla/service/service.h", "backend");
 return *execute_backend_; }
  Backend* mutable_backend() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSserviceDTh mht_1(mht_1_v, 355, "", "./tensorflow/compiler/xla/service/service.h", "mutable_backend");
 return execute_backend_.get(); }

  // Create a Hlo module config for the given program shape and arguments.
  // aot_options is optional; if not given a default is used.
  StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
      const ProgramShape& program_shape,
      absl::Span<const Shape* const> argument_shapes,
      const ExecutionOptions* execution_options,
      const AotCompilationOptions* aot_options = nullptr);

 private:
  // A private overload for Service itself, used by other methods within this
  // class.
  StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
      const ProgramShape& program_shape,
      absl::Span<const ShapedBuffer* const> arguments,
      const ExecutionOptions& execution_options,
      const AotCompilationOptions* aot_options = nullptr);

  // Prepare the executors for executing parallel.
  StatusOr<std::vector<se::StreamExecutor*>> GetExecutors(
      const ExecutionOptions& execution_options, int64_t requests_size,
      int64_t request_index) const;

  // Prepare the arguments for executing parallel.
  StatusOr<std::vector<std::vector<const ShapedBuffer*>>> GetArguments(
      const ExecutionOptions& execution_options,
      absl::Span<const GlobalDataHandle* const> arguments) const;

 protected:
  friend class LocalExecutable;

  // The constructor is private. Use the NewService factory to create new
  // service objects.
  Service(const ServiceOptions& options,
          std::unique_ptr<Backend> execute_backend);

  // Resolves the given argument handles in the allocation tracker and returns
  // the corresponding allocations for every replica. The function also verifies
  // that each allocation matches the execution platform and device ordinal of
  // the corresponding replica.
  StatusOr<std::vector<std::vector<const ShapedBuffer*>>>
  ResolveAndValidateArguments(
      absl::Span<const GlobalDataHandle* const> arguments,
      absl::Span<se::StreamExecutor* const> stream_executors) const;

  // Builds an Executable for the given parameters.
  //
  // If device_allocator is not null, the compiler may use it to allocate temp
  // buffers, which the compiler is responsible for freeing.  The allocator
  // given here need not match the allocator used when running the executable.
  StatusOr<std::unique_ptr<Executable>> BuildExecutable(
      const HloModuleProto& module_proto,
      std::unique_ptr<HloModuleConfig> module_config, Backend* backend,
      se::StreamExecutor* executor, const Compiler::CompileOptions& options,
      bool run_backend_only = false);

  // Same as BuildExecutable() above, but builds a list of Executables for the
  // given computations that may interact with each other.
  StatusOr<std::vector<std::unique_ptr<Executable>>> BuildExecutables(
      const std::vector<const HloModuleProto*>& module_protos,
      std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
      Backend* backend, std::vector<std::vector<se::StreamExecutor*>> executors,
      const Compiler::CompileOptions& options, bool run_backend_only = false);

  // Same as BuildExecutable() above, but builds a list of
  // AotCompilationResult(s), which can be persisted to later load Executable
  // objects.
  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>> BuildAotResults(
      const std::vector<const HloModuleProto*>& module_protos,
      std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
      Backend* backend, std::vector<std::vector<se::StreamExecutor*>> executors,
      const Compiler::CompileOptions& options, bool run_backend_only = false);

  // Runs the given executable with the given arguments and register the result
  // in the allocation tracker. The handle of the result from the tracker is
  // returned. If the parameter "profile" is not null, it points to an
  // ExecutionProfile object which will be filled in with profile data.
  StatusOr<GlobalDataHandle> ExecuteAndRegisterResult(
      Executable* executable,
      absl::Span<const std::vector<const ShapedBuffer*>> arguments,
      Backend* backend, const DeviceHandle& device_handle,
      const std::string& result_tag, ExecutionProfile* profile);

  // Runs the given executables with the given arguments and register the result
  // from each executable in the allocation tracker. The handles of the result
  // from the tracker are returned.
  StatusOr<std::vector<GlobalDataHandle>> ExecuteParallelAndRegisterResult(
      absl::Span<Executable* const> executables,
      absl::Span<const std::vector<std::vector<const ShapedBuffer*>>> arguments,
      Backend* backend, absl::Span<const DeviceHandle> device_handles,
      absl::Span<const std::string> result_tags, ExecutionProfile* profile);

  // Convenience function which checks whether the given client_shape
  // (presumably passed by the client to set the result layout) is valid for the
  // given computation result shape.
  Status ValidateResultShape(const Shape& client_shape,
                             const Shape& result_shape) const;

  // Returns the stream executors assigned to the replicas represented by the
  // given device handle. Each device_handle is a virtual replicated device that
  // represents a set of physical devices for the replicas.
  StatusOr<std::vector<se::StreamExecutor*>> Replicas(
      const Backend& backend, const DeviceHandle& device_handle) const;

  // Returns the device handle that represents the replicated device for a
  // single computation that is not model-parallelized.
  DeviceHandle SingleComputationDeviceHandle() const;

  ServiceOptions options_;

  // Cache containing previously built Executables.
  CompilationCache compilation_cache_;

  // Tracks channels created via the API.
  ChannelTracker channel_tracker_;

  // Tracks allocations made via the API and computation execution.
  AllocationTracker allocation_tracker_;

  // Tracks asynchronously launched executions via the API.
  ExecutionTracker execution_tracker_;

  // Backend to compile and execute computations on.
  std::unique_ptr<Backend> execute_backend_;

  Service(const Service&) = delete;
  Service& operator=(const Service&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_H_
