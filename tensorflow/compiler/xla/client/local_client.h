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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LOCAL_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LOCAL_CLIENT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTh() {
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


#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {

class LocalExecutable {
 public:
  // Low-level constructor; LocalClient::Compile() is the usual way to create
  // executables.
  LocalExecutable(std::unique_ptr<Executable> executable, Backend* backend,
                  ExecutableBuildOptions build_options);

  // Run the compiled computation with the given arguments and options and
  // return the result.
  StatusOr<ScopedShapedBuffer> Run(
      const absl::Span<const ShapedBuffer* const> arguments,
      ExecutableRunOptions run_options);

  // Similar to Run(), but allows for donating argument buffers to the
  // executable.
  StatusOr<ExecutionOutput> Run(std::vector<ExecutionInput> arguments,
                                ExecutableRunOptions run_options);

  // Similar to Run(), but need not block the host waiting for the computation
  // to complete before returning.
  StatusOr<ScopedShapedBuffer> RunAsync(
      const absl::Span<const ShapedBuffer* const> arguments,
      ExecutableRunOptions run_options);

  // Similar to RunAsync(), but allows for donating argument buffers to the
  // executable.
  StatusOr<ExecutionOutput> RunAsync(std::vector<ExecutionInput> arguments,
                                     ExecutableRunOptions run_options);

  // Return the options used to build the executable.
  const ExecutableBuildOptions& build_options() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTh mht_0(mht_0_v, 241, "", "./tensorflow/compiler/xla/client/local_client.h", "build_options");
 return build_options_; }

  // Return the built executable.
  Executable* executable() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTh mht_1(mht_1_v, 247, "", "./tensorflow/compiler/xla/client/local_client.h", "executable");
 return executable_.get(); }

 private:
  StatusOr<ExecutionOutput> RunAsync(
      absl::Span<Shape const* const> argument_host_shapes,
      std::vector<ExecutionInput> arguments, ExecutableRunOptions run_options);

  // Validates that the given arguments and options satisfy various constraints
  // of the computation.
  //
  // The given ExecutableRunOptions override any values from TF_XLA_FLAGS
  // environment variable.
  Status ValidateExecutionOptions(const ExecutableRunOptions& run_options,
                                  const Backend& backend);

  // Returns a literal containing the contents of the given ShapedBuffer.
  StatusOr<Literal> LiteralFromShapedBuffer(const ShapedBuffer& shaped_buffer);

  StatusOr<std::pair<ServiceExecutableRunOptions, StreamPool::Ptr>> RunHelper(
      const absl::Span<const Shape* const> argument_shapes,
      ExecutableRunOptions run_options);

  // The ordinal of the device which this executable was compiled for. The
  // executable can run on all equivalent devices (as determined by
  // Backend::devices_equivalent).
  int build_device_ordinal() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTh mht_2(mht_2_v, 275, "", "./tensorflow/compiler/xla/client/local_client.h", "build_device_ordinal");
 return build_options_.device_ordinal(); }

  template <typename T>
  StatusOr<T> AsyncCallAndBlockHostUntilDone(
      absl::Span<Shape const* const> argument_shapes,
      const ExecutableRunOptions& run_options,
      std::function<StatusOr<T>(const ExecutableRunOptions&)> async_callback) {
    TF_ASSIGN_OR_RETURN(auto options_and_stream,
                        RunHelper(argument_shapes, run_options));
    ExecutableRunOptions options = options_and_stream.first.run_options();
    options.set_device_ordinal(-1);
    StatusOr<T> result = async_callback(options);
    Status block_status = options.stream()->BlockHostUntilDone();
    TF_RETURN_IF_ERROR(result.status());
    TF_RETURN_IF_ERROR(block_status);
    return result;
  }

  // Compiled computation.
  std::unique_ptr<Executable> executable_;

  // Execution backend.
  Backend* backend_ = nullptr;

  // Options used to build the executable.
  const ExecutableBuildOptions build_options_;
};

// An XLA Client specialization for use when the client and service run in
// the same process.
class LocalClient : public Client {
 public:
  explicit LocalClient(LocalService* service)
      : Client(service), local_service_(service) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlocal_clientDTh mht_3(mht_3_v, 311, "", "./tensorflow/compiler/xla/client/local_client.h", "LocalClient");
}

  LocalClient(const LocalClient&) = delete;
  void operator=(const LocalClient&) = delete;

  // Build and return LocalExecutable objects (one per partition, as specified
  // by the build options). The executable is compiled using the given
  // XlaComputation, argument layouts and options.
  //
  // The given ExecutableBuildOptions overrides any values from XLA_FLAGS
  // environment variable.
  StatusOr<std::vector<std::unique_ptr<LocalExecutable>>> Compile(
      const XlaComputation& computation,
      const absl::Span<const Shape* const> argument_layouts,
      const ExecutableBuildOptions& options);

  // Same as Compile() above, but return AotCompilationResult objects (instead
  // of LocalExecutable objects), which can be persisted to later load
  // LocalExecutable(s) using the Load() method below.
  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(const XlaComputation& computation,
                     const absl::Span<const Shape* const> argument_layouts,
                     const ExecutableBuildOptions& options);

  // Return a LocalExecutable object loaded from a serialized
  // AotCompilationResult.
  StatusOr<std::unique_ptr<LocalExecutable>> Load(
      const std::string& serialized_aot_result,
      const ExecutableBuildOptions& options);

  // Copy the literal data to the device with the given ordinal and return as a
  // ScopedShapedBuffer. If non-null the given memory allocator is used for
  // device memory allocation. If null, the default memory allocator for the
  // device is used.
  StatusOr<ScopedShapedBuffer> LiteralToShapedBuffer(
      const LiteralSlice& literal, int device_ordinal,
      se::DeviceMemoryAllocator* allocator = nullptr);

  // Transfer the BorrowingLiteral to the device with the given ordinal.
  StatusOr<TransferToServerResponse> TransferToLocalServer(
      const ::xla::BorrowingLiteral& literal, int device_ordinal);

  // Copy the data from the device contained in the given ShapedBuffer and
  // return as a Literal.
  StatusOr<Literal> ShapedBufferToLiteral(const ShapedBuffer& shaped_buffer);

  // Converts a GlobalDataHandle into a pointer to a ShapedBuffer that's valid
  // as long as the handle is valid.
  StatusOr<const ShapedBuffer*> GlobalDataToShapedBuffer(
      const GlobalDataHandle& data, int replica_number);

  // Transfer the given literal to the infeed queue of the given device.
  // TODO(b/69670845): Remove the 'Local' from the name when LocalClient does
  // not inherit from Client and there is no possibility of confusion with
  // Client::TransferToInfeed.
  Status TransferToInfeedLocal(const LiteralSlice& literal, int device_ordinal);

  // Transfer and return a value from the outfeed of the given device. The
  // shape of the object to transfer is determined by `literal`'s shape.
  // TODO(b/69670845): Remove the 'Local' from the name when LocalClient does
  // not inherit from Client and there is no possibility of confusion with
  // Client::TransferFromOutfeed.
  Status TransferFromOutfeedLocal(int device_ordinal,
                                  MutableBorrowingLiteral literal);

  // Returns the device ordinal that corresponds to the given replica number.
  //
  // This returns an error if there is not a one-to-one correspondence of
  // replicas to device ordinals, but is useful as a short term mechanism for
  // the "easy" case where a single replica is a single device.
  StatusOr<int> ReplicaNumberToDeviceOrdinal(int replica_number);

  // Returns the platform that the underlying service targets.
  se::Platform* platform() const;

  // Returns the number of devices on the system of the service platform
  // type. Not all devices may be supported by the service (see
  // device_ordinal_supported method).
  int device_count() const;

  // Returns the default device ordinal that the service will run computations
  // on if no device ordinal is specified in execute options.
  int default_device_ordinal() const;

  // Returns whether the device with the given ordinal can be used by the
  // service to execute computations. Not all devices of a particular platform
  // may be usable by the service (eg, a GPU with insufficient CUDA compute
  // capability).
  bool device_ordinal_supported(int device_ordinal) const;

  // Returns the backend used to execute computations.
  const Backend& backend() const;
  Backend* mutable_backend();

 private:
  LocalService* local_service_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LOCAL_CLIENT_H_
