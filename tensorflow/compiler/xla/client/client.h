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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_CLIENT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTh() {
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
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// XLA service's client object -- wraps the service with convenience and
// lifetime-oriented methods.
class Client {
 public:
  explicit Client(ServiceInterface* stub);
  virtual ~Client();

  // Compile the computation with the given argument shapes and returns the
  // handle to the compiled executable. The compiled executable is cached on the
  // service, and the returned handle can be used for execution without
  // re-compile.
  // * The shape and layout of the arguments being executed with will affect how
  //   the computation is compiled. If argument_shapes is empty, the parameters'
  //   shape and layout will be used in the compilation.
  // * If execution_options is not nullptr, these options are passed to the
  //   service to affect how it compiles our computation.  (The pointer does not
  //   need to live beyond this call.)
  // * If execution_options.device_handles should be empty. If you need
  //   non-empty device handles, call 'Execute' instead.
  //
  // TODO(b/122731460): This call caches the resulting Executable in the Service
  // *forever*.  If you're only going to run the computation once, you may want
  // to call the Execute(const XlaComputation&) overload.  If you're going to
  // run the computation more than once but you want control over when the
  // Executable is unloaded, use the LocalClient API.
  StatusOr<ExecutionHandle> Compile(
      const XlaComputation& computation,
      absl::Span<const Shape> argument_shapes,
      const ExecutionOptions* execution_options = nullptr);

  // Executes the compiled executable for the given handle with the given
  // arguments and returns the global data that was produced from the execution.
  // * If execution_profile is not nullptr then the pointed-to ExecutionProfile
  //   will be filled with profile data from the execution.
  StatusOr<std::unique_ptr<GlobalData>> Execute(
      const ExecutionHandle& handle, absl::Span<GlobalData* const> arguments,
      ExecutionProfile* execution_profile = nullptr);

  // Executes the computation with the given arguments and returns the global
  // data that was produced from the execution.
  // * If execution_options is not nullptr, these options are passed to the
  //   service to affect how it compiles our computation.  (The pointer does not
  //   need to live beyond this call.)
  // * If execution_options.device_handles is not empty, the computation is
  //   executed on the devices associated with the handles by partitioning the
  //   computation based on the attached sharding attributes. Otherwise, a
  //   device is chosen by the service.
  // * If execution_profile is not nullptr then the pointed-to ExecutionProfile
  //   will be filled with profile data from the execution.
  //
  // TODO(b/122731460): The given computation is compiled and then thrown away
  // immediately after it's run.  If you want control over how long the
  // resulting Executable lives, use the LocalClient API.
  StatusOr<std::unique_ptr<GlobalData>> Execute(
      const XlaComputation& computation,
      absl::Span<GlobalData* const> arguments,
      const ExecutionOptions* execution_options = nullptr,
      ExecutionProfile* execution_profile = nullptr);

  // A struct to represent a computation instance to be executed.
  // * If execution_options.device_handles is not empty, the computation is
  //   executed on the devices associated with the handles by partitioning the
  //   computation based on the attached sharding attributes. Otherwise, a
  //   device is chosen by the service.
  struct XlaComputationInstance {
    const XlaComputation& computation;
    std::vector<GlobalData*> arguments;
    ExecutionOptions execution_options;
    ExecutionProfile* execution_profile;

    XlaComputationInstance(const XlaComputation& computation,
                           std::vector<GlobalData*> arguments,
                           ExecutionOptions execution_options,
                           ExecutionProfile* execution_profile)
        : computation(computation),
          arguments(std::move(arguments)),
          execution_options(execution_options),
          execution_profile(execution_profile) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTh mht_0(mht_0_v, 281, "", "./tensorflow/compiler/xla/client/client.h", "XlaComputationInstance");
}
  };

  // Executes a list XlaComputationInstances and returns global data produced
  // from each computation.
  //
  StatusOr<std::vector<std::unique_ptr<GlobalData>>> ExecuteParallel(
      absl::Span<const XlaComputationInstance> computations);

  // Requests device_count device handles available on the target. The returned
  // device handles are used to specify the devices to execute the computations
  // (see ExecuteParallel) or to transfer data (see TransferToServer or
  // TransferToInfeed).
  StatusOr<std::vector<DeviceHandle>> GetDeviceHandles(int64_t device_count);

  // Transfer the global data provided to this client process, which is
  // returned in the provided literal. Use sparingly to avoid transfer
  // overheads.
  //
  // If shape_with_layout is not nullptr, it points to a shape whose layout will
  // be the layout of the returned literal.
  StatusOr<Literal> Transfer(const GlobalData& data,
                             const Shape* shape_with_layout = nullptr);

  // Transfer the given literal to the server. This allocates memory on the
  // device and copies the literal's contents over. Returns a global data handle
  // that can be used to refer to this value from the client.
  //
  // If device_handle is not nullptr, data is transferred to the associated
  // device (and its replicas if replication is enabled). Otherwise, data is
  // transferred to the default device (and its replicas).
  StatusOr<std::unique_ptr<GlobalData>> TransferToServer(
      const LiteralSlice& literal, const DeviceHandle* device_handle = nullptr);

  // Transfer the given literal to the Infeed interface of the device.
  //
  // device_handle and replica_id together specify a particular device; a device
  // assigned for the given replica_id among the replicas that the given device
  // handle belongs to.
  Status TransferToInfeed(const LiteralSlice& literal, int64_t replica_id = 0,
                          const DeviceHandle* device_handle = nullptr);

  // Transfers from the Outfeed of the device.
  //
  // device_handle and replica_id together specify a particular device; a device
  // assigned for the given replica_id among the replicas that the given device
  // handle belongs to.
  StatusOr<Literal> TransferFromOutfeed(
      const Shape* shape_with_layout, int64_t replica_id = 0,
      const DeviceHandle* device_handle = nullptr);

  // Resets the device, clearing all existing state on the device.
  Status ResetDevice();

  // Executes the computation with the given arguments and transfers the result
  // to the client as a literal. Parameters are defined the same as for
  // Execute() and Transfer().
  StatusOr<Literal> ExecuteAndTransfer(
      const XlaComputation& computation,
      absl::Span<GlobalData* const> arguments,
      const ExecutionOptions* execution_options = nullptr,
      ExecutionProfile* execution_profile = nullptr);

  // Computes the value of the given computation using a non-optimized
  // interpreter on the host.
  //
  // The computation must not depend on any parameters, or on stateful operators
  // such as `RngNormal` or `Infeed`.
  //
  // This functionality can be useful when translating a computation into XLA
  // where something that looked dynamic is required by XLA to be specified as a
  // constant. E.g. the source computation (outside of XLA) may include a
  // dynamic computation of the shape of something and ComputeConstant lets you
  // determine what the value of that computation is in the case where the value
  // can be determined at compile time.
  //
  // If output_layout is non-null, then the output of the computation will be
  // stored using that layout.
  StatusOr<Literal> ComputeConstant(
      const XlaComputation& computation,
      const Layout* output_layout = nullptr) const;

  // Unregister the memory for the given GlobalData on the device.
  Status Unregister(const GlobalData& data);

  // Returns a vector of global data handles that point to the tuple elements.
  StatusOr<std::vector<std::unique_ptr<GlobalData>>> DeconstructTuple(
      const GlobalData& data);

  // Retrieves the statistics of the given computation.
  StatusOr<ComputationStats> GetComputationStats(
      const XlaComputation& computation,
      const DebugOptions& debug_options) const;

  // Returns the Shape of the given array specified by 'data'. The shape
  // includes the Layout of the array as it is stored on the service.
  StatusOr<Shape> GetShape(const GlobalData& data);

  // As above, but returns the shape of the provided computation (parameter
  // types/names and return type).
  StatusOr<std::unique_ptr<ProgramShape>> GetComputationShape(
      const XlaComputation& computation);

  // Creates a channel handle that can be used to transfer data between two
  // computations on different devices via a pair of Send and Recv instructions.
  StatusOr<ChannelHandle> CreateChannelHandle();

  // Create a channel for communicating with the host via a SendtoHost or
  // RecvFromHost operation.
  StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle();
  StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle();

  StatusOr<XlaComputation> LoadSnapshot(const HloSnapshot& module);

  ServiceInterface* stub() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTh mht_1(mht_1_v, 398, "", "./tensorflow/compiler/xla/client/client.h", "stub");
 return stub_; }

 private:
  // Returns the execution statistics (e.g., gflop/s) as a string from the
  // ExecutionProfile returned from an execution of the computation.
  StatusOr<std::string> ExecutionStatsAsString(
      const XlaComputation& computation, const ExecutionProfile& profile);

  StatusOr<ChannelHandle> CreateChannelHandleByType(
      ChannelHandle::ChannelType type);

  ServiceInterface* stub_;  // Stub that this client is connected on.

  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_CLIENT_H_
