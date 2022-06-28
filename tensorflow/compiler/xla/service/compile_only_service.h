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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COMPILE_ONLY_SERVICE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COMPILE_ONLY_SERVICE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTh() {
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


#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

// An XLA Service specialization for ahead-of-time compilation.  This only
// instantiates a Compiler object for the relevant platform; it does not
// instantiate or require an execution backend.
class CompileOnlyService : public Service {
 public:
  // Factory for creating a CompileOnlyService. The parameter platform is the
  // platform that the service should target. If platform is null then the
  // default platform is used.
  static StatusOr<std::unique_ptr<CompileOnlyService>> NewService(
      se::Platform* platform);
  static StatusOr<std::unique_ptr<CompileOnlyService>> NewService(
      const ServiceOptions& options);

  // A description of a xla computation to compile using CompileAheadOfTime.
  struct AotXlaComputationInstance {
    HloModuleProto computation;
    std::vector<const Shape*> argument_layouts;
    const Shape* result_layout = nullptr;
  };

  // Compiles a list of xla computations for ahead-of-time execution.  This is
  // intended for use in static compilation.  See
  // |CompileOnlyClient::CompileAheadOfTime| for additional details.
  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(
      const absl::Span<const AotXlaComputationInstance> computations,
      const AotCompilationOptions& options);

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(
      const absl::Span<const AotXlaComputationInstance> computations,
      const AotCompilationOptions& options,
      std::unique_ptr<AotCompilationMetadata>* metadata);

  Status GetDeviceHandles(const GetDeviceHandlesRequest* arg,
                          GetDeviceHandlesResponse* result) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTh mht_0(mht_0_v, 232, "", "./tensorflow/compiler/xla/service/compile_only_service.h", "GetDeviceHandles");

    return Unimplemented("CompileOnlyService does not support devices.");
  }
  Status WaitForExecution(const WaitForExecutionRequest* arg,
                          WaitForExecutionResponse* result) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTh mht_1(mht_1_v, 239, "", "./tensorflow/compiler/xla/service/compile_only_service.h", "WaitForExecution");

    return Unimplemented("CompileOnlyService does not support execution.");
  }
  Status TransferToServer(const TransferToServerRequest* arg,
                          TransferToServerResponse* result) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTh mht_2(mht_2_v, 246, "", "./tensorflow/compiler/xla/service/compile_only_service.h", "TransferToServer");

    return Unimplemented(
        "CompileOnlyService does not support device data transfers.");
  }
  Status TransferToInfeed(const TransferToInfeedRequest* arg,
                          TransferToInfeedResponse* result) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTh mht_3(mht_3_v, 254, "", "./tensorflow/compiler/xla/service/compile_only_service.h", "TransferToInfeed");

    return Unimplemented(
        "CompileOnlyService does not support device data transfers.");
  }
  Status TransferFromOutfeed(const TransferFromOutfeedRequest* arg,
                             TransferFromOutfeedResponse* result) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTh mht_4(mht_4_v, 262, "", "./tensorflow/compiler/xla/service/compile_only_service.h", "TransferFromOutfeed");

    return Unimplemented(
        "CompileOnlyService does not support device data transfers.");
  }
  Status ResetDevice(const ResetDeviceRequest* arg,
                     ResetDeviceResponse* result) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompile_only_serviceDTh mht_5(mht_5_v, 270, "", "./tensorflow/compiler/xla/service/compile_only_service.h", "ResetDevice");

    return Unimplemented("CompileOnlyService does not support devices.");
  }

 private:
  explicit CompileOnlyService(const ServiceOptions& options,
                              Compiler* compiler);
  CompileOnlyService(const CompileOnlyService&) = delete;
  void operator=(const CompileOnlyService&) = delete;

  // The compiler for the target platform.  This is included in place of
  // the Service::execute_backend_'s compiler, since execute_backend_ is a
  // nullptr in CompileOnlyService.
  Compiler* compiler_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPILE_ONLY_SERVICE_H_
