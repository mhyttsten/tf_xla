/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_INTERFACE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_INTERFACE_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmaster_interfaceDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmaster_interfaceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmaster_interfaceDTh() {
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


#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

// Abstract interface for communicating with the TensorFlow Master service.
//
// This interface supports both RPC-based master implementations, and
// in-process master implementations that do not require an RPC
// roundtrip.
class MasterInterface {
 public:
  virtual ~MasterInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmaster_interfaceDTh mht_0(mht_0_v, 204, "", "./tensorflow/core/distributed_runtime/master_interface.h", "~MasterInterface");
}
  virtual Status CreateSession(CallOptions* call_options,
                               const CreateSessionRequest* request,
                               CreateSessionResponse* response) = 0;

  virtual Status ExtendSession(CallOptions* call_options,
                               const ExtendSessionRequest* request,
                               ExtendSessionResponse* response) = 0;

  virtual Status PartialRunSetup(CallOptions* call_options,
                                 const PartialRunSetupRequest* request,
                                 PartialRunSetupResponse* response) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmaster_interfaceDTh mht_1(mht_1_v, 218, "", "./tensorflow/core/distributed_runtime/master_interface.h", "PartialRunSetup");

    return errors::Unimplemented("Partial run not implemented for this master");
  }

  virtual Status RunStep(CallOptions* call_options,
                         RunStepRequestWrapper* request,
                         MutableRunStepResponseWrapper* response) = 0;

  virtual Status RunStep(CallOptions* call_options,
                         const RunStepRequest* request,
                         RunStepResponse* response) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmaster_interfaceDTh mht_2(mht_2_v, 231, "", "./tensorflow/core/distributed_runtime/master_interface.h", "RunStep");

    std::unique_ptr<RunStepRequestWrapper> wrapped_request(
        new ProtoRunStepRequest(request));
    std::unique_ptr<MutableRunStepResponseWrapper> wrapped_response(
        new NonOwnedProtoRunStepResponse(response));
    return RunStep(call_options, wrapped_request.get(), wrapped_response.get());
  }

  // Returns a request object for use in calls to
  // `RunStep()`. Ownership is transferred to the caller.
  //
  // The message returned from this method must only be used in a
  // `RunStep()` call on the same `MasterInterface` instance.
  virtual MutableRunStepRequestWrapper* CreateRunStepRequest() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmaster_interfaceDTh mht_3(mht_3_v, 247, "", "./tensorflow/core/distributed_runtime/master_interface.h", "CreateRunStepRequest");

    MutableProtoRunStepRequest* ret = new MutableProtoRunStepRequest;
    ret->request_.set_request_id(GetUniqueRequestId());
    return ret;
  }

  // Returns a response object for use in calls to
  // `RunStep()`. Ownership is transferred to the caller.
  //
  // The message returned from this method must only be used in a
  // `RunStep()` call on the same `MasterInterface` instance.
  virtual MutableRunStepResponseWrapper* CreateRunStepResponse() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmaster_interfaceDTh mht_4(mht_4_v, 261, "", "./tensorflow/core/distributed_runtime/master_interface.h", "CreateRunStepResponse");

    return new OwnedProtoRunStepResponse;
  }

  virtual Status CloseSession(CallOptions* call_options,
                              const CloseSessionRequest* request,
                              CloseSessionResponse* response) = 0;

  virtual Status ListDevices(CallOptions* call_options,
                             const ListDevicesRequest* request,
                             ListDevicesResponse* response) = 0;

  virtual Status Reset(CallOptions* call_options, const ResetRequest* request,
                       ResetResponse* response) = 0;

  virtual Status MakeCallable(CallOptions* call_options,
                              const MakeCallableRequest* request,
                              MakeCallableResponse* response) = 0;
  virtual Status RunCallable(CallOptions* call_options,
                             const RunCallableRequest* request,
                             RunCallableResponse* response) = 0;
  virtual Status ReleaseCallable(CallOptions* call_options,
                                 const ReleaseCallableRequest* request,
                                 ReleaseCallableResponse* response) = 0;

 protected:
  // NOTE: This should only be called by implementations of this
  // interface whose CreateRunStepResponse() method returns a
  // proto-based wrappers for the RunStepResponse message.
  RunStepResponse* get_proto_from_wrapper(
      MutableRunStepResponseWrapper* wrapper) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmaster_interfaceDTh mht_5(mht_5_v, 294, "", "./tensorflow/core/distributed_runtime/master_interface.h", "get_proto_from_wrapper");

    return wrapper->get_proto();
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_INTERFACE_H_
