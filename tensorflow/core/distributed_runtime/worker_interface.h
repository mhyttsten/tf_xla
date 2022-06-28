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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh() {
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

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

// Status callback.
typedef std::function<void(const Status&)> StatusCallback;

// Custom decoder for a response to RecvTensorAsync.
class TensorResponse;

// Interface for talking with the TensorFlow Worker service.
class WorkerInterface {
 public:
  virtual void GetStatusAsync(CallOptions* opts,
                              const GetStatusRequest* request,
                              GetStatusResponse* response, bool fail_fast,
                              StatusCallback done) = 0;

  virtual void CreateWorkerSessionAsync(
      const CreateWorkerSessionRequest* request,
      CreateWorkerSessionResponse* response, StatusCallback done) = 0;

  virtual void DeleteWorkerSessionAsync(
      CallOptions* opts, const DeleteWorkerSessionRequest* request,
      DeleteWorkerSessionResponse* response, StatusCallback done) = 0;

  virtual void RegisterGraphAsync(const RegisterGraphRequest* request,
                                  RegisterGraphResponse* response,
                                  StatusCallback done) = 0;

  virtual void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                                    DeregisterGraphResponse* response,
                                    StatusCallback done) = 0;

  virtual void RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request,
                             MutableRunGraphResponseWrapper* response,
                             StatusCallback done) = 0;

  virtual void RunGraphAsync(CallOptions* opts, const RunGraphRequest* request,
                             RunGraphResponse* response, StatusCallback done) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_0(mht_0_v, 234, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "RunGraphAsync");

    RunGraphRequestWrapper* wrapped_request = new ProtoRunGraphRequest(request);
    MutableRunGraphResponseWrapper* wrapped_response =
        new NonOwnedProtoRunGraphResponse(response);
    RunGraphAsync(opts, wrapped_request, wrapped_response,
                  [wrapped_request, wrapped_response,
                   done = std::move(done)](const Status& s) {
                    done(s);
                    delete wrapped_request;
                    delete wrapped_response;
                  });
  }

  // Returns a request object for use in calls to
  // `RunGraphAsync()`. Ownership is transferred to the caller.
  //
  // The message returned from this method must only be used in a
  // `RunGraph()` call on the same `WorkerInterface` instance.
  virtual MutableRunGraphRequestWrapper* CreateRunGraphRequest() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_1(mht_1_v, 255, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "CreateRunGraphRequest");

    return new MutableProtoRunGraphRequest;
  }

  // Returns a response object for use in calls to
  // `RunGraphAsync()`. Ownership is transferred to the caller.
  //
  // The message returned from this method must only be used in a
  // `RunGraph()` call on the same `WorkerInterface` instance.
  virtual MutableRunGraphResponseWrapper* CreateRunGraphResponse() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_2(mht_2_v, 267, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "CreateRunGraphResponse");

    return new OwnedProtoRunGraphResponse;
  }

  virtual void CleanupGraphAsync(const CleanupGraphRequest* request,
                                 CleanupGraphResponse* response,
                                 StatusCallback done) = 0;

  virtual void CleanupAllAsync(const CleanupAllRequest* request,
                               CleanupAllResponse* response,
                               StatusCallback done) = 0;

  virtual void RecvTensorAsync(CallOptions* opts,
                               const RecvTensorRequest* request,
                               TensorResponse* response,
                               StatusCallback done) = 0;

  virtual void LoggingAsync(const LoggingRequest* request,
                            LoggingResponse* response, StatusCallback done) = 0;

  virtual void TracingAsync(const TracingRequest* request,
                            TracingResponse* response, StatusCallback done) = 0;

  virtual void RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                            RecvBufResponse* response, StatusCallback done) = 0;

  virtual void CompleteGroupAsync(CallOptions* opts,
                                  const CompleteGroupRequest* request,
                                  CompleteGroupResponse* response,
                                  StatusCallback done) = 0;

  virtual void CompleteInstanceAsync(CallOptions* ops,
                                     const CompleteInstanceRequest* request,
                                     CompleteInstanceResponse* response,
                                     StatusCallback done) = 0;

  virtual void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                                    GetStepSequenceResponse* response,
                                    StatusCallback done) = 0;

  Status GetStatus(const GetStatusRequest* request,
                   GetStatusResponse* response) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_3(mht_3_v, 311, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "GetStatus");

    Status ret;
    Notification n;
    GetStatusAsync(/*opts=*/nullptr, request, response, /*fail_fast=*/true,
                   [&ret, &n](const Status& s) {
                     ret = s;
                     n.Notify();
                   });
    n.WaitForNotification();
    return ret;
  }

  Status CreateWorkerSession(const CreateWorkerSessionRequest* request,
                             CreateWorkerSessionResponse* response) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_4(mht_4_v, 327, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "CreateWorkerSession");

    return CallAndWait(&ME::CreateWorkerSessionAsync, request, response);
  }

  Status DeleteWorkerSession(const DeleteWorkerSessionRequest* request,
                             DeleteWorkerSessionResponse* response) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_5(mht_5_v, 335, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "DeleteWorkerSession");

    return CallAndWaitWithOptions(&ME::DeleteWorkerSessionAsync, request,
                                  response);
  }

  Status RegisterGraph(const RegisterGraphRequest* request,
                       RegisterGraphResponse* response) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_6(mht_6_v, 344, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "RegisterGraph");

    return CallAndWait(&ME::RegisterGraphAsync, request, response);
  }

  Status DeregisterGraph(const DeregisterGraphRequest* request,
                         DeregisterGraphResponse* response) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_7(mht_7_v, 352, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "DeregisterGraph");

    return CallAndWait(&ME::DeregisterGraphAsync, request, response);
  }

  Status CleanupGraph(const CleanupGraphRequest* request,
                      CleanupGraphResponse* response) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_8(mht_8_v, 360, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "CleanupGraph");

    return CallAndWait(&ME::CleanupGraphAsync, request, response);
  }

  Status CleanupAll(const CleanupAllRequest* request,
                    CleanupAllResponse* response) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_9(mht_9_v, 368, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "CleanupAll");

    return CallAndWait(&ME::CleanupAllAsync, request, response);
  }

  Status Logging(const LoggingRequest* request, LoggingResponse* response) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_10(mht_10_v, 375, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "Logging");

    return CallAndWait(&ME::LoggingAsync, request, response);
  }

  Status Tracing(const TracingRequest* request, TracingResponse* response) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_11(mht_11_v, 382, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "Tracing");

    return CallAndWait(&ME::TracingAsync, request, response);
  }

  Status GetStepSequence(const GetStepSequenceRequest* request,
                         GetStepSequenceResponse* response) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_12(mht_12_v, 390, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "GetStepSequence");

    return CallAndWait(&ME::GetStepSequenceAsync, request, response);
  }

 protected:
  // Instances of WorkerInterface must be deleted by a call to
  // WorkerCacheInterface::ReleaseWorker().
  virtual ~WorkerInterface() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_13(mht_13_v, 400, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "~WorkerInterface");
}
  friend class WorkerCacheInterface;

  // NOTE: This should only be called by implementations of this
  // interface whose CreateRunGraphResponse() method returns a
  // proto-based wrappers for the RunGraphResponse message.
  RunGraphResponse* get_proto_from_wrapper(
      MutableRunGraphResponseWrapper* wrapper) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_interfaceDTh mht_14(mht_14_v, 410, "", "./tensorflow/core/distributed_runtime/worker_interface.h", "get_proto_from_wrapper");

    return wrapper->get_proto();
  }

 private:
  typedef WorkerInterface ME;

  template <typename Method, typename Req, typename Resp>
  Status CallAndWait(Method func, const Req* req, Resp* resp) {
    Status ret;
    Notification n;
    (this->*func)(req, resp, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }

  template <typename Method, typename Req, typename Resp>
  Status CallAndWaitWithOptions(Method func, const Req* req, Resp* resp) {
    CallOptions call_opts;
    Status ret;
    Notification n;
    (this->*func)(&call_opts, req, resp, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_
