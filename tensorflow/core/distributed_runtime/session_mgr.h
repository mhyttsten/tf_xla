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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSsession_mgrDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSsession_mgrDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSsession_mgrDTh() {
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
#include <string>

#include "tensorflow/core/distributed_runtime/coordination/coordination_service.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

class WorkerCacheInterface;
struct WorkerEnv;

// SessionMgr keeps track of information related to a given session.
//
// SessionMgr runs on the workers.
//
// SessionMgr is threadsafe.
class SessionMgr {
 public:
  typedef std::function<Status(const ServerDef&, WorkerCacheInterface**)>
      WorkerCacheFactory;

  explicit SessionMgr(
      WorkerEnv* worker_env, const std::string& default_worker_name,
      std::unique_ptr<WorkerCacheInterface> default_worker_cache,
      WorkerCacheFactory worker_cache_factory);
  ~SessionMgr() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSsession_mgrDTh mht_0(mht_0_v, 219, "", "./tensorflow/core/distributed_runtime/session_mgr.h", "~SessionMgr");
}

  // Allocates state for a new session.
  Status CreateSession(
      const std::string& session, const ServerDef& server_def,
      bool isolate_session_state,
      StatusCallback coordination_error_callback = [](Status s) {
        LOG(ERROR) << "Coordination agent is set to error: " << s;
      });
  Status CreateSession(
      const std::string& session, const ServerDef& server_def,
      const protobuf::RepeatedPtrField<DeviceAttributes>& device_attributes,
      bool isolate_session_state);

  // Create WorkerSession from the master with the given `master_task` and
  // `master_incarnation`. We first look for existing WorkerSessions associated
  // with the specified master task. If there are sessions created by the same
  // master but with a different incarnation, it indicates that the remote
  // master has restarted before deleting the sessions on worker. When it
  // happens, old sessions associated with the master will be automatically
  // removed before the new session is created.
  Status CreateSession(
      const std::string& session, const ServerDef& server_def,
      const protobuf::RepeatedPtrField<DeviceAttributes>& device_attributes,
      bool isolate_session_state, std::string master_task,
      int64_t master_incarnation,
      StatusCallback coordination_error_callback = [](Status s) {
        LOG(ERROR) << "Coordination agent is set to error: " << s;
      });

  void ResetDefaultWorkerCache(WorkerCacheInterface* worker_cache);

  // Updates state (worker cache, devices) of worker session identified by
  // session name (`session`) based on a new server_def and set of devices.
  Status UpdateSession(const std::string& session, const ServerDef& server_def,
                       const protobuf::RepeatedPtrField<DeviceAttributes>&
                           cluster_device_attributes);

  // Locates the worker session for a given session handle
  Status WorkerSessionForSession(const std::string& session_handle,
                                 std::shared_ptr<WorkerSession>* out_session);
  std::shared_ptr<WorkerSession> LegacySession();

  Status DeleteSession(const std::string& session);

  // Provides access to the coordination service. This method should only be
  // called after the agent has been initialized during session creation, or an
  // invalid nullptr is returned. Note: the agent is thread-safe and mutable.
  CoordinationServiceAgent* GetCoordinationServiceAgent();

  static std::string WorkerNameFromServerDef(const ServerDef& server_def);

  void SetLogging(bool active);

  void RetrieveLogs(int64_t step_id, LoggingResponse* response);

  void ClearLogs();

  // Agent should be torn down before service as it needs to disconnect first.
  void TeardownCoordinationServiceAgent();
  void TeardownCoordinationService();

 private:
  WorkerEnv* const worker_env_;  // Not owned.

  // A note about destruction:
  // We must delete graph_mgr before device_mgr, due to shared
  // ownership of OpKernels in the executors. (The graph_mgr will
  // free all stateless OpKernels, and pass over borrowed stateful
  // OpKernels, which are also held in their respective devices'
  // OpSegments.)
  //
  // legacy_session_ owns the worker_env_.device_mgr, and so we must ensure
  // that sessions_'s WorkerSessions are deleted (which do not own the
  // underlying devices, but instead own RenamedDevices) before
  // legacy_session_ is deleted. Further, we must ensure that WorkerSession's
  // device_mgr is deleted after WorkerSession's graph_mgr.

  std::unique_ptr<WorkerCacheInterface> default_worker_cache_;
  std::shared_ptr<WorkerSession> legacy_session_;
  std::unique_ptr<CoordinationServiceInterface> coordination_service_;
  std::unique_ptr<CoordinationServiceAgent> coordination_service_agent_;

  bool is_logging_active_ = false;

  const WorkerCacheFactory worker_cache_factory_;

  Status WorkerSessionForSessionLocked(
      const std::string& session_handle,
      std::shared_ptr<WorkerSession>* out_session)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutex mu_;
  // A map from session identifier to internal session structure.
  std::map<std::string, std::shared_ptr<WorkerSession>> sessions_
      TF_GUARDED_BY(mu_);

  // Incarnation and WorkerSession handle associated with a master task.
  struct MasterAssociatedSession {
    const int64_t master_incarnation;
    const std::string session_handle;
  };
  // A map from master task name to its associated worker sessions.
  std::unordered_multimap<std::string, MasterAssociatedSession>
      master_to_associated_sessions_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_H_
