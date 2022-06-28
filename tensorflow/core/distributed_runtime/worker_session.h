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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_SESSION_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_SESSION_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTh() {
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


#include <string>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/cluster_function_library_runtime.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"

namespace tensorflow {

class ClusterFunctionLibraryRuntime;
class GraphMgr;
class WorkerCacheInterface;

// WorkerSession encapsulates all of the state relating to a given session.
class WorkerSession {
 public:
  // Collection of local devices. These devices are typically
  // RenamedDevices in all except the SessionMgr.legacy_session_ and
  // sessions created with `isolate_session_state == false`. In the
  // those cases, this method returns a pointer to a borrowed
  // DeviceMgr (typically the `worker_env.device_mgr`).
  DeviceMgr* device_mgr() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTh mht_0(mht_0_v, 209, "", "./tensorflow/core/distributed_runtime/worker_session.h", "device_mgr");

    return device_mgr_ ? device_mgr_.get() : borrowed_device_mgr_;
  }

  DynamicDeviceMgr* remote_device_mgr() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTh mht_1(mht_1_v, 216, "", "./tensorflow/core/distributed_runtime/worker_session.h", "remote_device_mgr");
 return remote_device_mgr_.get(); }

  const string& session_name() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTh mht_2(mht_2_v, 221, "", "./tensorflow/core/distributed_runtime/worker_session.h", "session_name");
 return session_name_; }
  const string& worker_name() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTh mht_3(mht_3_v, 225, "", "./tensorflow/core/distributed_runtime/worker_session.h", "worker_name");
 return worker_name_; }

  WorkerCacheInterface* worker_cache() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTh mht_4(mht_4_v, 230, "", "./tensorflow/core/distributed_runtime/worker_session.h", "worker_cache");

    tf_shared_lock l(worker_session_state_mu_);
    return worker_cache_.get();
  }
  GraphMgr* graph_mgr() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTh mht_5(mht_5_v, 237, "", "./tensorflow/core/distributed_runtime/worker_session.h", "graph_mgr");
 return graph_mgr_.get(); }

  ClusterFunctionLibraryRuntime* cluster_flr() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSworker_sessionDTh mht_6(mht_6_v, 242, "", "./tensorflow/core/distributed_runtime/worker_session.h", "cluster_flr");

    return cluster_flr_.get();
  }

  WorkerSession(const string& session_name, const string& worker_name,
                std::unique_ptr<WorkerCacheInterface> worker_cache,
                std::unique_ptr<DeviceMgr> device_mgr,
                std::unique_ptr<GraphMgr> graph_mgr,
                std::unique_ptr<DynamicDeviceMgr> remote_device_mgr);

  static std::shared_ptr<WorkerSession> CreateWithBorrowedDeviceMgr(
      const string& session_name, const string& worker_name,
      std::unique_ptr<WorkerCacheInterface> worker_cache,
      DeviceMgr* borrowed_device_mgr, std::unique_ptr<GraphMgr> graph_mgr,
      std::unique_ptr<DynamicDeviceMgr> remote_device_mgr);

  // In the eager runtime we allow WorkerSession to be updated, where the
  // worker cache will be recreated. If WorkerSession upate is expected and a
  // worker in the cache is used in RPCs, the caller should hold a shared
  // pointer to avoid the workers getting deleted.
  std::shared_ptr<WorkerCacheInterface> GetSharedWorkerCache() {
    tf_shared_lock l(worker_session_state_mu_);
    return worker_cache_;
  }

  // Update an existing worker session with new set of remote workers and
  // devices. Added devices will be owned by the worker session, and removed
  // devices will be freed by their names.
  Status UpdateWorkerCacheAndDevices(
      std::unique_ptr<WorkerCacheInterface> new_worker_cache,
      std::vector<std::unique_ptr<Device>> added_remote_devices,
      const std::vector<Device*>& removed_remote_devices);

  ~WorkerSession();

 private:
  WorkerSession(const string& session_name, const string& worker_name,
                std::unique_ptr<WorkerCacheInterface> worker_cache,
                DeviceMgr* borrowed_device_mgr,
                std::unique_ptr<GraphMgr> graph_mgr,
                std::unique_ptr<DynamicDeviceMgr> remote_device_mgr);

  // The name of the session.
  const string session_name_;

  // The name of the worker. E.g., /job:mnist/replica:0/task:1.
  const string worker_name_;

  mutable mutex worker_session_state_mu_;
  // Object from which WorkerInterface instances can be obtained.
  std::shared_ptr<WorkerCacheInterface> worker_cache_
      TF_GUARDED_BY(worker_session_state_mu_);

  // graph_mgr keeps track of the registered graphs of this session.
  //
  // Note: graph_mgr must be deleted before rendezvous_mgr!
  // Note: graph_mgr must be deleted before device_mgr!
  const std::unique_ptr<GraphMgr> graph_mgr_;

  std::unique_ptr<ClusterFunctionLibraryRuntime> cluster_flr_;

  const std::unique_ptr<DeviceMgr> device_mgr_;
  DeviceMgr* const borrowed_device_mgr_;  // Not owned.
  std::unique_ptr<DynamicDeviceMgr> remote_device_mgr_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_SESSION_H_
