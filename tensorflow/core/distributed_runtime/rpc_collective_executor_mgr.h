/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_COLLECTIVE_EXECUTOR_MGR_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_COLLECTIVE_EXECUTOR_MGR_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTh() {
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


#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/framework/collective.h"

namespace tensorflow {
class CollectiveParamResolverDistributed;
class ConfigProto;
class DeviceMgr;
class DeviceResolverDistributed;
class WorkerCacheInterface;
class StepSequenceRequest;
class StepSequenceResponse;

// An implementation of CollectiveExecutorMgr for a distributed environment
// that uses WorkerInterface::RecvBufAsync to route data transfers over RPCs.
//
// In some execution environments it may be possible to implement a
// higher-performance solution and use it in place of this class.
class RpcCollectiveExecutorMgr : public CollectiveExecutorMgr {
 public:
  RpcCollectiveExecutorMgr(
      const ConfigProto& config, const DeviceMgr* dev_mgr,
      std::unique_ptr<DeviceResolverDistributed> dev_resolver,
      std::unique_ptr<CollectiveParamResolverDistributed> param_resolver,
      std::unique_ptr<NcclCommunicatorInterface> nccl_communicator,
      WorkerCacheInterface* worker_cache, const string& task_name);

  virtual ~RpcCollectiveExecutorMgr();

  // This function should only be called at the group_leader, by an RPC.
  // Other needs for StepIds should be satisfied by NextStepId.
  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            const StatusCallback& done) override;

  void RefreshStepIdSequenceAsync(int64_t graph_key,
                                  const StatusCallback& done) override;

  int64_t NextStepId(int64_t graph_key) override;

  void RetireStepId(int64_t graph_key, int64_t step_id) override;

 protected:
  virtual CollectiveExecutor* Create(int64_t step_id) override;

  WorkerCacheInterface* const worker_cache_;  // Not owned.
  const string task_name_;
  string group_leader_;
  friend class RpcCollectiveExecutorMgrTest;

 private:
  Status UpdateStepSequences(const GetStepSequenceResponse& resp);

  // This class maintains the step_id sequencing for a single
  // collective_graph_key.
  struct GraphKeySequence {
    explicit GraphKeySequence(int64_t k)
        : graph_key_(k), next_step_id_(CollectiveExecutor::kInvalidId) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTh mht_0(mht_0_v, 243, "", "./tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h", "GraphKeySequence");
}

    const int64_t graph_key_;
    int64_t next_step_id_;
  };

  mutex sequence_mu_;
  gtl::FlatMap<int64_t, GraphKeySequence*> sequence_table_
      TF_GUARDED_BY(sequence_mu_);
};

// Creates a distributed CollectiveExecutorMgr with production implementations
// of each components. Cases that need to inject other implementations of these
// components should call CollectiveExecutorMgr constructor directly.
std::unique_ptr<RpcCollectiveExecutorMgr> CreateProdRpcCollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* device_mgr,
    std::unique_ptr<NcclCommunicatorInterface> nccl_communicator,
    WorkerCacheInterface* worker_cache, const string& default_worker_name);

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_COLLECTIVE_EXECUTOR_MGR_H_
