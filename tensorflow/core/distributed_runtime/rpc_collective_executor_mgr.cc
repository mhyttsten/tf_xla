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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc() {
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
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/collective_rma_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

RpcCollectiveExecutorMgr::RpcCollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* dev_mgr,
    std::unique_ptr<DeviceResolverDistributed> dev_resolver,
    std::unique_ptr<CollectiveParamResolverDistributed> param_resolver,
    std::unique_ptr<NcclCommunicatorInterface> nccl_communicator,
    WorkerCacheInterface* worker_cache, const string& task_name)
    : CollectiveExecutorMgr(config, dev_mgr, std::move(dev_resolver),
                            std::move(param_resolver),
                            std::move(nccl_communicator)),
      worker_cache_(worker_cache),
      task_name_(task_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("task_name: \"" + task_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.cc", "RpcCollectiveExecutorMgr::RpcCollectiveExecutorMgr");

  group_leader_ = (task_name == config.experimental().collective_group_leader())
                      ? ""
                      : config.experimental().collective_group_leader();
}

RpcCollectiveExecutorMgr::~RpcCollectiveExecutorMgr() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.cc", "RpcCollectiveExecutorMgr::~RpcCollectiveExecutorMgr");

  for (auto it : sequence_table_) {
    delete it.second;
  }
}

CollectiveExecutor* RpcCollectiveExecutorMgr::Create(int64_t step_id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.cc", "RpcCollectiveExecutorMgr::Create");

  CollectiveRemoteAccessDistributed* rma =
      new CollectiveRemoteAccessDistributed(dev_mgr_, dev_resolver_.get(),
                                            work_queue_, worker_cache_, step_id,
                                            task_name_);
  return new BaseCollectiveExecutor(this, rma, step_id, dev_mgr_, work_queue_);
}

namespace {
// StepId must leave the most-significant 7 bits empty for future use.
static const int64_t kStepIdMask = (((1uLL << 56) - 1) | (1uLL << 56));

int64_t NewRandomStepId() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc mht_3(mht_3_v, 241, "", "./tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.cc", "NewRandomStepId");

  int64_t step_id = random::New64();
  // Leave MS 8 bits clear for future use.
  step_id &= kStepIdMask;
  return step_id;
}
}  // namespace

void RpcCollectiveExecutorMgr::RefreshStepIdSequenceAsync(
    int64_t graph_key, const StatusCallback& done) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc mht_4(mht_4_v, 253, "", "./tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.cc", "RpcCollectiveExecutorMgr::RefreshStepIdSequenceAsync");

  if (group_leader_.empty()) {
    mutex_lock l(sequence_mu_);
    GraphKeySequence* gks = nullptr;
    auto it = sequence_table_.find(graph_key);
    if (it == sequence_table_.end()) {
      gks = new GraphKeySequence(graph_key);
      sequence_table_[graph_key] = gks;
    } else {
      gks = it->second;
    }
    gks->next_step_id_ = NewRandomStepId();
    done(Status::OK());
  } else {
    WorkerInterface* wi = worker_cache_->GetOrCreateWorker(group_leader_);
    GetStepSequenceRequest* req = new GetStepSequenceRequest;
    GetStepSequenceResponse* resp = new GetStepSequenceResponse;
    req->add_graph_key(graph_key);
    wi->GetStepSequenceAsync(
        req, resp, [this, req, resp, done](const Status& s) {
          if (!s.ok()) {
            LOG(ERROR) << "Bad response [" << s
                       << "] from GetStepSequenceAsync call to "
                       << group_leader_;
            done(s);
          } else {
            done(UpdateStepSequences(*resp));
          }
          delete req;
          delete resp;
        });
  }
}

void RpcCollectiveExecutorMgr::GetStepSequenceAsync(
    const GetStepSequenceRequest* request, GetStepSequenceResponse* response,
    const StatusCallback& done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc mht_5(mht_5_v, 292, "", "./tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.cc", "RpcCollectiveExecutorMgr::GetStepSequenceAsync");

  if (!group_leader_.empty()) {
    LOG(ERROR) << "GetStepSequence called at non-group-leader";
    done(errors::Internal("GetStepSequenceAsync called at non-group-leader"));
  } else {
    mutex_lock l(sequence_mu_);
    for (int64_t graph_key : request->graph_key()) {
      auto it = sequence_table_.find(graph_key);
      GraphKeySequence* gks = nullptr;
      if (it == sequence_table_.end()) {
        gks = new GraphKeySequence(graph_key);
        gks->next_step_id_ = NewRandomStepId();
        sequence_table_[graph_key] = gks;
      } else {
        gks = it->second;
      }
      StepSequence* ss = response->add_step_sequence();
      ss->set_graph_key(graph_key);
      ss->set_next_step_id(gks->next_step_id_);
    }
    done(Status::OK());
  }
}

Status RpcCollectiveExecutorMgr::UpdateStepSequences(
    const GetStepSequenceResponse& resp) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc mht_6(mht_6_v, 320, "", "./tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.cc", "RpcCollectiveExecutorMgr::UpdateStepSequences");

  mutex_lock l(sequence_mu_);
  for (const StepSequence& ss : resp.step_sequence()) {
    GraphKeySequence* gks = nullptr;
    auto it = sequence_table_.find(ss.graph_key());
    if (it == sequence_table_.end()) {
      gks = new GraphKeySequence(ss.graph_key());
      sequence_table_[ss.graph_key()] = gks;
    } else {
      gks = it->second;
    }
    gks->next_step_id_ = ss.next_step_id();
  }
  return Status::OK();
}

int64_t RpcCollectiveExecutorMgr::NextStepId(int64_t graph_key) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc mht_7(mht_7_v, 339, "", "./tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.cc", "RpcCollectiveExecutorMgr::NextStepId");

  mutex_lock l(sequence_mu_);
  auto it = sequence_table_.find(graph_key);
  if (it != sequence_table_.end()) {
    return it->second->next_step_id_;
  }
  return CollectiveExecutor::kInvalidId;
}

void RpcCollectiveExecutorMgr::RetireStepId(int64_t graph_key,
                                            int64_t step_id) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpc_collective_executor_mgrDTcc mht_8(mht_8_v, 352, "", "./tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.cc", "RpcCollectiveExecutorMgr::RetireStepId");

  mutex_lock l(sequence_mu_);
  auto it = sequence_table_.find(graph_key);
  if (it != sequence_table_.end()) {
    if (step_id == it->second->next_step_id_) {
      it->second->next_step_id_ = (it->second->next_step_id_ + 1) & kStepIdMask;
    } else {
      it->second->next_step_id_ = CollectiveExecutor::kInvalidId;
    }
  } else {
    LOG(ERROR) << "Failed to find graph_key " << graph_key << " to retire.";
  }
}

std::unique_ptr<RpcCollectiveExecutorMgr> CreateProdRpcCollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* device_mgr,
    std::unique_ptr<NcclCommunicatorInterface> nccl_communicator,
    WorkerCacheInterface* worker_cache, const string& default_worker_name) {
  auto dev_resolver = absl::make_unique<DeviceResolverDistributed>(device_mgr);
  auto param_resolver = absl::make_unique<CollectiveParamResolverDistributed>(
      config, device_mgr, dev_resolver.get(), nccl_communicator.get(),
      worker_cache, default_worker_name);
  return absl::make_unique<RpcCollectiveExecutorMgr>(
      config, device_mgr, std::move(dev_resolver), std::move(param_resolver),
      std::move(nccl_communicator), worker_cache, default_worker_name);
}

}  // namespace tensorflow
