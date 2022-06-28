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
class MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_executor_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_executor_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_executor_mgrDTcc() {
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
#include "tensorflow/core/common_runtime/collective_executor_mgr.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/build_graph_options.h"
#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

CollectiveExecutorMgr::CollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* dev_mgr,
    std::unique_ptr<DeviceResolverInterface> dev_resolver,
    std::unique_ptr<ParamResolverInterface> param_resolver,
    std::unique_ptr<NcclCommunicatorInterface> nccl_communicator)
    : dev_mgr_(dev_mgr),
      dev_resolver_(std::move(dev_resolver)),
      param_resolver_(std::move(param_resolver)),
      gpu_ring_order_(
          config.gpu_options().experimental().collective_ring_order()),
      nccl_communicator_(std::move(nccl_communicator)),
      work_queue_(std::make_shared<UnboundedWorkQueue>(Env::Default(),
                                                       "collective_ops")) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_executor_mgrDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/common_runtime/collective_executor_mgr.cc", "CollectiveExecutorMgr::CollectiveExecutorMgr");
}

CollectiveExecutorMgr::~CollectiveExecutorMgr() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_executor_mgrDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/common_runtime/collective_executor_mgr.cc", "CollectiveExecutorMgr::~CollectiveExecutorMgr");

  for (auto iter : executor_table_) {
    iter.second->Unref();
  }
}

CollectiveExecutor* CollectiveExecutorMgr::FindOrCreate(int64_t step_id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_executor_mgrDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/common_runtime/collective_executor_mgr.cc", "CollectiveExecutorMgr::FindOrCreate");

  CollectiveExecutor* ce = nullptr;
  {
    mutex_lock l(exec_mu_);
    auto it = executor_table_.find(step_id);
    if (it != executor_table_.end()) {
      ce = it->second;
    } else {
      ce = Create(step_id);
      executor_table_[step_id] = ce;
    }
    ce->Ref();
  }
  return ce;
}

CollectiveExecutor* CollectiveExecutorMgr::Create(int64_t step_id) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_executor_mgrDTcc mht_3(mht_3_v, 243, "", "./tensorflow/core/common_runtime/collective_executor_mgr.cc", "CollectiveExecutorMgr::Create");

  CollectiveRemoteAccessLocal* rma =
      new CollectiveRemoteAccessLocal(dev_mgr_, dev_resolver_.get(), step_id);
  return new BaseCollectiveExecutor(this, rma, step_id, dev_mgr_, work_queue_);
}

void CollectiveExecutorMgr::Cleanup(int64_t step_id) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_executor_mgrDTcc mht_4(mht_4_v, 252, "", "./tensorflow/core/common_runtime/collective_executor_mgr.cc", "CollectiveExecutorMgr::Cleanup");

  CollectiveExecutor* ce = nullptr;
  {
    mutex_lock l(exec_mu_);
    auto it = executor_table_.find(step_id);
    if (it != executor_table_.end()) {
      ce = it->second;
      executor_table_.erase(it);
    }
  }
  if (ce) ce->Unref();
}

void CollectiveExecutorMgr::GetStepSequenceAsync(
    const GetStepSequenceRequest* request, GetStepSequenceResponse* response,
    const StatusCallback& done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_executor_mgrDTcc mht_5(mht_5_v, 270, "", "./tensorflow/core/common_runtime/collective_executor_mgr.cc", "CollectiveExecutorMgr::GetStepSequenceAsync");

  done(errors::Internal(
      "CollectiveExecutorMgr does not implement GetStepSequence."));
}

void CollectiveExecutorMgr::RefreshStepIdSequenceAsync(
    int64_t graph_key, const StatusCallback& done) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_executor_mgrDTcc mht_6(mht_6_v, 279, "", "./tensorflow/core/common_runtime/collective_executor_mgr.cc", "CollectiveExecutorMgr::RefreshStepIdSequenceAsync");

  done(errors::Internal(
      "CollectiveExecutorMgr does not implement RefreshStepIdSequence."));
}

std::unique_ptr<CollectiveExecutorMgr> CreateProdLocalCollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* device_mgr,
    std::unique_ptr<NcclCommunicatorInterface> nccl_communicator) {
  auto device_resolver = absl::make_unique<DeviceResolverLocal>(device_mgr);
  auto param_resolver = absl::make_unique<CollectiveParamResolverLocal>(
      config, device_mgr, device_resolver.get(), nccl_communicator.get(),
      "/job:localhost/replica:0/task:0");
  return absl::make_unique<CollectiveExecutorMgr>(
      config, device_mgr, std::move(device_resolver), std::move(param_resolver),
      std::move(nccl_communicator));
}

}  // namespace tensorflow
