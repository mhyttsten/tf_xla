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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_TEST_COLLECTIVE_EXECUTOR_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_TEST_COLLECTIVE_EXECUTOR_MGR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh() {
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


#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace tensorflow {

// Mock objects that can't actually execute a Collective, but satisfy
// general infrastructure expectations within tests that don't require
// full functionality.

class TestCollectiveExecutor : public CollectiveExecutor {
 public:
  explicit TestCollectiveExecutor(CollectiveExecutorMgrInterface* cem,
                                  CollectiveRemoteAccess* rma = nullptr)
      : CollectiveExecutor(cem), rma_(rma) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_0(mht_0_v, 201, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "TestCollectiveExecutor");
}

  void RunClosure(std::function<void()> fn) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_1(mht_1_v, 206, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "RunClosure");
 fn(); }

  CollectiveRemoteAccess* remote_access() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_2(mht_2_v, 211, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "remote_access");
 return rma_; }

 private:
  CollectiveRemoteAccess* rma_;
};

class TestParamResolver : public ParamResolverInterface {
  void CompleteParamsAsync(const DeviceAttributes& device, CollectiveParams* cp,
                           CancellationManager* cancel_mgr,
                           const StatusCallback& done) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_3(mht_3_v, 223, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "CompleteParamsAsync");

    done(errors::Internal("Unimplemented"));
  }

  void CompleteGroupAsync(const DeviceAttributes& device,
                          CollGroupParams* group_params,
                          CancellationManager* cancel_mgr,
                          const StatusCallback& done) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_4(mht_4_v, 233, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "CompleteGroupAsync");

    done(errors::Internal("Unimplemented"));
  }

  void CompleteInstanceAsync(const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             CancellationManager* cancel_mgr,
                             const StatusCallback& done) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_5(mht_5_v, 243, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "CompleteInstanceAsync");

    done(errors::Internal("Unimplemented"));
  }

  Status LookupGroup(int32_t group_key, CollGroupParams* group) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_6(mht_6_v, 250, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "LookupGroup");

    return errors::Internal("Unimplemented");
  }

  void StartAbort(const Status& s) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_7(mht_7_v, 257, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "StartAbort");
}
};

class TestCollectiveExecutorMgr : public CollectiveExecutorMgrInterface {
 public:
  explicit TestCollectiveExecutorMgr(ParamResolverInterface* param_resolver,
                                     CollectiveRemoteAccess* rma)
      : param_resolver_(param_resolver), rma_(rma) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_8(mht_8_v, 267, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "TestCollectiveExecutorMgr");
}

  TestCollectiveExecutorMgr() : param_resolver_(nullptr), rma_(nullptr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_9(mht_9_v, 272, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "TestCollectiveExecutorMgr");
}

  ~TestCollectiveExecutorMgr() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_10(mht_10_v, 277, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "~TestCollectiveExecutorMgr");

    for (auto& iter : table_) {
      iter.second->Unref();
    }
  }

  CollectiveExecutor* FindOrCreate(int64_t step_id) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_11(mht_11_v, 286, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "FindOrCreate");

    mutex_lock l(mu_);
    CollectiveExecutor* ce = nullptr;
    auto iter = table_.find(step_id);
    if (iter != table_.end()) {
      ce = iter->second;
    } else {
      ce = new TestCollectiveExecutor(this, rma_);
      table_[step_id] = ce;
    }
    ce->Ref();
    return ce;
  }

  void Cleanup(int64_t step_id) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_12(mht_12_v, 303, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "Cleanup");

    mutex_lock l(mu_);
    auto iter = table_.find(step_id);
    if (iter != table_.end()) {
      iter->second->Unref();
      table_.erase(iter);
    }
  }

  ParamResolverInterface* GetParamResolver() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_13(mht_13_v, 315, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "GetParamResolver");

    return param_resolver_;
  }

  DeviceResolverInterface* GetDeviceResolver() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_14(mht_14_v, 322, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "GetDeviceResolver");

    LOG(FATAL);
    return nullptr;
  }

  NcclCommunicatorInterface* GetNcclCommunicator() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_15(mht_15_v, 330, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "GetNcclCommunicator");

    return nullptr;
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            const StatusCallback& done) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_16(mht_16_v, 339, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "GetStepSequenceAsync");

    done(errors::Internal("unimplemented"));
  }

  void RefreshStepIdSequenceAsync(int64_t graph_key,
                                  const StatusCallback& done) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_17(mht_17_v, 347, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "RefreshStepIdSequenceAsync");

    done(errors::Internal("unimplemented"));
  }

  int64_t NextStepId(int64_t graph_key) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_18(mht_18_v, 354, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "NextStepId");

    return CollectiveExecutor::kInvalidId;
  }

  void RetireStepId(int64_t graph_key, int64_t step_id) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePStest_collective_executor_mgrDTh mht_19(mht_19_v, 361, "", "./tensorflow/core/common_runtime/test_collective_executor_mgr.h", "RetireStepId");
}

 protected:
  mutex mu_;
  gtl::FlatMap<int64_t, CollectiveExecutor*> table_ TF_GUARDED_BY(mu_);
  ParamResolverInterface* param_resolver_;
  CollectiveRemoteAccess* rma_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_TEST_COLLECTIVE_EXECUTOR_MGR_H_
