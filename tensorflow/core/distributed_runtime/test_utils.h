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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TEST_UTILS_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TEST_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh() {
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


#include <unordered_map>
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

// Some utilities for testing distributed-mode components in a single process
// without RPCs.

// Implements the worker interface with methods that just respond with
// "unimplemented" status.  Override just the methods needed for
// testing.
class TestWorkerInterface : public WorkerInterface {
 public:
  void GetStatusAsync(CallOptions* opts, const GetStatusRequest* request,
                      GetStatusResponse* response, bool fail_fast,
                      StatusCallback done) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_0(mht_0_v, 204, "", "./tensorflow/core/distributed_runtime/test_utils.h", "GetStatusAsync");

    done(errors::Unimplemented("GetStatusAsync"));
  }

  void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                CreateWorkerSessionResponse* response,
                                StatusCallback done) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_1(mht_1_v, 213, "", "./tensorflow/core/distributed_runtime/test_utils.h", "CreateWorkerSessionAsync");

    done(errors::Unimplemented("CreateWorkerSessionAsync"));
  }

  void DeleteWorkerSessionAsync(CallOptions* opts,
                                const DeleteWorkerSessionRequest* request,
                                DeleteWorkerSessionResponse* response,
                                StatusCallback done) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_2(mht_2_v, 223, "", "./tensorflow/core/distributed_runtime/test_utils.h", "DeleteWorkerSessionAsync");

    done(errors::Unimplemented("DeleteWorkerSessionAsync"));
  }

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_3(mht_3_v, 232, "", "./tensorflow/core/distributed_runtime/test_utils.h", "RegisterGraphAsync");

    done(errors::Unimplemented("RegisterGraphAsync"));
  }

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_4(mht_4_v, 241, "", "./tensorflow/core/distributed_runtime/test_utils.h", "DeregisterGraphAsync");

    done(errors::Unimplemented("DeregisterGraphAsync"));
  }

  void RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_5(mht_5_v, 250, "", "./tensorflow/core/distributed_runtime/test_utils.h", "RunGraphAsync");

    done(errors::Unimplemented("RunGraphAsync"));
  }

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_6(mht_6_v, 259, "", "./tensorflow/core/distributed_runtime/test_utils.h", "CleanupGraphAsync");

    done(errors::Unimplemented("CleanupGraphAsync"));
  }

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_7(mht_7_v, 268, "", "./tensorflow/core/distributed_runtime/test_utils.h", "CleanupAllAsync");

    done(errors::Unimplemented("CleanupAllAsync"));
  }

  void RecvTensorAsync(CallOptions* opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_8(mht_8_v, 276, "", "./tensorflow/core/distributed_runtime/test_utils.h", "RecvTensorAsync");

    done(errors::Unimplemented("RecvTensorAsync"));
  }

  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_9(mht_9_v, 284, "", "./tensorflow/core/distributed_runtime/test_utils.h", "LoggingAsync");

    done(errors::Unimplemented("LoggingAsync"));
  }

  void TracingAsync(const TracingRequest* request, TracingResponse* response,
                    StatusCallback done) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_10(mht_10_v, 292, "", "./tensorflow/core/distributed_runtime/test_utils.h", "TracingAsync");

    done(errors::Unimplemented("TracingAsync"));
  }

  void RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                    RecvBufResponse* response, StatusCallback done) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_11(mht_11_v, 300, "", "./tensorflow/core/distributed_runtime/test_utils.h", "RecvBufAsync");

    done(errors::Unimplemented("RecvBufAsync"));
  }

  void CompleteGroupAsync(CallOptions* opts,
                          const CompleteGroupRequest* request,
                          CompleteGroupResponse* response,
                          StatusCallback done) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_12(mht_12_v, 310, "", "./tensorflow/core/distributed_runtime/test_utils.h", "CompleteGroupAsync");

    done(errors::Unimplemented("CompleteGroupAsync"));
  }

  void CompleteInstanceAsync(CallOptions* ops,
                             const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             StatusCallback done) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_13(mht_13_v, 320, "", "./tensorflow/core/distributed_runtime/test_utils.h", "CompleteInstanceAsync");

    done(errors::Unimplemented("CompleteInstanceAsync"));
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            StatusCallback done) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_14(mht_14_v, 329, "", "./tensorflow/core/distributed_runtime/test_utils.h", "GetStepSequenceAsync");

    done(errors::Unimplemented("GetStepSequenceAsync"));
  }
};

class TestWorkerCache : public WorkerCacheInterface {
 public:
  virtual ~TestWorkerCache() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_15(mht_15_v, 339, "", "./tensorflow/core/distributed_runtime/test_utils.h", "~TestWorkerCache");
}

  void AddWorker(const string& target, WorkerInterface* wi) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_16(mht_16_v, 345, "", "./tensorflow/core/distributed_runtime/test_utils.h", "AddWorker");

    workers_[target] = wi;
  }

  void AddDevice(const string& device_name, const DeviceLocality& dev_loc) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_17(mht_17_v, 353, "", "./tensorflow/core/distributed_runtime/test_utils.h", "AddDevice");

    localities_[device_name] = dev_loc;
  }

  void ListWorkers(std::vector<string>* workers) const override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_18(mht_18_v, 360, "", "./tensorflow/core/distributed_runtime/test_utils.h", "ListWorkers");

    workers->clear();
    for (auto it : workers_) {
      workers->push_back(it.first);
    }
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_19(mht_19_v, 372, "", "./tensorflow/core/distributed_runtime/test_utils.h", "ListWorkersInJob");

    workers->clear();
    for (auto it : workers_) {
      DeviceNameUtils::ParsedName device_name;
      CHECK(DeviceNameUtils::ParseFullName(it.first, &device_name));
      CHECK(device_name.has_job);
      if (job_name == device_name.job) {
        workers->push_back(it.first);
      }
    }
  }

  WorkerInterface* GetOrCreateWorker(const string& target) override {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_20(mht_20_v, 388, "", "./tensorflow/core/distributed_runtime/test_utils.h", "GetOrCreateWorker");

    auto it = workers_.find(target);
    if (it != workers_.end()) {
      return it->second;
    }
    return nullptr;
  }

  void ReleaseWorker(const string& target, WorkerInterface* worker) override {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_21(mht_21_v, 400, "", "./tensorflow/core/distributed_runtime/test_utils.h", "ReleaseWorker");
}

  Status GetEagerClientCache(
      std::unique_ptr<eager::EagerClientCache>* eager_client_cache) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_22(mht_22_v, 406, "", "./tensorflow/core/distributed_runtime/test_utils.h", "GetEagerClientCache");

    return errors::Unimplemented("Unimplemented.");
  }

  Status GetCoordinationClientCache(
      std::unique_ptr<CoordinationClientCache>* coord_client_cache) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_23(mht_23_v, 414, "", "./tensorflow/core/distributed_runtime/test_utils.h", "GetCoordinationClientCache");

    return errors::Unimplemented("Unimplemented.");
  }

  bool GetDeviceLocalityNonBlocking(const string& device,
                                    DeviceLocality* locality) override {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_24(mht_24_v, 423, "", "./tensorflow/core/distributed_runtime/test_utils.h", "GetDeviceLocalityNonBlocking");

    auto it = localities_.find(device);
    if (it != localities_.end()) {
      *locality = it->second;
      return true;
    }
    return false;
  }

  void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality,
                              StatusCallback done) override {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStest_utilsDTh mht_25(mht_25_v, 437, "", "./tensorflow/core/distributed_runtime/test_utils.h", "GetDeviceLocalityAsync");

    auto it = localities_.find(device);
    if (it != localities_.end()) {
      *locality = it->second;
      done(Status::OK());
      return;
    }
    done(errors::Internal("Device not found: ", device));
  }

 protected:
  std::unordered_map<string, WorkerInterface*> workers_;
  std::unordered_map<string, DeviceLocality> localities_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_TEST_UTILS_H_
