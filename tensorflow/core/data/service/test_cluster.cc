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
class MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/service/test_cluster.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/data/service/server_lib.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {
namespace {
constexpr const char kProtocol[] = "grpc";
}  // namespace

TestCluster::TestCluster(int num_workers) : num_workers_(num_workers) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/data/service/test_cluster.cc", "TestCluster::TestCluster");
}

TestCluster::TestCluster(const TestCluster::Config& config)
    : num_workers_(config.num_workers), config_(config) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/data/service/test_cluster.cc", "TestCluster::TestCluster");
}

Status TestCluster::Initialize() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTcc mht_2(mht_2_v, 218, "", "./tensorflow/core/data/service/test_cluster.cc", "TestCluster::Initialize");

  if (initialized_) {
    return errors::FailedPrecondition(
        "Test cluster has already been initialized.");
  }
  initialized_ = true;
  experimental::DispatcherConfig dispatcher_config;
  dispatcher_config.set_protocol(kProtocol);
  for (int i = 0; i < num_workers_; ++i) {
    dispatcher_config.add_worker_addresses("localhost");
  }
  dispatcher_config.set_deployment_mode(DEPLOYMENT_MODE_COLOCATED);
  dispatcher_config.set_job_gc_check_interval_ms(
      config_.job_gc_check_interval_ms);
  dispatcher_config.set_job_gc_timeout_ms(config_.job_gc_timeout_ms);
  dispatcher_config.set_client_timeout_ms(config_.client_timeout_ms);
  TF_RETURN_IF_ERROR(NewDispatchServer(dispatcher_config, dispatcher_));
  TF_RETURN_IF_ERROR(dispatcher_->Start());
  dispatcher_address_ = absl::StrCat("localhost:", dispatcher_->BoundPort());
  workers_.reserve(num_workers_);
  worker_addresses_.reserve(num_workers_);
  for (int i = 0; i < num_workers_; ++i) {
    TF_RETURN_IF_ERROR(AddWorker());
  }
  return Status::OK();
}

Status TestCluster::AddWorker() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTcc mht_3(mht_3_v, 248, "", "./tensorflow/core/data/service/test_cluster.cc", "TestCluster::AddWorker");

  std::unique_ptr<WorkerGrpcDataServer> worker;
  experimental::WorkerConfig config;
  config.set_protocol(kProtocol);
  config.set_dispatcher_address(dispatcher_address_);
  config.set_worker_address("localhost:%port%");
  TF_RETURN_IF_ERROR(NewWorkerServer(config, worker));
  TF_RETURN_IF_ERROR(worker->Start());
  worker_addresses_.push_back(absl::StrCat("localhost:", worker->BoundPort()));
  workers_.push_back(std::move(worker));
  return Status::OK();
}

std::string TestCluster::DispatcherAddress() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTcc mht_4(mht_4_v, 264, "", "./tensorflow/core/data/service/test_cluster.cc", "TestCluster::DispatcherAddress");

  return dispatcher_address_;
}

std::string TestCluster::WorkerAddress(int index) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTcc mht_5(mht_5_v, 271, "", "./tensorflow/core/data/service/test_cluster.cc", "TestCluster::WorkerAddress");

  DCHECK_GE(index, 0);
  DCHECK_LT(index, worker_addresses_.size());
  return worker_addresses_[index];
}

void TestCluster::StopWorker(size_t index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTcc mht_6(mht_6_v, 280, "", "./tensorflow/core/data/service/test_cluster.cc", "TestCluster::StopWorker");

  DCHECK_GE(index, 0);
  DCHECK_LT(index, worker_addresses_.size());
  workers_[index]->Stop();
}

void TestCluster::StopWorkers() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_clusterDTcc mht_7(mht_7_v, 289, "", "./tensorflow/core/data/service/test_cluster.cc", "TestCluster::StopWorkers");

  for (std::unique_ptr<WorkerGrpcDataServer>& worker : workers_) {
    worker->Stop();
  }
}

}  // namespace data
}  // namespace tensorflow
