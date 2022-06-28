/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_CLIENT_H_
#define TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_CLIENT_H_
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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTh {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTh() {
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


#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

// Client for communicating with the tf.data service dispatcher.
class DataServiceDispatcherClient : public DataServiceClientBase {
 public:
  DataServiceDispatcherClient(const std::string& address,
                              const std::string& protocol)
      : DataServiceClientBase(address, protocol) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("address: \"" + address + "\"");
   mht_0_v.push_back("protocol: \"" + protocol + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdispatcher_clientDTh mht_0(mht_0_v, 216, "", "./tensorflow/core/data/service/dispatcher_client.h", "DataServiceDispatcherClient");
}

  // Sends a heartbeat to the dispatcher. If the worker wasn't already
  // registered with the dispatcher, this will register the worker. The
  // dispatcher will report which new tasks the worker should run, and which
  // tasks it should delete.
  StatusOr<WorkerHeartbeatResponse> WorkerHeartbeat(
      const WorkerHeartbeatRequest& request);

  // Updates the dispatcher with information about the worker's state.
  Status WorkerUpdate(const std::string& worker_address,
                      std::vector<TaskProgress>& task_progress);

  // Gets a dataset definition for the given dataset id, and stores the
  // definition in `dataset_def`.
  Status GetDatasetDef(int64_t dataset_id, DatasetDef& dataset_def);

  // Gets the next split for the specified job id, iteration, and split
  // provider index.
  Status GetSplit(int64_t job_id, int64_t iteration,
                  int64_t split_provider_index, Tensor& split,
                  bool& end_of_splits);

  // Registers a dataset with the tf.data service, and stores the generated
  // dataset id in `dataset_id`.
  Status RegisterDataset(const DatasetDef& dataset,
                         const DataServiceMetadata& metadata,
                         int64_t& dataset_id);

  // If `job_key` is set, looks up a job matching `job_key`. If `job_key` is
  // absent or no matching job is found, creates a new job. The resulting job
  // id is stored in `job_client_id`.
  Status GetOrCreateJob(int64_t dataset_id,
                        const ProcessingModeDef& processing_mode,
                        const absl::optional<JobKeyDef>& job_key,
                        absl::optional<int64_t> num_consumers,
                        TargetWorkers target_workers, int64_t& job_client_id);

  // Releases a job client id, indicating that the id will no longer be used to
  // read from the job.
  Status ReleaseJobClient(int64_t job_client_id);

  // Attempts to remove a task. The task is removed if all consumers try to
  // remove the task in the same round.
  Status MaybeRemoveTask(int64_t task_id, int64_t consumer_index, int64_t round,
                         bool& removed);

  // Heartbeats to the dispatcher, getting back the tasks that should be
  // running, and whether the job is finished.
  Status ClientHeartbeat(ClientHeartbeatRequest& req,
                         ClientHeartbeatResponse& resp);

  // Queries the dispatcher for its registered workers. The worker info will be
  // stored in `workers`.
  Status GetWorkers(std::vector<WorkerInfo>& workers);

  // Returns data service metadata for the registered dataset.
  Status GetDataServiceMetadata(int64_t dataset_id,
                                DataServiceMetadata& metadata);

  // Returns data service config of the data service cluster.
  Status GetDataServiceConfig(DataServiceConfig& config);

 protected:
  Status EnsureInitialized() override;

 private:
  mutex mu_;
  // Initialization is guarded by `mu_`, but using the stub does not require
  // holding `mu_`
  std::unique_ptr<DispatcherService::Stub> stub_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_CLIENT_H_
