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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_AUTO_SHARD_REWRITER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_AUTO_SHARD_REWRITER_H_
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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTh {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTh() {
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

#include "absl/strings/string_view.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace data {

// Rewrites the dataset graph by applying an auto-shard policy.
class AutoShardRewriter {
 public:
  // Creates an `AutoShardRewriter` according to `task_def`. Returns an error if
  // the sharding policy is not a valid auto-shard policy.
  static StatusOr<AutoShardRewriter> Create(const TaskDef& task_def);

  // Applies auto-sharding to `graph_def`. If auto-shard policy is OFF, returns
  // the same graph as `graph_def`. Otherwise, returns the re-written graph.
  StatusOr<GraphDef> ApplyAutoShardRewrite(const GraphDef& graph_def);

 private:
  AutoShardRewriter(AutoShardPolicy auto_shard_policy, int64 num_workers,
                    int64 worker_index);

  // Creates a rewrite config based on the auto-shard policy.
  tensorflow::RewriterConfig::CustomGraphOptimizer GetRewriteConfig() const;

  const AutoShardPolicy auto_shard_policy_;
  const int64_t num_workers_;
  const int64_t worker_index_;
};

// Maps a worker to its index, given a list of workers. For example, suppose
// `worker_addresses` contains
//   /worker/task/0:worker, /worker/task/1:worker, /worker/task/2:worker,
// then
//   /worker/task/0:worker maps to index 0,
//   /worker/task/1:worker maps to index 1,
//   /worker/task/2:worker maps to index 2.
// This is useful for deterministically sharding a dataset among a fixed set of
// tf.data service workers.
class WorkerIndexResolver {
 public:
  // Constructs a `WorkerIndexResolver` to generate worker indexes according to
  // the specified worker addresses. The worker addresses can be "host" or
  // "host:port", where "port" is a number, named port, or "%port%" to be
  // replaced with the actual port.
  template <class T>
  explicit WorkerIndexResolver(const T& worker_addresses)
      : worker_addresses_(worker_addresses.cbegin(), worker_addresses.cend()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTh mht_0(mht_0_v, 241, "", "./tensorflow/core/data/service/auto_shard_rewriter.h", "WorkerIndexResolver");
}

  // Validates `worker_address`. Returns an error if the `worker_addresses` list
  // is non-empty and `worker_address` is not specified in the worker addresses
  // list (with optional port replacement).
  Status ValidateWorker(absl::string_view worker_address) const;

  // Processes a worker at address `worker_address`. Its index can be retrieved
  // by calling `GetWorkerIndex`.
  void AddWorker(absl::string_view worker_address);

  // Returns the worker index for the worker at `worker_address`. Returns a
  // NotFound error if the worker is not registered.
  StatusOr<int64_t> GetWorkerIndex(absl::string_view worker_address) const;

 private:
  std::vector<std::string> worker_addresses_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_AUTO_SHARD_REWRITER_H_
