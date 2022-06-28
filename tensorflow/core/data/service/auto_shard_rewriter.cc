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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc() {
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
#include "tensorflow/core/data/service/auto_shard_rewriter.h"

#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/rewrite_utils.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/url.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/data/auto_shard.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"
#include "tensorflow/core/kernels/data/experimental/auto_shard_dataset_op.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::data::experimental::AutoShardDatasetOp;

// A dynamic port has form %port% or %port_foo% that is to be replaced with the
// actual port.
bool HasDynamicPort(absl::string_view address) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("address: \"" + std::string(address.data(), address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc mht_0(mht_0_v, 229, "", "./tensorflow/core/data/service/auto_shard_rewriter.cc", "HasDynamicPort");

  URL url(address);
  return url.has_port() && absl::StartsWith(url.port(), "%port") &&
         absl::EndsWith(url.port(), "%");
}

// Returns true if `config_address` has no port or a dynamic port (e.g.: %port%)
// and `worker_address` has an actual port (number of named port).
//
// For example, it returns true for the following cases:
//
//  config_address                    worker_address
//  ----------------------------------------------------------
//  /worker/task/0                    /worker/task/0:worker
//  /worker/task/0:%port%             /worker/task/0:10000
//  /worker/task/0:%port_worker%      /worker/task/0:worker
//  /worker/task/0:%port_worker%      /worker/task/0:10000
//  localhost                         localhost:10000
//  localhost:%port%                  localhost:10000
bool ShouldReplaceDynamicPort(absl::string_view config_address,
                              absl::string_view worker_address) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("config_address: \"" + std::string(config_address.data(), config_address.size()) + "\"");
   mht_1_v.push_back("worker_address: \"" + std::string(worker_address.data(), worker_address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc mht_1(mht_1_v, 254, "", "./tensorflow/core/data/service/auto_shard_rewriter.cc", "ShouldReplaceDynamicPort");

  URL config_url(config_address), worker_url(worker_address);
  return (!config_url.has_port() || HasDynamicPort(config_address)) &&
         worker_url.has_port() && config_url.host() == worker_url.host();
}
}  // namespace

StatusOr<AutoShardRewriter> AutoShardRewriter::Create(const TaskDef& task_def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc mht_2(mht_2_v, 264, "", "./tensorflow/core/data/service/auto_shard_rewriter.cc", "AutoShardRewriter::Create");

  TF_ASSIGN_OR_RETURN(
      AutoShardPolicy auto_shard_policy,
      ToAutoShardPolicy(task_def.processing_mode_def().sharding_policy()));
  return AutoShardRewriter(auto_shard_policy, task_def.num_workers(),
                           task_def.worker_index());
}

StatusOr<GraphDef> AutoShardRewriter::ApplyAutoShardRewrite(
    const GraphDef& graph_def) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc mht_3(mht_3_v, 276, "", "./tensorflow/core/data/service/auto_shard_rewriter.cc", "AutoShardRewriter::ApplyAutoShardRewrite");

  if (auto_shard_policy_ == AutoShardPolicy::OFF) {
    return graph_def;
  }

  VLOG(2) << "Applying auto-shard policy "
          << AutoShardPolicy_Name(auto_shard_policy_)
          << ". Number of workers: " << num_workers_
          << "; worker index: " << worker_index_ << ".";
  grappler::AutoShard autoshard;
  tensorflow::RewriterConfig::CustomGraphOptimizer config = GetRewriteConfig();
  TF_RETURN_IF_ERROR(autoshard.Init(&config));

  GraphDef input_graph = graph_def;
  TF_ASSIGN_OR_RETURN(std::string dataset_node, GetDatasetNode(input_graph));
  std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
      GetGrapplerItem(&input_graph, &dataset_node, /*add_fake_sinks=*/false);

  GraphDef rewritten_graph;
  std::unordered_map<std::string, tensorflow::DeviceProperties> device_map;
  tensorflow::grappler::VirtualCluster cluster(device_map);
  grappler::AutoShard::OptimizationStats stats;
  TF_RETURN_IF_ERROR(autoshard.OptimizeAndCollectStats(
      &cluster, *grappler_item, &rewritten_graph, &stats));
  return rewritten_graph;
}

AutoShardRewriter::AutoShardRewriter(AutoShardPolicy auto_shard_policy,
                                     int64 num_workers, int64 worker_index)
    : auto_shard_policy_(auto_shard_policy),
      num_workers_(num_workers),
      worker_index_(worker_index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc mht_4(mht_4_v, 310, "", "./tensorflow/core/data/service/auto_shard_rewriter.cc", "AutoShardRewriter::AutoShardRewriter");
}

tensorflow::RewriterConfig::CustomGraphOptimizer
AutoShardRewriter::GetRewriteConfig() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc mht_5(mht_5_v, 316, "", "./tensorflow/core/data/service/auto_shard_rewriter.cc", "AutoShardRewriter::GetRewriteConfig");

  tensorflow::RewriterConfig::CustomGraphOptimizer config;
  config.set_name("tf-data-service-auto-shard");
  (*config.mutable_parameter_map())[AutoShardDatasetOp::kNumWorkers].set_i(
      num_workers_);
  (*config.mutable_parameter_map())[AutoShardDatasetOp::kIndex].set_i(
      worker_index_);
  (*config.mutable_parameter_map())[AutoShardDatasetOp::kAutoShardPolicy].set_i(
      auto_shard_policy_);
  // This parameter is used internally by tf.distribute to rebatch the dataset.
  // It is not used outside the context of `experimental_distribute_dataset`.
  (*config.mutable_parameter_map())[AutoShardDatasetOp::kNumReplicas].set_i(1);
  return config;
}

Status WorkerIndexResolver::ValidateWorker(
    absl::string_view worker_address) const {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("worker_address: \"" + std::string(worker_address.data(), worker_address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc mht_6(mht_6_v, 336, "", "./tensorflow/core/data/service/auto_shard_rewriter.cc", "WorkerIndexResolver::ValidateWorker");

  if (worker_addresses_.empty()) {
    return Status::OK();
  }

  for (absl::string_view config_address : worker_addresses_) {
    if (config_address == worker_address ||
        ShouldReplaceDynamicPort(config_address, worker_address)) {
      return Status::OK();
    }
  }

  return errors::FailedPrecondition(absl::Substitute(
      "Failed to assign an index for worker $0. Configured workers list: [$1]. "
      "The worker's address is not configured, or other workers are already "
      "running at the configured host. If your worker has restarted, make sure "
      "it runs at the same address and port.",
      worker_address, absl::StrJoin(worker_addresses_, ", ")));
}

void WorkerIndexResolver::AddWorker(absl::string_view worker_address) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("worker_address: \"" + std::string(worker_address.data(), worker_address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc mht_7(mht_7_v, 360, "", "./tensorflow/core/data/service/auto_shard_rewriter.cc", "WorkerIndexResolver::AddWorker");

  for (std::string& config_address : worker_addresses_) {
    if (config_address == worker_address) {
      return;
    }
    if (ShouldReplaceDynamicPort(config_address, worker_address)) {
      config_address = std::string(worker_address);
      return;
    }
  }
}

StatusOr<int64_t> WorkerIndexResolver::GetWorkerIndex(
    absl::string_view worker_address) const {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("worker_address: \"" + std::string(worker_address.data(), worker_address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSauto_shard_rewriterDTcc mht_8(mht_8_v, 377, "", "./tensorflow/core/data/service/auto_shard_rewriter.cc", "WorkerIndexResolver::GetWorkerIndex");

  const auto it = absl::c_find(worker_addresses_, worker_address);
  if (it == worker_addresses_.cend()) {
    return errors::NotFound(absl::Substitute(
        "Failed to shard dataset in tf.data service: Worker $0 is not in the "
        "workers list. Got workers list $1.",
        worker_address, absl::StrJoin(worker_addresses_, ",")));
  }
  return std::distance(worker_addresses_.cbegin(), it);
}

}  // namespace data
}  // namespace tensorflow
