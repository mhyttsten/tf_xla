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
class MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

Cluster::Cluster(int timeout_s) : timeout_s_(timeout_s) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc mht_0(mht_0_v, 191, "", "./tensorflow/core/grappler/clusters/cluster.cc", "Cluster::Cluster");

  DisableDetailedStats(false);
}

Cluster::~Cluster() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc mht_1(mht_1_v, 198, "", "./tensorflow/core/grappler/clusters/cluster.cc", "Cluster::~Cluster");
}

void Cluster::AllowSoftPlacement(bool soft_placement_state) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc mht_2(mht_2_v, 203, "", "./tensorflow/core/grappler/clusters/cluster.cc", "Cluster::AllowSoftPlacement");

  options_.config.set_allow_soft_placement(soft_placement_state);
}

void Cluster::SetNumInterOpThreads(int num_threads) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc mht_3(mht_3_v, 210, "", "./tensorflow/core/grappler/clusters/cluster.cc", "Cluster::SetNumInterOpThreads");

  for (int i = 0; i < options_.config.session_inter_op_thread_pool_size();
       ++i) {
    options_.config.mutable_session_inter_op_thread_pool(i)->set_num_threads(
        num_threads);
  }
}

void Cluster::SetNumWarmupSteps(int num_steps) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc mht_4(mht_4_v, 221, "", "./tensorflow/core/grappler/clusters/cluster.cc", "Cluster::SetNumWarmupSteps");

  options_.config.mutable_graph_options()->set_build_cost_model_after(
      num_steps);
}

// Set executor type to instantiate
void Cluster::SetExecutorType(const string* executor_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc mht_5(mht_5_v, 230, "", "./tensorflow/core/grappler/clusters/cluster.cc", "Cluster::SetExecutorType");

  options_.config.mutable_experimental()->set_executor_type(*executor_type);
}

int Cluster::NumWarmupSteps() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc mht_6(mht_6_v, 237, "", "./tensorflow/core/grappler/clusters/cluster.cc", "Cluster::NumWarmupSteps");

  return options_.config.graph_options().build_cost_model_after();
}

void Cluster::DisableDetailedStats(bool disable) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc mht_7(mht_7_v, 244, "", "./tensorflow/core/grappler/clusters/cluster.cc", "Cluster::DisableDetailedStats");

  if (disable) {
    options_.config.mutable_graph_options()->set_build_cost_model(0);
    run_options_.set_trace_level(RunOptions::NO_TRACE);
  } else {
    options_.config.mutable_graph_options()->set_build_cost_model(1);
    run_options_.set_trace_level(RunOptions::HARDWARE_TRACE);
  }
}

bool Cluster::DetailedStatsEnabled() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc mht_8(mht_8_v, 257, "", "./tensorflow/core/grappler/clusters/cluster.cc", "Cluster::DetailedStatsEnabled");

  return options_.config.graph_options().build_cost_model() != 0;
}

void Cluster::DisableOptimizer(bool disable) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc mht_9(mht_9_v, 264, "", "./tensorflow/core/grappler/clusters/cluster.cc", "Cluster::DisableOptimizer");

  OptimizerOptions* options =
      options_.config.mutable_graph_options()->mutable_optimizer_options();
  if (disable) {
    options->set_opt_level(OptimizerOptions::L0);
    // Disable Grappler optimizations.
    auto rewriter_config =
        options_.config.mutable_graph_options()->mutable_rewrite_options();
    rewriter_config->set_layout_optimizer(RewriterConfig::OFF);
    rewriter_config->set_disable_model_pruning(true);
    rewriter_config->set_function_optimization(RewriterConfig::OFF);
    rewriter_config->set_arithmetic_optimization(RewriterConfig::OFF);
    rewriter_config->set_loop_optimization(RewriterConfig::OFF);
    rewriter_config->set_dependency_optimization(RewriterConfig::OFF);
    rewriter_config->set_constant_folding(RewriterConfig::OFF);
    rewriter_config->set_memory_optimization(RewriterConfig::NO_MEM_OPT);
    rewriter_config->set_shape_optimization(RewriterConfig::OFF);
    rewriter_config->set_remapping(RewriterConfig::OFF);
    rewriter_config->set_pin_to_host_optimization(RewriterConfig::OFF);
    rewriter_config->mutable_auto_parallel()->set_enable(false);
    rewriter_config->clear_optimizers();
  } else {
    options->set_opt_level(OptimizerOptions::L1);
    auto rewriter_config =
        options_.config.mutable_graph_options()->mutable_rewrite_options();
    rewriter_config->set_constant_folding(RewriterConfig::DEFAULT);
    rewriter_config->set_memory_optimization(RewriterConfig::DEFAULT_MEM_OPT);
  }
}

const std::vector<string> Cluster::GetDeviceNames() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTcc mht_10(mht_10_v, 297, "", "./tensorflow/core/grappler/clusters/cluster.cc", "Cluster::GetDeviceNames");

  std::vector<string> device_names;
  device_names.reserve(devices_.size());
  for (const auto& device : devices_) {
    device_names.push_back(device.first);
  }
  std::sort(device_names.begin(), device_names.end());
  return device_names;
}

}  // end namespace grappler
}  // end namespace tensorflow
