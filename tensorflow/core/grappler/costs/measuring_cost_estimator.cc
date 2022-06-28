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
class MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSmeasuring_cost_estimatorDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSmeasuring_cost_estimatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSmeasuring_cost_estimatorDTcc() {
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

#include "tensorflow/core/grappler/costs/measuring_cost_estimator.h"

#include <limits>

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/robust_stats.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

MeasuringCostEstimator::MeasuringCostEstimator(Cluster* cluster,
                                               int measurement_steps,
                                               int measurement_threads)
    : measurement_steps_(measurement_steps),
      measurement_threads_(measurement_threads) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSmeasuring_cost_estimatorDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/grappler/costs/measuring_cost_estimator.cc", "MeasuringCostEstimator::MeasuringCostEstimator");

  CHECK_GE(measurement_steps, 1);
  if (measurement_threads > 0) {
    thread_pool_.reset(new thread::ThreadPool(
        Env::Default(), SanitizeThreadSuffix("measurements"),
        measurement_threads));
  }
  cluster_ = cluster;
}

Status MeasuringCostEstimator::Initialize(const GrapplerItem& item) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSmeasuring_cost_estimatorDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/grappler/costs/measuring_cost_estimator.cc", "MeasuringCostEstimator::Initialize");

  feed_ = item.feed;
  fetch_ = item.fetch;
  return cluster_->Initialize(item);
}

Status MeasuringCostEstimator::PredictCosts(const GraphDef& optimized_graph,
                                            RunMetadata* run_metadata,
                                            Costs* costs) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSmeasuring_cost_estimatorDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/grappler/costs/measuring_cost_estimator.cc", "MeasuringCostEstimator::PredictCosts");

  CostGraphDef* cost_graph = nullptr;
  if (run_metadata) {
    cost_graph = run_metadata->mutable_cost_graph();
  }
  const bool running_simulation = (cluster_->type() == "virtual");

  std::vector<double> times(measurement_steps_);
  BlockingCounter barrier(measurement_steps_);

  mutex status_mu;
  Status status;

  auto measurement_fn = [&](const int step) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSmeasuring_cost_estimatorDTcc mht_3(mht_3_v, 246, "", "./tensorflow/core/grappler/costs/measuring_cost_estimator.cc", "lambda");

    const Costs::MicroSeconds start = Env::Default()->NowMicros();

    RunMetadata metadata;
    const Status local_status =
        cluster_->Run(optimized_graph, feed_, fetch_, &metadata);
    {
      mutex_lock lock(status_mu);
      status.Update(local_status);
    }
    if (step < 0) {
      // Discard the first iteration as it triggers the warmup, and therefore
      // takes much longer than a normal step.
      return;
    }
    if (!local_status.ok()) {
      // Discard the data if the run wasn't successful.
      barrier.DecrementCount();
      return;
    }

    const Costs::MicroSeconds finish = Env::Default()->NowMicros();
    if (running_simulation) {
      // When running simulation, return the estimated runtime, not the time it
      // takes to run the simulation.
      double time = 0.0;
      for (const DeviceStepStats& stepstats :
           metadata.step_stats().dev_stats()) {
        for (const NodeExecStats& node_stats : stepstats.node_stats()) {
          const double completion_time =
              node_stats.all_end_rel_micros() + node_stats.all_start_micros();
          time = std::max(time, completion_time * 1e3);
        }
      }
      times[step] = time;
    } else {
      const double time = (finish - start).count() * 1e3;
      times[step] = time;
    }
    if (cost_graph && (step + 1 == measurement_steps_)) {
      metadata.mutable_cost_graph()->Swap(cost_graph);
    }

    barrier.DecrementCount();
  };

  // Initialize the computation and warm up TensorFlow.
  measurement_fn(-1);

  if (!status.ok()) {
    LOG(ERROR) << "Failed to run start measurements: "
               << status.error_message();
    costs->execution_time = Costs::Duration::max();
    return status;
  }

  // Run "measurement_steps_" and measure the time.
  VLOG(1) << "Number of measurement steps: " << measurement_steps_;
  if (measurement_threads_ > 0) {
    for (int i = 0; i < measurement_steps_; ++i) {
      thread_pool_->Schedule([i, &measurement_fn]() { measurement_fn(i); });
    }
    barrier.Wait();
  } else {
    for (int i = 0; i < measurement_steps_ && status.ok(); ++i) {
      measurement_fn(i);
    }
  }

  if (!status.ok()) {
    LOG(ERROR) << "Failed to measure graph performance: "
               << status.error_message();
    costs->execution_time = Costs::Duration::max();
    return status;
  }

  // Compute the average time of the measure steps. Use Huber statistics
  // to filter out outliers.
  RobustStats stats(times);
  costs->execution_time = Costs::Duration(stats.mean());

  return Status::OK();
}
}  // end namespace grappler
}  // end namespace tensorflow
