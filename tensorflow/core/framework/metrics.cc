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
class MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc() {
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

#include "tensorflow/core/framework/metrics.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace metrics {
namespace {

auto* graph_runs = monitoring::Counter<0>::New(
    "/tensorflow/core/graph_runs",
    "The number of graph executions used to collect "
    "/tensorflow/core/graph_run_time_usecs");

auto* graph_run_time_usecs = monitoring::Counter<0>::New(
    "/tensorflow/core/graph_run_time_usecs",
    "The total time spent on executing graphs in microseconds.");

auto* graph_run_time_usecs_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_run_time_usecs_histogram",
     "The wall-clock time spent on executing graphs in microseconds."},
    // Power of 2 with bucket count 20 (> 17 minutes)
    {monitoring::Buckets::Exponential(1000, 2, 20)});

auto* graph_pending_queue_length_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_pending_queue_length_histogram",
     "The number of pending (ready but not running) tasks in graph executor."},
    // Power of 1.5 with bucket count 30 (> 191k)
    {monitoring::Buckets::Exponential(1, 1.5, 30)});

auto* graph_run_input_tensor_bytes = monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_run_input_tensor_bytes",
     "The size of input tensors in bytes."},
    // Power of 2 with bucket count 14 (256MB)
    {monitoring::Buckets::Exponential(1, 4, 14)});

auto* graph_run_output_tensor_bytes = monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_run_output_tensor_bytes",
     "The size of output tensors in bytes."},
    // Power of 2 with bucket count 14 (256MB)
    {monitoring::Buckets::Exponential(1, 4, 14)});

auto* graph_unused_outputs = monitoring::Counter<1>::New(
    "/tensorflow/core/graph_unused_outputs",
    "The number of unused outputs for ops of a given type.", "name");

auto* tf_data_autotune_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/autotune", "tf.data autotuning", "name");

auto* tf_data_bytes_consumed_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/bytes_consumed",
    "The number of bytes consumed by a tf.data Dataset.", "name");

auto* tf_data_bytes_produced_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/bytes_produced",
    "The number of bytes produced by a tf.data Dataset.", "name");

auto* tf_data_bytes_read_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/bytes_read",
    "The number of bytes read by tf.data Dataset sources.", "name");

auto* tf_data_bytes_fetched_counter = monitoring::Counter<0>::New(
    "/tensorflow/data/bytes_fetched",
    "The number of bytes fetched from tf.data Dataset iterator.");

auto* tf_data_elements_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/elements", "tf.data elements", "name");

auto* tf_data_experiment_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/experiment",
    "The number of times tf.data experiment is applied to input pipelines.",
    "name");

auto* tf_data_fingerprint_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/fingerprint", "tf.data fingerprint", "name");

auto* tf_data_get_next_duration_usecs_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/data/getnext_duration",
     "Microseconds spent fetching an element from tf.data iterator."},
    // Power of 2 with bucket count 10 (1024 microseconds) and 1 second.
    {monitoring::Buckets::Explicit(
        {2., 4., 8., 16., 32., 64., 128., 256., 512., 1024., 1e6})});

auto* tf_data_used_vs_budget_ratio_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/data/used_vs_budget_ratio",
     "Ratio of tf.data used ram over ram budget when running optimization."},
    // Uniform linear buckets with count 10 from 0 to 2
    {monitoring::Buckets::Explicit(
        {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0})});

auto* tf_data_buffered_vs_budget_ratio_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/data/buffered_vs_budget_ratio",
     "Ratio of tf.data max buffer bytes over ram budget when running "
     "optimization."},
    // Uniform linear buckets with count 10 from 0 to 2
    {monitoring::Buckets::Explicit(
        {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0})});

auto* tf_data_iterator_busy_counter =
    monitoring::Counter<0>::New("/tensorflow/data/iterator_busy",
                                "The time (in microseconds) during which a "
                                "tf.data iterator was busy processing at "
                                "least one `GetNext()` request.");

auto* tf_data_iterator_lifetime_counter = monitoring::Counter<0>::New(
    "/tensorflow/data/iterator_lifetime",
    "The time (in microseconds) between a tf.data iterator receiving the first "
    "`GetNext()` request and responding to the last `GetNext()` request.");

auto* tf_data_iterator_gap_usec_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/data/iterator_gap",
     "The time (in microseconds) between a tf.data iterator responding to a "
     "`GetNext()` request and receiving the next `GetNext()` request."},
    // Buckets of 0.1ms, 0.2ms, 0.4ms, ..., 2s.
    {monitoring::Buckets::Exponential(100, 2, 12)});

auto* tf_data_optimization_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/optimization", "tf.data optimization", "name");

auto* tf_data_service_workers_created_counter =
    monitoring::Counter<0>::New("/tensorflow/data/service/workers_created",
                                "Number of tf.data service workers created");

auto* tf_data_service_jobs_created_counter = monitoring::Counter<2>::New(
    "/tensorflow/data/service/jobs_created", "Number of tf.data service jobs.",
    "processing_mode", "coordinated_read");

auto* tf_data_service_client_iterators_counter = monitoring::Counter<4>::New(
    "/tensorflow/data/service/client_iterators",
    "Number of tf.data service client iterators created.", "worker_uid",
    "deployment_mode", "processing_mode", "is_coordinated_read");

auto* tf_data_service_multi_trainer_cache_queries_counter =
    monitoring::Counter<1>::New(
        "/tensorflow/data/service/multi_trainer_cache_queries",
        "tf.data service multi-client cache queries counter. The result can be "
        "hit or miss.",
        "cache_hit");

auto* tf_data_filename_counter = monitoring::Counter<2>::New(
    "/tensorflow/data/filename", "The file name read by a tf.data Dataset.",
    "name", "filename");

auto* tf_data_model_gauge =
    monitoring::Gauge<std::function<std::string()>, 1>::New(
        "/tensorflow/data/model", "tf.data autotuning model proto.", "id");

auto* tf_data_auto_shard = monitoring::Gauge<int64, 2>::New(
    "/tensorflow/data/autoshard", "tf.data autoshard statistics.", "id",
    "name");

auto* tf_data_auto_shard_rewrite_batch_size_eligible =
    monitoring::Counter<1>::New(
        "/tensorflow/data/autoshard_rewrite_batch_size/eligible",
        "Whether tf.data pipelines that are eligible for autoshard "
        "to rewrite the batch size.",
        "eligible");

auto* tf_data_auto_shard_rewrite_batch_size_reason =
    monitoring::Counter<1>::New(
        "/tensorflow/data/autoshard_rewrite_batch_size/reason",
        "The reasons that tf.data pipelines are ineligible for autoshard "
        "to rewrite the batch size.",
        "reason");

auto* tf_data_autotune_stopping_criteria_counter =
    monitoring::Counter<1>::New("/tensorflow/data/autotune_stopping_criteria",
                                "The number of times each tf.data autotune "
                                "algorithm stopping criterion is met.",
                                "name");

auto* parse_dense_feature_counter = monitoring::Counter<0>::New(
    "/tensorflow/data/dense_feature",
    "The number of dense features parsed by ops for parsing tf.Example.");

auto* parse_sparse_feature_counter = monitoring::Counter<0>::New(
    "/tensorflow/data/sparse_feature",
    "The number of sparse features parsed by ops for parsing tf.Example.");

auto* parse_ragged_feature_counter = monitoring::Counter<0>::New(
    "/tensorflow/data/ragged_feature",
    "The number of ragged features parsed by ops for parsing tf.Example.");

auto* build_graph_calls = monitoring::Counter<0>::New(
    "/tensorflow/core/graph_build_calls",
    "The number of times TensorFlow has created a new client graph. "
    "A client graph is a sub-graph of the full graph, induced by a set of "
    "options, including the requested feeds and fetches. It includes time "
    "spent optimizing the graph with Grappler, and time spent pruning the "
    "sub-graph.");

auto* build_graph_time_usecs = monitoring::Counter<0>::New(
    "/tensorflow/core/graph_build_time_usecs",
    "The amount of time TensorFlow has spent creating new client graphs in "
    "microseconds. "
    "A client graph is a sub-graph of the full graph, induced by a set of "
    "options, including the requested feeds and fetches. It includes time "
    "spent optimizing the graph with Grappler, and time spent pruning the "
    "sub-graph.");

auto* xla_compilations = monitoring::Counter<0>::New(
    "/tensorflow/core/xla_compilations",
    "The number of XLA compilations used to collect "
    "/tensorflow/core/xla_compilation_time_usecs");

auto* xla_compilation_time_usecs = monitoring::Counter<0>::New(
    "/tensorflow/core/xla_compilation_time_usecs",
    "The total time spent on compiling XLA graphs in microseconds.");

auto* xla_tpu_spmd_cores_per_replica = monitoring::Counter<1>::New(
    "/tensorflow/tpu/xla_spmd_cores_per_replica",
    "The number of cores used by XLA SPMD-replicated models.", "cores");

auto* bfc_allocator_delay =
    monitoring::Counter<0>::New("/tensorflow/core/bfc_allocator_delay",
                                "The total time spent running each graph "
                                "optimization pass in microseconds.");

auto* tpu_variable_distribution_time_usecs = monitoring::Counter<0>::New(
    "/tensorflow/tpu/variable_distribution_time",
    "Time spent sending variables from primary task to other worker tasks "
    "at the start of a call to TPUExecute.  Timer starts at RunGraph "
    "invocation and ends when TPUExecute args are ready on the current task.");

auto* test_counters =
    monitoring::Counter<2>::New("/tensorflow/core/test_counters",
                                "Counters used for testing.", "name", "label");

}  // namespace

auto* tpu_op_error_counter = monitoring::Counter<2>::New(
    "/tensorflow/tpu/op_error_count",
    "Count the tpu related errors by op and error_type.", "op", "error_type");

monitoring::Counter<2>* GetGraphOptimizationCounter() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_0(mht_0_v, 424, "", "./tensorflow/core/framework/metrics.cc", "GetGraphOptimizationCounter");

  static auto* graph_optimization_counter =
      monitoring::Counter<2>::New("/tensorflow/core/graph_optimization_usecs",
                                  "The total time spent running each graph "
                                  "optimization pass in microseconds.",
                                  "kind", "name");
  return graph_optimization_counter;
}

void RecordTFDataAutotune(const string& name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_1(mht_1_v, 437, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataAutotune");

  tf_data_autotune_counter->GetCell(name)->IncrementBy(1);
}

monitoring::CounterCell* GetTFDataBytesConsumedCounter(const string& name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_2(mht_2_v, 445, "", "./tensorflow/core/framework/metrics.cc", "GetTFDataBytesConsumedCounter");

  return tf_data_bytes_consumed_counter->GetCell(name);
}

monitoring::CounterCell* GetTFDataBytesProducedCounter(const string& name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_3(mht_3_v, 453, "", "./tensorflow/core/framework/metrics.cc", "GetTFDataBytesProducedCounter");

  return tf_data_bytes_produced_counter->GetCell(name);
}

monitoring::CounterCell* GetTFDataBytesReadCounter(const string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_4(mht_4_v, 461, "", "./tensorflow/core/framework/metrics.cc", "GetTFDataBytesReadCounter");

  return tf_data_bytes_read_counter->GetCell(name);
}

monitoring::CounterCell* GetTFDataElementsCounter(const string& name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_5(mht_5_v, 469, "", "./tensorflow/core/framework/metrics.cc", "GetTFDataElementsCounter");

  return tf_data_elements_counter->GetCell(name);
}

monitoring::GaugeCell<std::function<std::string()>>* GetTFDataModelGauge(
    const string& id) {
  return tf_data_model_gauge->GetCell(id);
}

void RecordTFDataBytesFetched(int64_t num_bytes) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_6(mht_6_v, 481, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataBytesFetched");

  tf_data_bytes_fetched_counter->GetCell()->IncrementBy(num_bytes);
}

void RecordTFDataExperiment(const string& name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_7(mht_7_v, 489, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataExperiment");

  tf_data_experiment_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataFingerprint(const string& name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_8(mht_8_v, 497, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataFingerprint");

  tf_data_fingerprint_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataGetNextDuration(uint64 duration_us) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_9(mht_9_v, 504, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataGetNextDuration");

  static auto* tf_data_get_next_duration_cell =
      tf_data_get_next_duration_usecs_histogram->GetCell();
  tf_data_get_next_duration_cell->Add(duration_us);
}

void RecordTFDataAutotuneUsedRamBudgetRatio(const double ratio) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_10(mht_10_v, 513, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataAutotuneUsedRamBudgetRatio");

  static auto* tf_data_used_vs_budget_ratio_histogram_cell =
      tf_data_used_vs_budget_ratio_histogram->GetCell();
  tf_data_used_vs_budget_ratio_histogram_cell->Add(ratio);
}

void RecordTFDataAutotuneMaxBufferBudgetRatio(const double ratio) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_11(mht_11_v, 522, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataAutotuneMaxBufferBudgetRatio");

  static auto* tf_data_buffered_vs_budget_ratio_histogram_cell =
      tf_data_buffered_vs_budget_ratio_histogram->GetCell();
  tf_data_buffered_vs_budget_ratio_histogram_cell->Add(ratio);
}

void RecordTFDataIteratorBusy(uint64 duration_us) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_12(mht_12_v, 531, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataIteratorBusy");

  static auto* tf_data_iterator_busy_cell =
      tf_data_iterator_busy_counter->GetCell();
  tf_data_iterator_busy_cell->IncrementBy(duration_us);
}

void RecordTFDataIteratorLifetime(uint64 duration_us) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_13(mht_13_v, 540, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataIteratorLifetime");

  static auto* tf_data_iterator_lifetime_cell =
      tf_data_iterator_lifetime_counter->GetCell();
  tf_data_iterator_lifetime_cell->IncrementBy(duration_us);
}

void RecordTFDataIteratorGap(uint64 duration_us) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_14(mht_14_v, 549, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataIteratorGap");

  static auto* tf_data_iterator_gap_usec_histogram_cell =
      tf_data_iterator_gap_usec_histogram->GetCell();
  tf_data_iterator_gap_usec_histogram_cell->Add(duration_us);
}

void RecordTFDataOptimization(const string& name, int64_t num_changes) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_15(mht_15_v, 559, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataOptimization");

  tf_data_optimization_counter->GetCell(name)->IncrementBy(num_changes);
}

void RecordTFDataServiceWorkerCreated() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_16(mht_16_v, 566, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataServiceWorkerCreated");

  tf_data_service_workers_created_counter->GetCell()->IncrementBy(1);
}

void RecordTFDataServiceJobsCreated(
    const tensorflow::data::ProcessingModeDef& processing_mode,
    bool is_coordinated_read) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_17(mht_17_v, 575, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataServiceJobsCreated");

  const std::string sharding_policy_str =
      data::ProcessingModeDef::ShardingPolicy_Name(
          processing_mode.sharding_policy());
  const std::string coordinated_read_str =
      is_coordinated_read ? "true" : "false";
  tf_data_service_jobs_created_counter
      ->GetCell(sharding_policy_str, coordinated_read_str)
      ->IncrementBy(1);
}

void RecordTFDataServiceClientIterators(
    int64_t worker_uid, tensorflow::data::DeploymentMode deployment_mode,
    const tensorflow::data::ProcessingModeDef& processing_mode,
    bool is_coordinated_read) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_18(mht_18_v, 592, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataServiceClientIterators");

  const std::string deployment_mode_str =
      tensorflow::data::DeploymentMode_Name(deployment_mode);
  const std::string sharding_policy_str =
      data::ProcessingModeDef::ShardingPolicy_Name(
          processing_mode.sharding_policy());
  const std::string coordinated_read_str =
      is_coordinated_read ? "true" : "false";
  tf_data_service_client_iterators_counter
      ->GetCell(absl::StrCat(worker_uid), deployment_mode_str,
                sharding_policy_str, coordinated_read_str)
      ->IncrementBy(1);
}

void RecordTFDataServiceMultiTrainerCacheQuery(bool cache_hit) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_19(mht_19_v, 609, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataServiceMultiTrainerCacheQuery");

  std::string cache_hit_str = cache_hit ? "true" : "false";
  tf_data_service_multi_trainer_cache_queries_counter->GetCell(cache_hit_str)
      ->IncrementBy(1);
}

void RecordTFDataFilename(const string& name, const string& filename) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("name: \"" + name + "\"");
   mht_20_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_20(mht_20_v, 620, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataFilename");

  tf_data_filename_counter->GetCell(name, filename)->IncrementBy(1);
}

void RecordTFDataAutoShard(const string& id, data::AutoShardPolicy policy,
                           int64 num_workers, int64 num_replicas) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("id: \"" + id + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_21(mht_21_v, 629, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataAutoShard");

  tf_data_auto_shard->GetCell(id, "policy")->Set(static_cast<int64_t>(policy));
  tf_data_auto_shard->GetCell(id, "num_workers")->Set(num_workers);
  tf_data_auto_shard->GetCell(id, "num_replicas")->Set(num_replicas);
}

void RecordTFDataAutoShardRewriteBatchSize(
    bool eligible, const std::vector<string>& ineligible_reason) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_22(mht_22_v, 639, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataAutoShardRewriteBatchSize");

  tf_data_auto_shard_rewrite_batch_size_eligible
      ->GetCell(eligible ? "true" : "false")
      ->IncrementBy(1);
  for (const string& reason : ineligible_reason) {
    tf_data_auto_shard_rewrite_batch_size_reason->GetCell(reason)->IncrementBy(
        1);
  }
}

void RecordTFDataAutotuneStoppingCriteria(const string& name) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_23(mht_23_v, 653, "", "./tensorflow/core/framework/metrics.cc", "RecordTFDataAutotuneStoppingCriteria");

  tf_data_autotune_stopping_criteria_counter->GetCell(name)->IncrementBy(1);
}

void RecordParseDenseFeature(int64 num_features) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_24(mht_24_v, 660, "", "./tensorflow/core/framework/metrics.cc", "RecordParseDenseFeature");

  static auto* parse_dense_feature_counter_cell =
      parse_dense_feature_counter->GetCell();
  parse_dense_feature_counter_cell->IncrementBy(num_features);
}

void RecordParseSparseFeature(int64_t num_features) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_25(mht_25_v, 669, "", "./tensorflow/core/framework/metrics.cc", "RecordParseSparseFeature");

  static auto* parse_sparse_feature_counter_cell =
      parse_sparse_feature_counter->GetCell();
  parse_sparse_feature_counter_cell->IncrementBy(num_features);
}

void RecordParseRaggedFeature(int64_t num_features) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_26(mht_26_v, 678, "", "./tensorflow/core/framework/metrics.cc", "RecordParseRaggedFeature");

  static auto* parse_ragged_feature_counter_cell =
      parse_ragged_feature_counter->GetCell();
  parse_ragged_feature_counter_cell->IncrementBy(num_features);
}

void RecordGraphInputTensors(const size_t size) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_27(mht_27_v, 687, "", "./tensorflow/core/framework/metrics.cc", "RecordGraphInputTensors");

  static auto* graph_run_input_tensor_bytes_cell =
      graph_run_input_tensor_bytes->GetCell();
  graph_run_input_tensor_bytes_cell->Add(size);
}

void RecordGraphOutputTensors(const size_t size) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_28(mht_28_v, 696, "", "./tensorflow/core/framework/metrics.cc", "RecordGraphOutputTensors");

  static auto* graph_run_output_tensor_bytes_cell =
      graph_run_output_tensor_bytes->GetCell();
  graph_run_output_tensor_bytes_cell->Add(size);
}

void RecordTPUXlaSpmdCoresPerReplica(int64_t cores_per_replica) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_29(mht_29_v, 705, "", "./tensorflow/core/framework/metrics.cc", "RecordTPUXlaSpmdCoresPerReplica");

  xla_tpu_spmd_cores_per_replica->GetCell(absl::StrCat(cores_per_replica))
      ->IncrementBy(1);
}

void UpdateGraphExecTime(const uint64 running_time_usecs) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_30(mht_30_v, 713, "", "./tensorflow/core/framework/metrics.cc", "UpdateGraphExecTime");

  if (running_time_usecs > 0) {
    static auto* graph_runs_cell = graph_runs->GetCell();
    static auto* graph_run_time_usecs_cell = graph_run_time_usecs->GetCell();
    static auto* graph_run_time_usecs_histogram_cell =
        graph_run_time_usecs_histogram->GetCell();
    graph_runs_cell->IncrementBy(1);
    graph_run_time_usecs_cell->IncrementBy(running_time_usecs);
    graph_run_time_usecs_histogram_cell->Add(running_time_usecs);
  }
}

void UpdateGraphPendingQueueLength(uint64 len) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_31(mht_31_v, 728, "", "./tensorflow/core/framework/metrics.cc", "UpdateGraphPendingQueueLength");

  static auto* graph_pending_queue_length_cell =
      graph_pending_queue_length_histogram->GetCell();
  graph_pending_queue_length_cell->Add(len);
}

void UpdateGraphBuildTime(const uint64 running_time_usecs) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_32(mht_32_v, 737, "", "./tensorflow/core/framework/metrics.cc", "UpdateGraphBuildTime");

  if (running_time_usecs > 0) {
    static auto* build_graph_calls_cell = build_graph_calls->GetCell();
    static auto* build_graph_time_usecs_cell =
        build_graph_time_usecs->GetCell();
    build_graph_calls_cell->IncrementBy(1);
    build_graph_time_usecs_cell->IncrementBy(running_time_usecs);
  }
}

void UpdateTpuVariableDistributionTime(const uint64 distribution_time_usecs) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_33(mht_33_v, 750, "", "./tensorflow/core/framework/metrics.cc", "UpdateTpuVariableDistributionTime");

  if (distribution_time_usecs > 0) {
    tpu_variable_distribution_time_usecs->GetCell()->IncrementBy(
        distribution_time_usecs);
  }
}

void UpdateXlaCompilationTime(const uint64 compilation_time_usecs) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_34(mht_34_v, 760, "", "./tensorflow/core/framework/metrics.cc", "UpdateXlaCompilationTime");

  if (compilation_time_usecs > 0) {
    static auto* xla_compilations_cell = xla_compilations->GetCell();
    static auto* xla_compilation_time_usecs_cell =
        xla_compilation_time_usecs->GetCell();
    xla_compilations_cell->IncrementBy(1);
    xla_compilation_time_usecs_cell->IncrementBy(compilation_time_usecs);
  }
}

void UpdateBfcAllocatorDelayTime(const uint64 delay_usecs) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_35(mht_35_v, 773, "", "./tensorflow/core/framework/metrics.cc", "UpdateBfcAllocatorDelayTime");

  static auto* bfc_allocator_delay_cell = bfc_allocator_delay->GetCell();
  if (delay_usecs > 0) {
    bfc_allocator_delay_cell->IncrementBy(delay_usecs);
  }
}

void RecordUnusedOutput(const string& op_name) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_36(mht_36_v, 784, "", "./tensorflow/core/framework/metrics.cc", "RecordUnusedOutput");

  graph_unused_outputs->GetCell(op_name)->IncrementBy(1);
}

void IncrementTestCounter(const string& name, const string& label) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("name: \"" + name + "\"");
   mht_37_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_37(mht_37_v, 793, "", "./tensorflow/core/framework/metrics.cc", "IncrementTestCounter");

  test_counters->GetCell(name, label)->IncrementBy(1);
}

const monitoring::CounterCell* TestCounter(const string& name,
                                           const string& label) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("name: \"" + name + "\"");
   mht_38_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_38(mht_38_v, 803, "", "./tensorflow/core/framework/metrics.cc", "TestCounter");

  return test_counters->GetCell(name, label);
}

TestDelta::TestDelta(const string& name, const string& label)
    : cell_(TestCounter(name, label)) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("name: \"" + name + "\"");
   mht_39_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_39(mht_39_v, 813, "", "./tensorflow/core/framework/metrics.cc", "TestDelta::TestDelta");

  Reset();
}

void TestDelta::Reset() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_40(mht_40_v, 820, "", "./tensorflow/core/framework/metrics.cc", "TestDelta::Reset");
 last_value_ = cell_->value(); }

int64 TestDelta::Get() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_41(mht_41_v, 825, "", "./tensorflow/core/framework/metrics.cc", "TestDelta::Get");
 return cell_->value() - last_value_; }

void UpdateTfMlirGraphOptimizationPassStateCounter(
    const std::string& pass_state, const std::string& processing_state) {
   std::vector<std::string> mht_42_v;
   mht_42_v.push_back("pass_state: \"" + pass_state + "\"");
   mht_42_v.push_back("processing_state: \"" + processing_state + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_42(mht_42_v, 833, "", "./tensorflow/core/framework/metrics.cc", "UpdateTfMlirGraphOptimizationPassStateCounter");

  static auto* metric = monitoring::Counter<2>::New(
      "/tensorflow/core/tf_mlir_update_graph_optimization_pass_state_counter",
      "Tracks changes in a graph's UpdateTfMlirGraphOptimizationPassState",
      "PassState", "ProcessingState");

  metric->GetCell(pass_state, processing_state)->IncrementBy(1);
}

void UpdateTfMlirBridgeFirstPhaseCounter(const std::string& device_type,
                                         const std::string& bridge_version,
                                         bool fallback_enabled,
                                         const std::string& result) {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("device_type: \"" + device_type + "\"");
   mht_43_v.push_back("bridge_version: \"" + bridge_version + "\"");
   mht_43_v.push_back("result: \"" + result + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_43(mht_43_v, 851, "", "./tensorflow/core/framework/metrics.cc", "UpdateTfMlirBridgeFirstPhaseCounter");

  static auto* metric = monitoring::Counter<4>::New(
      "/tensorflow/core/tf_mlir_bridge_first_phase_count",
      "Tracks processing state in first phase of mlir bridge", "device",
      "version", "fallback", "result");
  std::string fallback_status =
      fallback_enabled ? "fallback_enabled" : "fallback_disabled";
  metric->GetCell(device_type, bridge_version, fallback_status, result)
      ->IncrementBy(1);
}

void UpdateTpuErrorCounter(const string& op, const string& error_type) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("op: \"" + op + "\"");
   mht_44_v.push_back("error_type: \"" + error_type + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSmetricsDTcc mht_44(mht_44_v, 867, "", "./tensorflow/core/framework/metrics.cc", "UpdateTpuErrorCounter");

  tpu_op_error_counter->GetCell(op, error_type)->IncrementBy(1);
}

}  // namespace metrics
}  // namespace tensorflow
