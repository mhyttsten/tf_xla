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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc() {
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

#include "tensorflow/core/profiler/convert/op_stats_to_overview_page.h"

#include <string>

#include "google/protobuf/any.pb.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_function.pb.h"
#include "tensorflow/core/profiler/utils/diagnostics.h"
#include "tensorflow/core/profiler/utils/format_utils.h"
#include "tensorflow/core/profiler/utils/hardware_type_utils.h"
#include "tensorflow/core/profiler/utils/html_utils.h"
#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

// If the use of low-precision ops is less than this percentage threshold, a
// statement of suggestion will be made.
constexpr double kLowPrecisionPercentThreshold = 10;

struct TfFunctionInfo {
  absl::string_view function_name;
  double expensive_call_percent;
};

OverviewPageTip MakeOverviewPageTip(std::string text) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_0(mht_0_v, 226, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "MakeOverviewPageTip");

  OverviewPageTip tip;
  tip.set_link(std::move(text));
  return tip;
}

// Makes a recommendation for looking up a document.
// doc_url is expected to be already be escaped suitably for use in an HTML
// attribute.
OverviewPageTip MakeOverviewPageTipDocLink(absl::string_view doc_url,
                                           absl::string_view text) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("doc_url: \"" + std::string(doc_url.data(), doc_url.size()) + "\"");
   mht_1_v.push_back("text: \"" + std::string(text.data(), text.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "MakeOverviewPageTipDocLink");

  return MakeOverviewPageTip(AnchorElement(doc_url, text));
}

void ComputeHostTips(OverviewPageRecommendation* re) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "ComputeHostTips");

  *re->add_host_tips() = MakeOverviewPageTip(
      "input_pipeline_analyzer (especially Section 3 for the breakdown of "
      "input operations on the Host)");
  *re->add_host_tips() = MakeOverviewPageTip(
      "tf_data_bottleneck_analysis (find the bottleneck in the tf.data input "
      "pipeline)");
  *re->add_host_tips() = MakeOverviewPageTip(
      "trace_viewer (look at the activities on the timeline of each Host "
      "Thread near the bottom of the trace view)");
}

void ComputeDeviceTips(HardwareType hardware_type,
                       OverviewPageRecommendation* re) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_3(mht_3_v, 264, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "ComputeDeviceTips");

  absl::string_view device_name = HardwareType_Name(hardware_type);
  absl::string_view timeline_name = device_name;
  absl::string_view op_stats_toolname = "tensorflow_stats";
  if (hardware_type == tensorflow::profiler::TPU) {
    timeline_name = "TPU core";
    op_stats_toolname = "op_profile";
  }
  *re->add_device_tips() = MakeOverviewPageTip(
      absl::StrCat(op_stats_toolname,
                   " (identify the time-consuming operations "
                   "executed on the ",
                   device_name, ")"));
  *re->add_device_tips() = MakeOverviewPageTip(absl::StrCat(
      "trace_viewer (look at the activities on the timeline of each ",
      timeline_name, " in the trace view)"));
}

void ComputeFaqTips(OverviewPageRecommendation* re) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_4(mht_4_v, 285, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "ComputeFaqTips");

  *re->add_faq_tips() = MakeOverviewPageTip("Refer to the TF2 Profiler FAQ");
}

void ComputeDocumentationTips(OverviewPageRecommendation* re) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_5(mht_5_v, 292, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "ComputeDocumentationTips");

  *re->add_documentation_tips() = MakeOverviewPageTipDocLink(
      "https://www.tensorflow.org/guide/data_performance_analysis",
      "Analyze tf.data performance with the TF Profiler");
  *re->add_documentation_tips() = MakeOverviewPageTipDocLink(
      "https://www.tensorflow.org/guide/"
      "data_performance",
      "Better performance with the tf.data API");
}

std::string GeneratePrecisionStatement(const PrecisionStats& precision_stats) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_6(mht_6_v, 305, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "GeneratePrecisionStatement");

  uint64 total_compute_ps =
      precision_stats.compute_16bit_ps() + precision_stats.compute_32bit_ps();
  if (total_compute_ps > 0) {
    double percent_16bit =
        (100.0 * precision_stats.compute_16bit_ps()) / total_compute_ps;
    if (percent_16bit < kLowPrecisionPercentThreshold) {
      return absl::StrCat(
          "Only ", OneDigit(percent_16bit),
          "% of device computation is 16 bit. So you might want to replace "
          "more 32-bit Ops by 16-bit Ops to improve performance (if the "
          "reduced accuracy is acceptable).");
    }
  }
  return "";
}

}  // namespace

void SetCommonRecommendation(
    absl::string_view input_classification, absl::string_view input_statement,
    absl::string_view output_statement, HardwareType hardware_type,
    absl::string_view tf_function_statement_html,
    absl::string_view eager_statement_html,
    absl::string_view outside_compilation_statement_html,
    OverviewPageRecommendation* re) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("input_classification: \"" + std::string(input_classification.data(), input_classification.size()) + "\"");
   mht_7_v.push_back("input_statement: \"" + std::string(input_statement.data(), input_statement.size()) + "\"");
   mht_7_v.push_back("output_statement: \"" + std::string(output_statement.data(), output_statement.size()) + "\"");
   mht_7_v.push_back("tf_function_statement_html: \"" + std::string(tf_function_statement_html.data(), tf_function_statement_html.size()) + "\"");
   mht_7_v.push_back("eager_statement_html: \"" + std::string(eager_statement_html.data(), eager_statement_html.size()) + "\"");
   mht_7_v.push_back("outside_compilation_statement_html: \"" + std::string(outside_compilation_statement_html.data(), outside_compilation_statement_html.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_7(mht_7_v, 339, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "SetCommonRecommendation");

  re->set_bottleneck(std::string(input_classification));
  re->set_statement(std::string(input_statement));
  re->set_output_statement(std::string(output_statement));
  re->set_tf_function_statement_html(std::string(tf_function_statement_html));
  re->set_eager_statement_html(std::string(eager_statement_html));
  re->set_outside_compilation_statement_html(
      std::string(outside_compilation_statement_html));
  ComputeHostTips(re);
  ComputeDeviceTips(hardware_type, re);
  ComputeDocumentationTips(re);
  ComputeFaqTips(re);
}

OverviewPageRecommendation ComputeGenericRecommendation(
    const BottleneckAnalysis& bottleneck,
    const PrecisionStats& precision_stats) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_8(mht_8_v, 358, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "ComputeGenericRecommendation");

  OverviewPageRecommendation re;
  GenericRecommendation generic;
  generic.set_device_collectives_bottleneck(
      bottleneck.device_collectives_classification());
  generic.set_device_collectives_statement(
      bottleneck.device_collectives_statement());
  generic.set_kernel_launch_bottleneck(
      bottleneck.kernel_launch_classification());
  generic.set_kernel_launch_statement(bottleneck.kernel_launch_statement());
  generic.set_all_other_bottleneck(bottleneck.all_other_classification());
  generic.set_all_other_statement(bottleneck.all_other_statement());
  generic.set_precision_statement(GeneratePrecisionStatement(precision_stats));
  re.mutable_recommendation()->PackFrom(generic);
  return re;
}

OverviewPageAnalysis ComputeAnalysisResult(const OpStats& op_stats) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_9(mht_9_v, 378, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "ComputeAnalysisResult");

  OverviewPageAnalysis analysis;
  OpMetricsDb device_tf_op_metrics_db = CreateTfMetricsDbFromDeviceOpMetricsDb(
      op_stats.device_op_metrics_db(), /*with_idle=*/false);
  KernelStatsByOpName kernel_stats_by_op_name =
      GroupKernelReportsByOpName(op_stats.kernel_stats_db());
  uint64 total_device_time_ps = device_tf_op_metrics_db.total_time_ps();
  constexpr int kNumTopOpsShown = 10;
  double device_cumulative_fraction = 0.0;
  for (const OpMetrics* metrics :
       SortedOpMetricsDb(device_tf_op_metrics_db, kNumTopOpsShown)) {
    OverviewTfOp* op = analysis.add_top_device_ops();
    op->set_name(metrics->name());
    op->set_category(metrics->category());
    op->set_self_time_fraction(
        SafeDivide(metrics->self_time_ps(), total_device_time_ps));
    device_cumulative_fraction += op->self_time_fraction();
    op->set_cumulative_time_fraction(device_cumulative_fraction);
    op->set_flop_rate(
        SafeDivide(metrics->flops(), PicoToNano(metrics->time_ps())));
    auto iter = kernel_stats_by_op_name.find(op->name());
    if (iter != kernel_stats_by_op_name.end()) {
      op->set_is_op_tensorcore_eligible(
          iter->second.is_op_tensor_core_eligible);
      op->set_is_op_using_tensorcore(iter->second.tensor_core_duration_ns != 0);
    }
  }
  uint64 total_device_compute_ps =
      op_stats.device_op_metrics_db().precision_stats().compute_16bit_ps() +
      op_stats.device_op_metrics_db().precision_stats().compute_32bit_ps();
  analysis.set_device_compute_16bit_percent(
      100.0 *
      SafeDivide(
          op_stats.device_op_metrics_db().precision_stats().compute_16bit_ps(),
          total_device_compute_ps));
  analysis.set_device_compute_32bit_percent(
      100.0 *
      SafeDivide(
          op_stats.device_op_metrics_db().precision_stats().compute_32bit_ps(),
          total_device_compute_ps));

  uint64 num_host_tf_ops = 0;
  uint64 total_host_op_time_ps_exclude_idle = 0;
  uint64 eager_host_op_time_ps = 0;
  for (const OpMetrics& metrics : op_stats.host_op_metrics_db().metrics_db()) {
    num_host_tf_ops += metrics.occurrences();
    if (!IsIdleOp(metrics)) {
      total_host_op_time_ps_exclude_idle += metrics.self_time_ps();
      if (metrics.is_eager()) eager_host_op_time_ps += metrics.self_time_ps();
    }
  }
  uint64 num_device_tf_ops = 0;
  uint64 total_device_op_time_ps_exclude_idle = 0;
  uint64 eager_device_op_time_ps = 0;
  for (const OpMetrics& metrics : device_tf_op_metrics_db.metrics_db()) {
    num_device_tf_ops += metrics.occurrences();
    if (!IsIdleOp(metrics)) {
      total_device_op_time_ps_exclude_idle += metrics.self_time_ps();
      if (metrics.is_eager()) eager_device_op_time_ps += metrics.self_time_ps();
    }
  }
  // Figures out outside_compilation time from
  // op_stats.device_op_metrics_db().metrics_db(). We don't use the
  // {metrics.provenance(), metrics.name()} from
  // device_tf_op_metrics_db.metrics_db(), because metrics.provenance() there is
  // not set and metrics.name() can be either HLO-Op name or TF-Op name, which
  // will confuse IsOutsideCompilationOp().
  uint64 outside_compilation_device_op_time_ps = 0;
  for (const OpMetrics& metrics :
       op_stats.device_op_metrics_db().metrics_db()) {
    if (!IsOutsideCompilationOp(metrics.provenance(), metrics.long_name()))
      continue;
    outside_compilation_device_op_time_ps += metrics.self_time_ps();
  }
  uint64 num_total_tf_ops = num_host_tf_ops + num_device_tf_ops;
  analysis.set_host_tf_op_percent(
      100.0 * SafeDivide(num_host_tf_ops, num_total_tf_ops));
  analysis.set_device_tf_op_percent(
      100.0 * SafeDivide(num_device_tf_ops, num_total_tf_ops));
  analysis.set_host_trace_level(op_stats.run_environment().host_trace_level());
  analysis.set_host_op_time_eager_percent(
      100.0 *
      SafeDivide(eager_host_op_time_ps, total_host_op_time_ps_exclude_idle));
  analysis.set_device_op_time_eager_percent(
      100.0 * SafeDivide(eager_device_op_time_ps,
                         total_device_op_time_ps_exclude_idle));
  analysis.set_device_op_time_outside_compilation_percent(
      100.0 * SafeDivide(outside_compilation_device_op_time_ps,
                         total_device_op_time_ps_exclude_idle));
  return analysis;
}

// Converts from HostIndependentJobInfo to OverviewPageHostIndependentJobInfo.
OverviewPageHostIndependentJobInfo ToOverviewPageHostIndependentJobInfo(
    const HostIndependentJobInfoResult& host_independent_job_info) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_10(mht_10_v, 475, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "ToOverviewPageHostIndependentJobInfo");

  OverviewPageHostIndependentJobInfo result;
  result.set_change_list(host_independent_job_info.change_list());
  result.set_build_time(host_independent_job_info.build_time());
  result.set_build_target(host_independent_job_info.build_target());
  result.set_profile_duration_ms(
      host_independent_job_info.profile_duration_ms());
  return result;
}

// Converts from HostDependentJobInfo to OverviewPageHostDependentJobInfo.
OverviewPageHostDependentJobInfo ToOverviewPageHostDependentJobInfo(
    const HostDependentJobInfoResult& host_dependent_job_info) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_11(mht_11_v, 490, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "ToOverviewPageHostDependentJobInfo");

  OverviewPageHostDependentJobInfo result;
  result.set_host_id(host_dependent_job_info.host_id());
  result.set_command_line(host_dependent_job_info.command_line());
  result.set_start_time(host_dependent_job_info.start_time());
  result.set_bns_address(host_dependent_job_info.bns_address());
  result.set_profile_time_ns(host_dependent_job_info.profile_time_ns());
  return result;
}

OverviewPageRunEnvironment ComputeRunEnvironment(
    const RunEnvironment& run_environment) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_12(mht_12_v, 504, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "ComputeRunEnvironment");

  OverviewPageRunEnvironment re;
  re.set_host_count(run_environment.host_count());
  re.set_task_count(run_environment.task_count());
  re.set_device_type(run_environment.device_type());
  re.set_device_core_count(run_environment.device_core_count());
  re.set_per_core_batch_size(run_environment.per_core_batch_size());
  re.set_replica_count(run_environment.replica_count());
  re.set_num_cores_per_replica(run_environment.num_cores_per_replica());
  *re.mutable_host_independent_job_info() =
      ToOverviewPageHostIndependentJobInfo(
          run_environment.host_independent_job_info());
  for (const auto& host_dependent_job_info :
       run_environment.host_dependent_job_info()) {
    *re.add_host_dependent_job_info() =
        ToOverviewPageHostDependentJobInfo(host_dependent_job_info);
  }
  return re;
}

std::string TfFunctionRecommendationHtml(const TfFunctionDb& tf_function_db) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_13(mht_13_v, 527, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "TfFunctionRecommendationHtml");

  std::vector<TfFunctionInfo> candidates;
  for (const auto& name_fun : tf_function_db.tf_functions()) {
    const auto& fun = name_fun.second;
    if (fun.expensive_call_percent() >= kTfFunctionReportThresholdInPercent) {
      candidates.push_back({name_fun.first, fun.expensive_call_percent()});
    }
  }
  if (candidates.empty()) return "";
  auto cmp = [](const TfFunctionInfo& a, const TfFunctionInfo& b) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_14(mht_14_v, 539, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "lambda");

    return a.expensive_call_percent > b.expensive_call_percent;
  };
  // Sorts candidates in descending order of expensive_call_percent.
  absl::c_sort(candidates, cmp);
  std::string expensive_functions = "";
  auto num_functions_shown = std::min(
      static_cast<decltype(candidates)::size_type>(3), candidates.size());

  for (decltype(candidates)::size_type i = 0; i < num_functions_shown; i++) {
    if (i > 0) absl::StrAppend(&expensive_functions, ", ");
    absl::StrAppend(&expensive_functions, "\"", candidates[i].function_name,
                    "\"");
  }
  if (candidates.size() > num_functions_shown)
    absl::StrAppend(&expensive_functions, " and more");
  return absl::StrCat("Expensive tf-functions detected (", expensive_functions,
                      ") due to either retracing or eager execution.");
}

std::string EagerRecommendationHtml(double host_op_time_eager_percent,
                                    double device_op_time_eager_percent) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_15(mht_15_v, 563, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "EagerRecommendationHtml");

  std::string recommendation = "";
  if (host_op_time_eager_percent > kEagerReportThresholdInPercent)
    absl::StrAppend(&recommendation, OneDigit(host_op_time_eager_percent),
                    "% of Op time on the host used eager execution. ");
  if (device_op_time_eager_percent > kEagerReportThresholdInPercent)
    absl::StrAppend(&recommendation, OneDigit(device_op_time_eager_percent),
                    "% of Op time on the device used eager execution. ");
  if (!recommendation.empty())
    absl::StrAppend(&recommendation, "Performance could be improved with ",
                    AnchorElement("https://www.tensorflow.org/guide/function",
                                  "tf.function."));
  return recommendation;
}

std::string OutsideCompilationRecommendationHtml(
    double device_op_time_outside_compilation_percent) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_16(mht_16_v, 582, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "OutsideCompilationRecommendationHtml");

  if (device_op_time_outside_compilation_percent <=
      kOutsideCompilationThresholdInPercent)
    return "";
  return absl::StrCat(
      OneDigit(device_op_time_outside_compilation_percent),
      " % of Op time on the device are for outside compilation. Performance "
      "could be improved by avoiding outside compilation.");
}

OverviewPage ConvertOpStatsToOverviewPage(const OpStats& op_stats) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_overview_pageDTcc mht_17(mht_17_v, 595, "", "./tensorflow/core/profiler/convert/op_stats_to_overview_page.cc", "ConvertOpStatsToOverviewPage");

  OverviewPage overview_page;
  *overview_page.mutable_run_environment() =
      ComputeRunEnvironment(op_stats.run_environment());
  *overview_page.mutable_analysis() = ComputeAnalysisResult(op_stats);
  *overview_page.mutable_input_analysis() =
      ConvertOpStatsToInputPipelineAnalysis(op_stats);
  BottleneckAnalysis bottleneck = ComputeBottleneckAnalysis(
      overview_page.input_analysis().input_time_breakdown(),
      overview_page.input_analysis().step_details());
  *overview_page.mutable_recommendation() = ComputeGenericRecommendation(
      bottleneck, op_stats.device_op_metrics_db().precision_stats());
  SetCommonRecommendation(
      bottleneck.input_classification(), bottleneck.input_statement(), "",
      ParseHardwareType(op_stats.run_environment().device_type()),
      TfFunctionRecommendationHtml(op_stats.tf_function_db()),
      EagerRecommendationHtml(
          overview_page.analysis().host_op_time_eager_percent(),
          overview_page.analysis().device_op_time_eager_percent()),
      OutsideCompilationRecommendationHtml(
          overview_page.analysis()
              .device_op_time_outside_compilation_percent()),
      overview_page.mutable_recommendation());
  PopulateOverviewDiagnostics(op_stats, overview_page.mutable_diagnostics());
  return overview_page;
}

}  // namespace profiler
}  // namespace tensorflow
