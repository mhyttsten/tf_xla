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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc() {
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

#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"

namespace tensorflow {
namespace profiler {

namespace {

// The maximum number of Kernels displayed on Kernel Stats page.
const int kMaxNumOfKernels = 1000;

// A list of patterns to help determine if a kernel uses Tensor Core.
// A kernel uses Tensor Core if its kernel name contains any of these patterns.
// Some examples of kernel names: volta_h884gemm, turing_fp16_s1688cudnn_fp16
constexpr absl::string_view kTensorCoreKernelNamePatterns[] = {
    "16816",
    "c1688",
    "conv1x1",
    "conv2d_c1_k1",
    "dgrad_1x1_stride_2x2",
    "direct_group",
    "first_layer_wgrad_kernel",
    "h1688",
    "h884",
    "hmma",
    "i16832",
    "i8816",
    "s884",
    "s1688",
    "xmma_gemm",
    "xmma_implicit_gemm",
    "xmma_sparse_conv",
    "xmma_sparse_gemm",
    "xmma_warp_specialized_implicit_gemm"};

}  // namespace

void ParseKernelLaunchParams(absl::string_view xstat_kernel_details,
                             KernelReport* kernel) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("xstat_kernel_details: \"" + std::string(xstat_kernel_details.data(), xstat_kernel_details.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc mht_0(mht_0_v, 237, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.cc", "ParseKernelLaunchParams");

  const std::vector<absl::string_view> params =
      absl::StrSplit(xstat_kernel_details, absl::ByAnyChar(" \n"));

  constexpr uint32 kNumDimensions = 3;
  for (uint32 dim = 0; dim < kNumDimensions; ++dim) {
    kernel->add_block_dim(1);
    kernel->add_grid_dim(1);
  }

  // Process tokens.
  for (const auto& param : params) {
    const std::vector<absl::string_view> key_value = absl::StrSplit(param, ':');
    if (key_value.size() != 2) {
      // Unrecognized token.
      continue;
    }
    absl::string_view key = key_value[0];
    absl::string_view value_str = key_value[1];
    uint32 value = 0;
    double pct = 0.0;
    // Cases that consume a pair of tokens "key:value".
    if (key == "regs" && absl::SimpleAtoi(value_str, &value)) {
      kernel->set_registers_per_thread(value);
    } else if (key == "static_shared" && absl::SimpleAtoi(value_str, &value)) {
      kernel->set_static_shmem_bytes(value);
    } else if (key == "dynamic_shared" && absl::SimpleAtoi(value_str, &value)) {
      kernel->set_dynamic_shmem_bytes(value);
    } else if (key == "block") {
      const std::vector<absl::string_view>& block =
          absl::StrSplit(value_str, ',');
      uint32 tmp[3];
      if (block.size() == 3 && absl::SimpleAtoi(block[0], &tmp[0]) &&
          absl::SimpleAtoi(block[1], &tmp[1]) &&
          absl::SimpleAtoi(block[2], &tmp[2])) {
        std::copy_n(tmp, 3, kernel->mutable_block_dim()->begin());
      }
    } else if (key == "grid") {
      const std::vector<absl::string_view>& grid =
          absl::StrSplit(value_str, ',');
      uint32 tmp[3];
      if (grid.size() == 3 && absl::SimpleAtoi(grid[0], &tmp[0]) &&
          absl::SimpleAtoi(grid[1], &tmp[1]) &&
          absl::SimpleAtoi(grid[2], &tmp[2])) {
        std::copy_n(tmp, 3, kernel->mutable_grid_dim()->begin());
      }
    } else if (key == "occ_pct" && absl::SimpleAtod(value_str, &pct)) {
      kernel->set_occupancy_pct(pct);
    }
  }
}

bool IsKernelUsingTensorCore(absl::string_view kernel_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("kernel_name: \"" + std::string(kernel_name.data(), kernel_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc mht_1(mht_1_v, 293, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.cc", "IsKernelUsingTensorCore");

  VLOG(1) << "kernel name: " << kernel_name;
  for (absl::string_view pattern : kTensorCoreKernelNamePatterns) {
    if (absl::StrContains(kernel_name, pattern)) {
      return true;
    }
  }
  return false;
}

// This list is not exhaustive.
bool IsOpTensorCoreEligible(absl::string_view tf_op_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tf_op_name: \"" + std::string(tf_op_name.data(), tf_op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc mht_2(mht_2_v, 308, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.cc", "IsOpTensorCoreEligible");

  // Disable formatting to keep inline comments vertically aligned.
  // clang-format off
  return false
      // Using EndsWith to match Fused operations.
      || absl::EndsWith(tf_op_name, "Conv2D")
      || absl::EndsWith(tf_op_name, "Conv2DBackpropFilter")
      || absl::EndsWith(tf_op_name, "Conv2DBackpropInput")
      || absl::EndsWith(tf_op_name, "Conv3D")
      || absl::EndsWith(tf_op_name, "DepthwiseConv2dNative")
      || absl::EndsWith(tf_op_name, "DepthwiseConv2dNativeBackpropFilter")
      || absl::EndsWith(tf_op_name, "DepthwiseConv2dNativeBackpropInput")
      // Using Contains to match V2/V3 suffixes.
      || absl::StrContains(tf_op_name, "BatchMatMul")
      // MatMul requires exact matching.
      || absl::EndsWith(tf_op_name, "/MatMul")
      || absl::EndsWith(tf_op_name, "FusedMatMul")
      // cuDNN operations.
      || absl::EndsWith(tf_op_name, "/CudnnRNN")
      || absl::StrContains(tf_op_name, "CudnnRNNV")
      || absl::StrContains(tf_op_name, "CudnnRNNForward")
      || absl::StrContains(tf_op_name, "CudnnRNNBackprop")
      // Special cases.
      || absl::EndsWith(tf_op_name, "XlaDot")
      || absl::EndsWith(tf_op_name, "XlaDotV2");
  // clang-format on
}

bool IsEinsumTensorCoreEligible(absl::string_view equation) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("equation: \"" + std::string(equation.data(), equation.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc mht_3(mht_3_v, 340, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.cc", "IsEinsumTensorCoreEligible");

  if (equation.empty()) {
    return false;
  }
  const std::vector<absl::string_view> input_output =
      absl::StrSplit(equation, "->");
  if (input_output.size() != 2) {
    return false;
  }
  const std::vector<absl::string_view> lhs_rhs =
      absl::StrSplit(input_output[0], ',');
  return lhs_rhs.size() == 2;
}

bool KernelReportLessThanComparator::operator()(const KernelReport& lhs,
                                                const KernelReport& rhs) const {
  // Disable formatting to keep vertical alignment for better readability,
  // and make it easier to reorder columns.
  // clang-format off
  auto lhs_tuple = std::make_tuple(
      lhs.name(),
      lhs.grid_dim(0),
      lhs.grid_dim(1),
      lhs.grid_dim(2),
      lhs.block_dim(0),
      lhs.block_dim(1),
      lhs.block_dim(2),
      lhs.registers_per_thread(),
      lhs.static_shmem_bytes(),
      lhs.dynamic_shmem_bytes(),
      lhs.is_kernel_using_tensor_core(),
      lhs.is_op_tensor_core_eligible(),
      lhs.op_name());

  auto rhs_tuple = std::make_tuple(
      rhs.name(),
      rhs.grid_dim(0),
      rhs.grid_dim(1),
      rhs.grid_dim(2),
      rhs.block_dim(0),
      rhs.block_dim(1),
      rhs.block_dim(2),
      rhs.registers_per_thread(),
      rhs.static_shmem_bytes(),
      rhs.dynamic_shmem_bytes(),
      rhs.is_kernel_using_tensor_core(),
      rhs.is_op_tensor_core_eligible(),
      rhs.op_name());
  // clang-format on
  return lhs_tuple < rhs_tuple;
}

bool KernelReportEqualToComparator::operator()(const KernelReport& lhs,
                                               const KernelReport& rhs) const {
  // Disable formatting to keep vertical alignment for better readability,
  // and make it easier to reorder columns.
  // clang-format off
  // Put the most expensive string comparisons last.
  return (
      lhs.is_kernel_using_tensor_core() == rhs.is_kernel_using_tensor_core() &&
      lhs.is_op_tensor_core_eligible() == rhs.is_op_tensor_core_eligible() &&
      lhs.block_dim(0) == rhs.block_dim(0) &&
      lhs.block_dim(1) == rhs.block_dim(1) &&
      lhs.block_dim(2) == rhs.block_dim(2) &&
      lhs.grid_dim(0) == rhs.grid_dim(0) &&
      lhs.grid_dim(1) == rhs.grid_dim(1) &&
      lhs.grid_dim(2) == rhs.grid_dim(2) &&
      lhs.registers_per_thread() == rhs.registers_per_thread() &&
      lhs.static_shmem_bytes() == rhs.static_shmem_bytes() &&
      lhs.dynamic_shmem_bytes() == rhs.dynamic_shmem_bytes() &&
      lhs.name() == rhs.name() &&
      lhs.op_name() == rhs.op_name());
  // clang-format on
}

void SortAndKeepTopKDurationKernelReportsInDb(KernelStatsDb* kernel_stats_db) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc mht_4(mht_4_v, 418, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.cc", "SortAndKeepTopKDurationKernelReportsInDb");

  auto comp = [](const KernelReport& lhs, const KernelReport& rhs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc mht_5(mht_5_v, 422, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.cc", "lambda");

    return lhs.total_duration_ns() > rhs.total_duration_ns() ||
           (lhs.total_duration_ns() == rhs.total_duration_ns() &&
            KernelReportLessThanComparator()(lhs, rhs));
  };

  // Sort and keep at most <kMaxNumOfKernels> kernel reports.
  if (kernel_stats_db->reports_size() > kMaxNumOfKernels) {
    std::partial_sort(
        kernel_stats_db->mutable_reports()->begin(),
        kernel_stats_db->mutable_reports()->begin() + kMaxNumOfKernels,
        kernel_stats_db->mutable_reports()->end(), comp);
    kernel_stats_db->mutable_reports()->erase(
        kernel_stats_db->mutable_reports()->begin() + kMaxNumOfKernels,
        kernel_stats_db->mutable_reports()->end());
  } else {
    std::sort(kernel_stats_db->mutable_reports()->begin(),
              kernel_stats_db->mutable_reports()->end(), comp);
  }
}

void CopyTopKDurationKernelReportsToDb(const KernelReportMap& reports,
                                       KernelStatsDb* dst) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc mht_6(mht_6_v, 447, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.cc", "CopyTopKDurationKernelReportsToDb");

  std::vector<std::pair<const KernelReport*, const KernelReportValue*>>
      kernels_to_sort;
  kernels_to_sort.reserve(reports.size());
  for (const auto& report_value : reports) {
    kernels_to_sort.push_back(
        std::make_pair(&report_value.first, &report_value.second));
  }

  auto comp =
      [](const std::pair<const KernelReport*, const KernelReportValue*>& lhs,
         const std::pair<const KernelReport*, const KernelReportValue*>& rhs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc mht_7(mht_7_v, 461, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.cc", "lambda");

        return lhs.second->total_duration_ns > rhs.second->total_duration_ns ||
               (lhs.second->total_duration_ns ==
                    rhs.second->total_duration_ns &&
                KernelReportLessThanComparator()(*lhs.first, *rhs.first));
      };

  // Sort and copy at most <kMaxNumOfKernels> kernels to <dst>.
  if (kernels_to_sort.size() > kMaxNumOfKernels) {
    absl::c_partial_sort(kernels_to_sort,
                         kernels_to_sort.begin() + kMaxNumOfKernels, comp);
  } else {
    absl::c_sort(kernels_to_sort, comp);
  }

  int copy_size =
      std::min(kMaxNumOfKernels, static_cast<int>(kernels_to_sort.size()));
  for (int i = 0; i < copy_size; i++) {
    KernelReport* report = dst->add_reports();
    *report = *kernels_to_sort[i].first;
    const KernelReportValue& kernel_value = *kernels_to_sort[i].second;
    // Set value using KernelReportValue.
    report->set_occurrences(kernel_value.occurrences);
    report->set_min_duration_ns(kernel_value.min_duration_ns);
    report->set_max_duration_ns(kernel_value.max_duration_ns);
    report->set_total_duration_ns(kernel_value.total_duration_ns);
  }
}

void InsertOrUpdateKernelReport(const KernelReport& kernel,
                                const KernelReportValue& value,
                                KernelReportMap* dst) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc mht_8(mht_8_v, 495, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.cc", "InsertOrUpdateKernelReport");

  KernelReportValue& element = (*dst)[kernel];
  if (element.occurrences == 0) {
    element = value;
  } else {
    element.total_duration_ns += value.total_duration_ns;
    element.min_duration_ns =
        std::min(element.min_duration_ns, value.min_duration_ns);
    element.max_duration_ns =
        std::max(element.max_duration_ns, value.max_duration_ns);
    element.occurrences += 1;
  }
}

void MergeKernelReports(const KernelReportMap& reports, KernelReportMap* dst) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc mht_9(mht_9_v, 512, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.cc", "MergeKernelReports");

  for (auto& kernel_value : reports) {
    InsertOrUpdateKernelReport(kernel_value.first, kernel_value.second, dst);
  }
}

KernelStatsByOpName GroupKernelReportsByOpName(
    const KernelStatsDb& kernel_stats_db) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSkernel_stats_utilsDTcc mht_10(mht_10_v, 522, "", "./tensorflow/core/profiler/utils/kernel_stats_utils.cc", "GroupKernelReportsByOpName");

  KernelStatsByOpName op_level_kernel_stats;
  for (const KernelReport& kernel_report : kernel_stats_db.reports()) {
    auto ret = op_level_kernel_stats.emplace(kernel_report.op_name(),
                                             OpLevelKernelStats());
    if (ret.second) {
      // Inserted. Add a new op in <op_level_kernel_stats>.
      OpLevelKernelStats& stats = ret.first->second;
      stats.is_op_tensor_core_eligible =
          kernel_report.is_op_tensor_core_eligible();
      stats.total_duration_ns += kernel_report.total_duration_ns();
      if (kernel_report.is_kernel_using_tensor_core()) {
        stats.tensor_core_duration_ns += kernel_report.total_duration_ns();
      }
    } else {
      // Not inserted. Aggregate kernel stats to op level.
      OpLevelKernelStats& stats = ret.first->second;
      // Verifies operations with the same name have the same TensorCore
      // eligibility.
      DCHECK_EQ(stats.is_op_tensor_core_eligible,
                kernel_report.is_op_tensor_core_eligible());
      stats.total_duration_ns += kernel_report.total_duration_ns();
      if (kernel_report.is_kernel_using_tensor_core()) {
        stats.tensor_core_duration_ns += kernel_report.total_duration_ns();
      }
    }
  }
  return op_level_kernel_stats;
}

}  // namespace profiler
}  // namespace tensorflow
