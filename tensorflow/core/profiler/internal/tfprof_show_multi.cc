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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTcc() {
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

/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/internal/tfprof_show_multi.h"

#include <memory>
#include <set>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/internal/tfprof_scope.h"

namespace tensorflow {
namespace tfprof {

const MultiGraphNodeProto& TFMultiShow::Show(const string& prefix,
                                             const Options& opts) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/profiler/internal/tfprof_show_multi.cc", "TFMultiShow::Show");

  if (opts.output_type == kOutput[0]) {
    Timeline timeline(opts.step, opts.output_options.at(kTimelineOpts[0]));
    return ShowInternal(opts, &timeline)->proto();
  } else {
    const ShowMultiNode* ret = ShowInternal(opts, nullptr);
    if (opts.output_type == kOutput[1]) {
      absl::PrintF("%s%s", prefix, ret->formatted_str);
      fflush(stdout);
    } else if (opts.output_type == kOutput[2]) {
      Status s = WriteStringToFile(Env::Default(),
                                   opts.output_options.at(kFileOpts[0]),
                                   prefix + ret->formatted_str);
      if (!s.ok()) {
        absl::FPrintF(stderr, "%s\n", s.ToString());
      }
    } else if (opts.output_type == kOutput[3] ||
               opts.output_type == kOutput[4]) {
    } else {
      absl::FPrintF(stderr, "Unknown output type: %s\n", opts.output_type);
    }
    return ret->proto();
  }
}

bool TFMultiShow::ShouldShow(const ShowMultiNode* node, const Options& opts,
                             int depth) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/profiler/internal/tfprof_show_multi.cc", "TFMultiShow::ShouldShow");

  // Always show kTFProfRoot.
  if (node->name() == kTFProfRoot) return true;

  // TODO(xpan): Think more carefully about node filtering in code view.
  // Unlike graph/scope view, which users want to see the exact leaf op.
  // In code view, users want to see the middle code traces they wrote.
  //
  // This is a subtle difference from scope/graph view. Usually mostly
  // want to see the middle code traces (i.e. their own codes.), instead
  // of the TensorFlow internal codes traces.
  if (node->proto().total_requested_bytes() < opts.min_bytes ||
      node->proto().total_peak_bytes() < opts.min_peak_bytes ||
      node->proto().total_residual_bytes() < opts.min_residual_bytes ||
      node->proto().total_output_bytes() < opts.min_output_bytes ||
      node->proto().total_exec_micros() < opts.min_micros ||
      node->proto().total_accelerator_exec_micros() <
          opts.min_accelerator_micros ||
      node->proto().total_cpu_exec_micros() < opts.min_cpu_micros ||
      node->proto().total_parameters() < opts.min_params ||
      node->proto().total_float_ops() < opts.min_float_ops ||
      depth > opts.max_depth || !ShouldShowIfExtra(node, opts, depth)) {
    return false;
  }

  bool show = false;
  if (opts.show_name_regexes.size() == 1 && opts.show_name_regexes[0] == ".*") {
    show = true;
  } else {
    for (const string& regex : opts.show_name_regexes) {
      if (RE2::FullMatch(node->name(), regex)) {
        show = true;
        break;
      }
    }
  }
  // Don't show if show_name_regexes don't cover it.
  if (!show) return false;
  // Don't show if hide_name_regexes cover it.
  for (const string& regex : opts.hide_name_regexes) {
    if (RE2::FullMatch(node->name(), regex)) return false;
  }
  return true;
}

bool TFMultiShow::ShouldTrim(const ShowMultiNode* node,
                             const std::vector<string>& regexes) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTcc mht_2(mht_2_v, 279, "", "./tensorflow/core/profiler/internal/tfprof_show_multi.cc", "TFMultiShow::ShouldTrim");

  for (const string& regex : regexes) {
    if (RE2::FullMatch(node->name(), regex)) {
      return true;
    }
  }
  return false;
}

bool TFMultiShow::ReAccount(ShowMultiNode* node, const Options& opts) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTcc mht_3(mht_3_v, 291, "", "./tensorflow/core/profiler/internal/tfprof_show_multi.cc", "TFMultiShow::ReAccount");

  return node->ReInit(opts.step, opts.account_type_regexes);
}

string TFMultiShow::FormatLegend(const Options& opts) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTcc mht_4(mht_4_v, 298, "", "./tensorflow/core/profiler/internal/tfprof_show_multi.cc", "TFMultiShow::FormatLegend");

  std::vector<string> legends;
  if (opts.select.find(kShown[0]) != opts.select.end()) {
    legends.push_back("requested bytes");
  }
  if (opts.select.find(kShown[11]) != opts.select.end()) {
    legends.push_back("peak bytes");
  }
  if (opts.select.find(kShown[12]) != opts.select.end()) {
    legends.push_back("residual bytes");
  }
  if (opts.select.find(kShown[13]) != opts.select.end()) {
    legends.push_back("output bytes");
  }
  if (opts.select.find(kShown[1]) != opts.select.end()) {
    legends.push_back("total execution time");
    legends.push_back("accelerator execution time");
    legends.push_back("cpu execution time");
  }
  if (opts.select.find(kShown[9]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    legends.push_back("accelerator execution time");
  }
  if (opts.select.find(kShown[10]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    legends.push_back("cpu execution time");
  }
  if (opts.select.find(kShown[2]) != opts.select.end()) {
    legends.push_back("# parameters");
  }
  if (opts.select.find(kShown[3]) != opts.select.end()) {
    legends.push_back("# float_ops");
  }
  if (opts.select.find(kShown[5]) != opts.select.end()) {
    legends.push_back("assigned devices");
  }
  if (opts.select.find(kShown[6]) != opts.select.end()) {
    legends.push_back("op types");
  }
  if (opts.select.find(kShown[7]) != opts.select.end()) {
    legends.push_back("op occurrence (run|defined)");
  }
  if (opts.select.find(kShown[8]) != opts.select.end()) {
    legends.push_back("input shapes");
  }
  return absl::StrFormat("node name | %s\n", absl::StrJoin(legends, " | "));
}

string TFMultiShow::FormatInputShapes(const MultiGraphNodeProto& proto) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTcc mht_5(mht_5_v, 349, "", "./tensorflow/core/profiler/internal/tfprof_show_multi.cc", "TFMultiShow::FormatInputShapes");

  // input_shape string -> (static defined count, run count, run_micros)
  std::map<string, std::tuple<int64_t, int64_t, int64_t>> input_shapes_attr;
  for (int i = 0; i < proto.graph_nodes_size(); ++i) {
    const GraphNodeProto& gnode = proto.graph_nodes(i);
    // Convert and sort by input_idx.
    std::map<int, std::vector<int64_t>> input_shapes;
    for (const auto& inp : gnode.input_shapes()) {
      input_shapes[inp.first] = ShapeProtoToVec(inp.second);
    }

    std::vector<string> input_vec;
    for (const auto& s : input_shapes) {
      if (s.second.empty()) {
        input_vec.push_back(absl::StrFormat("%d:unknown", s.first));
      } else {
        input_vec.push_back(
            absl::StrFormat("%d:%s", s.first, absl::StrJoin(s.second, "x")));
      }
    }
    string shape_type_str =
        absl::StrFormat("input_type: %s", absl::StrJoin(input_vec, ",\t"));
    auto t = input_shapes_attr.find(shape_type_str);
    if (t == input_shapes_attr.end()) {
      input_shapes_attr.insert(
          std::make_pair(shape_type_str, std::make_tuple(0, 0, 0)));
      t = input_shapes_attr.find(shape_type_str);
    }
    input_shapes_attr[shape_type_str] = std::make_tuple(
        std::get<0>(t->second) + 1, std::get<1>(t->second) + gnode.run_count(),
        std::get<2>(t->second) + gnode.exec_micros());
  }
  if (input_shapes_attr.empty()) {
    return "";
  }

  std::vector<std::pair<string, std::tuple<int64_t, int64_t, int64_t>>>
      shape_count_vec(input_shapes_attr.begin(), input_shapes_attr.end());
  std::stable_sort(
      shape_count_vec.begin(), shape_count_vec.end(),
      [](const std::pair<const string, std::tuple<int64, int64, int64>>& a,
         const std::pair<const string, std::tuple<int64, int64, int64>>& b) {
        return std::get<1>(a.second) > std::get<1>(b.second);
      });

  std::vector<string> input_types;
  input_types.reserve(shape_count_vec.size());
  for (const auto& s : shape_count_vec) {
    std::tuple<int64_t, int64_t, int64_t> t = s.second;
    input_types.push_back(absl::StrFormat(
        "%s\t(run*%d|defined*%d)\texec_time: %s", s.first, std::get<1>(t),
        std::get<0>(t), FormatTime(std::get<2>(t))));
  }
  return absl::StrJoin(input_types, "\n");
}

std::vector<string> TFMultiShow::FormatTimes(const ShowMultiNode* node,
                                             const Options& opts) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_show_multiDTcc mht_6(mht_6_v, 409, "", "./tensorflow/core/profiler/internal/tfprof_show_multi.cc", "TFMultiShow::FormatTimes");

  std::vector<string> attrs;
  if (opts.select.find(kShown[1]) != opts.select.end()) {
    attrs.push_back(FormatTotalExecTime(node, opts));
    attrs.push_back(FormatAcceleratorExecTime(node, opts));
    attrs.push_back(FormatCPUExecTime(node, opts));
  }
  if (opts.select.find(kShown[9]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    attrs.push_back(FormatAcceleratorExecTime(node, opts));
  }
  if (opts.select.find(kShown[10]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    attrs.push_back(FormatCPUExecTime(node, opts));
  }
  return attrs;
}

}  // namespace tfprof
}  // namespace tensorflow
