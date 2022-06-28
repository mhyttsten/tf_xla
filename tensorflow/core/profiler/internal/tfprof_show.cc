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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_showDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_showDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_showDTcc() {
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

#include "tensorflow/core/profiler/internal/tfprof_show.h"

#include <memory>
#include <set>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace tfprof {

const GraphNodeProto& TFShow::Show(const string& prefix, const Options& opts) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_showDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/profiler/internal/tfprof_show.cc", "TFShow::Show");

  if (opts.output_type == kOutput[0]) {
    Timeline timeline(opts.step, opts.output_options.at(kTimelineOpts[0]));
    return ShowInternal(opts, &timeline)->proto();
  } else {
    const ShowNode* ret = ShowInternal(opts, nullptr);
    if (opts.output_type == kOutput[1]) {
      absl::PrintF("%s", (prefix + ret->formatted_str));
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

bool TFShow::LookUpCheckPoint(const string& name,
                              std::unique_ptr<TFProfTensor>* tensor) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_showDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/profiler/internal/tfprof_show.cc", "TFShow::LookUpCheckPoint");

  if (name == kTFProfRoot || !ckpt_reader_ || !tensor) {
    return false;
  }
  std::unique_ptr<Tensor> out_tensor;
  TF_Status* status = TF_NewStatus();
  ckpt_reader_->GetTensor(name, &out_tensor, status);
  if (TF_GetCode(status) != TF_OK) {
    absl::FPrintF(stderr, "%s\n", TF_Message(status));
    TF_DeleteStatus(status);
    return false;
  }
  tensor->reset(new TFProfTensor(std::move(out_tensor)));
  TF_DeleteStatus(status);
  return true;
}

bool TFShow::ShouldShow(const ShowNode* node, const Options& opts,
                        int depth) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_showDTcc mht_2(mht_2_v, 250, "", "./tensorflow/core/profiler/internal/tfprof_show.cc", "TFShow::ShouldShow");

  // Always show kTFProfRoot.
  if (node->name() == kTFProfRoot) return true;

  if (node->proto().total_requested_bytes() < opts.min_bytes ||
      node->proto().total_peak_bytes() < opts.min_peak_bytes ||
      node->proto().total_residual_bytes() < opts.min_residual_bytes ||
      node->proto().total_output_bytes() < opts.min_output_bytes ||
      node->proto().total_exec_micros() < opts.min_micros ||
      node->proto().total_accelerator_exec_micros() <
          opts.min_accelerator_micros ||
      node->proto().total_cpu_exec_micros() < opts.min_cpu_micros ||
      node->proto().parameters() < opts.min_params ||
      node->proto().float_ops() < opts.min_float_ops ||
      node->proto().run_count() < opts.min_occurrence ||
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

bool TFShow::ShouldTrim(const ShowNode* node,
                        const std::vector<string>& regexes) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_showDTcc mht_3(mht_3_v, 293, "", "./tensorflow/core/profiler/internal/tfprof_show.cc", "TFShow::ShouldTrim");

  for (const string& regex : regexes) {
    if (RE2::FullMatch(node->name(), regex)) {
      return true;
    }
  }
  return false;
}

bool TFShow::ReAccount(ShowNode* node, const Options& opts) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_showDTcc mht_4(mht_4_v, 305, "", "./tensorflow/core/profiler/internal/tfprof_show.cc", "TFShow::ReAccount");

  node->ReInit(opts.step);
  if (opts.account_type_regexes.size() == 1 &&
      opts.account_type_regexes[0] == ".*") {
    return true;
  }
  for (const string& regex : opts.account_type_regexes) {
    for (const string& type : node->node->op_types()) {
      if (RE2::FullMatch(type, regex)) {
        return true;
      }
    }
  }
  return false;
}

string TFShow::FormatNodeMemory(ShowNode* node, int64_t bytes,
                                int64_t total_bytes) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_showDTcc mht_5(mht_5_v, 325, "", "./tensorflow/core/profiler/internal/tfprof_show.cc", "TFShow::FormatNodeMemory");

  string memory = FormatMemory(total_bytes);
  if (node->account) {
    memory = FormatMemory(bytes) + "/" + memory;
  } else {
    memory = "--/" + memory;
  }
  return memory;
}

string TFShow::FormatNode(ShowNode* node, const Options& opts) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_showDTcc mht_6(mht_6_v, 338, "", "./tensorflow/core/profiler/internal/tfprof_show.cc", "TFShow::FormatNode");

  std::vector<string> info;
  if (opts.select.find(kShown[2]) != opts.select.end()) {
    const string shape = FormatShapes(node->node->shape());
    if (!shape.empty()) {
      info.push_back(shape);
    }
    string params = FormatNumber(node->proto().total_parameters()) + " params";
    if (node->account) {
      params = FormatNumber(node->proto().parameters()) + "/" + params;
    } else {
      params = "--/" + params;
    }
    info.push_back(params);
  }
  if (opts.select.find(kShown[3]) != opts.select.end()) {
    string fops = FormatNumber(node->proto().total_float_ops()) + " flops";
    if (node->account) {
      fops = FormatNumber(node->proto().float_ops()) + "/" + fops;
    } else {
      fops = "--/" + fops;
    }
    info.push_back(fops);
  }
  if (opts.select.find(kShown[0]) != opts.select.end()) {
    info.push_back(FormatNodeMemory(node, node->proto().requested_bytes(),
                                    node->proto().total_requested_bytes()));
  }
  if (opts.select.find(kShown[11]) != opts.select.end()) {
    info.push_back(FormatNodeMemory(node, node->proto().peak_bytes(),
                                    node->proto().total_peak_bytes()));
  }
  if (opts.select.find(kShown[12]) != opts.select.end()) {
    info.push_back(FormatNodeMemory(node, node->proto().residual_bytes(),
                                    node->proto().total_residual_bytes()));
  }
  if (opts.select.find(kShown[13]) != opts.select.end()) {
    info.push_back(FormatNodeMemory(node, node->proto().output_bytes(),
                                    node->proto().total_output_bytes()));
  }
  if (opts.select.find(kShown[1]) != opts.select.end()) {
    info.push_back(FormatTotalExecTime(node, opts));
    info.push_back(FormatAcceleratorExecTime(node, opts));
    info.push_back(FormatCPUExecTime(node, opts));
  }
  if (opts.select.find(kShown[9]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    info.push_back(FormatAcceleratorExecTime(node, opts));
  }
  if (opts.select.find(kShown[10]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    info.push_back(FormatCPUExecTime(node, opts));
  }
  if (opts.select.find(kShown[5]) != opts.select.end()) {
    if (node->proto().devices_size() > 0) {
      info.push_back(absl::StrJoin(node->proto().devices(), "|"));
    }
  }
  if (opts.select.find(kShown[6]) != opts.select.end()) {
    const std::set<string>& op_types = node->node->op_types();
    info.push_back(absl::StrJoin(op_types, "|"));
  }
  if (opts.select.find(kShown[7]) != opts.select.end()) {
    string run = FormatNumber(node->proto().total_run_count());
    if (node->account) {
      run = FormatNumber(node->proto().run_count()) + "/" + run;
    } else {
      run = "--/" + run;
    }
    string definition = FormatNumber(node->proto().total_definition_count());
    if (node->account) {
      definition = "1/" + definition;
    } else {
      definition = "--/" + definition;
    }
    info.push_back(run + "|" + definition);
  }
  if (opts.select.find(kShown[8]) != opts.select.end()) {
    std::vector<string> shape_vec;
    for (const auto& s : node->node->input_shapes()) {
      if (s.second.empty()) {
        shape_vec.push_back(absl::StrFormat("%d:unknown", s.first));
      } else {
        shape_vec.push_back(
            absl::StrFormat("%d:%s", s.first, absl::StrJoin(s.second, "x")));
      }
    }
    info.push_back(absl::StrJoin(shape_vec, "|"));
  }

  return absl::StrFormat("%s (%s)", node->name(), absl::StrJoin(info, ", "));
}

string TFShow::FormatLegend(const Options& opts) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_showDTcc mht_7(mht_7_v, 434, "", "./tensorflow/core/profiler/internal/tfprof_show.cc", "TFShow::FormatLegend");

  std::vector<string> legends;
  if (opts.select.find(kShown[2]) != opts.select.end()) {
    legends.push_back("# parameters");
  }
  if (opts.select.find(kShown[3]) != opts.select.end()) {
    legends.push_back("# float_ops");
  }
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
  if (opts.select.find(kShown[5]) != opts.select.end()) {
    legends.push_back("assigned devices");
  }
  if (opts.select.find(kShown[6]) != opts.select.end()) {
    legends.push_back("op types");
  }
  if (opts.select.find(kShown[7]) != opts.select.end()) {
    legends.push_back("op count (run|defined)");
  }
  if (opts.select.find(kShown[8]) != opts.select.end()) {
    legends.push_back("input shapes");
  }
  return absl::StrFormat("node name | %s\n", absl::StrJoin(legends, " | "));
}

}  // namespace tfprof
}  // namespace tensorflow
