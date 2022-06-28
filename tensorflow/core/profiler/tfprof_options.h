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

#ifndef TENSORFLOW_CORE_PROFILER_TFPROF_OPTIONS_H_
#define TENSORFLOW_CORE_PROFILER_TFPROF_OPTIONS_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTh() {
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


#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tfprof {
static const char* const kOptions[] = {
    "-max_depth",
    "-min_bytes",
    "-min_peak_bytes",
    "-min_residual_bytes",
    "-min_output_bytes",
    "-min_micros",
    "-min_accelerator_micros",
    "-min_cpu_micros",
    "-min_params",
    "-min_float_ops",
    "-min_occurrence",
    "-step",
    "-order_by",
    "-account_type_regexes",
    "-start_name_regexes",
    "-trim_name_regexes",
    "-show_name_regexes",
    "-hide_name_regexes",
    "-account_displayed_op_only",
    "-select",
    "-output",
};

static const char* const kOrderBy[] = {
    "name",         "bytes",     "peak_bytes",         "residual_bytes",
    "output_bytes", "micros",    "accelerator_micros", "cpu_micros",
    "params",       "float_ops", "occurrence",
};

// Append Only.
// TODO(xpan): As we are adding more fields to be selected, we
// need to have a way to tell users what fields are available in which view.
static const char* const kShown[] = {"bytes",          "micros",
                                     "params",         "float_ops",
                                     "tensor_value",   "device",
                                     "op_types",       "occurrence",
                                     "input_shapes",   "accelerator_micros",
                                     "cpu_micros",     "peak_bytes",
                                     "residual_bytes", "output_bytes"};

static const char* const kCmds[] = {
    "scope", "graph", "code", "op", "advise", "set", "help",
};

static const char* const kOutput[] = {"timeline", "stdout", "file", "pprof",
                                      "none"};

static const char* const kTimelineOpts[] = {
    "outfile",
};

static const char* const kTimelineRequiredOpts[] = {"outfile"};

static const char* const kFileOpts[] = {
    "outfile",
};

static const char* const kFileRequiredOpts[] = {
    "outfile",
};

static const char* const kPprofOpts[] = {
    "outfile",
};

static const char* const kPprofRequiredOpts[] = {
    "outfile",
};

struct Options {
 public:
  static tensorflow::Status FromProtoStr(const string& opts_proto_str,
                                         Options* opts);

  virtual ~Options() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTh mht_0(mht_0_v, 273, "", "./tensorflow/core/profiler/tfprof_options.h", "~Options");
}
  Options()
      : Options(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "", {}, {}, {}, {}, {},
                false, {}, "", {}) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTh mht_1(mht_1_v, 279, "", "./tensorflow/core/profiler/tfprof_options.h", "Options");
}

  Options(int max_depth, int64_t min_bytes, int64_t min_peak_bytes,
          int64_t min_residual_bytes, int64_t min_output_bytes,
          int64_t min_micros, int64_t min_accelerator_micros,
          int64_t min_cpu_micros, int64_t min_params, int64_t min_float_ops,
          int64_t min_occurrence, int64_t step, const string& order_by,
          const std::vector<string>& account_type_regexes,
          const std::vector<string>& start_name_regexes,
          const std::vector<string>& trim_name_regexes,
          const std::vector<string>& show_name_regexes,
          const std::vector<string>& hide_name_regexes,
          bool account_displayed_op_only, const std::vector<string>& select,
          const string& output_type,
          const std::map<string, string>& output_options)
      : max_depth(max_depth),
        min_bytes(min_bytes),
        min_peak_bytes(min_peak_bytes),
        min_residual_bytes(min_residual_bytes),
        min_output_bytes(min_output_bytes),
        min_micros(min_micros),
        min_accelerator_micros(min_accelerator_micros),
        min_cpu_micros(min_cpu_micros),
        min_params(min_params),
        min_float_ops(min_float_ops),
        min_occurrence(min_occurrence),
        step(step),
        order_by(order_by),
        account_type_regexes(account_type_regexes),
        start_name_regexes(start_name_regexes),
        trim_name_regexes(trim_name_regexes),
        show_name_regexes(show_name_regexes),
        hide_name_regexes(hide_name_regexes),
        account_displayed_op_only(account_displayed_op_only),
        select(select.begin(), select.end()),
        output_type(output_type),
        output_options(output_options) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("order_by: \"" + order_by + "\"");
   mht_2_v.push_back("output_type: \"" + output_type + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTh mht_2(mht_2_v, 320, "", "./tensorflow/core/profiler/tfprof_options.h", "Options");
}

  string ToString() const;

  int max_depth;
  int64_t min_bytes;
  int64_t min_peak_bytes;
  int64_t min_residual_bytes;
  int64_t min_output_bytes;
  int64_t min_micros;
  int64_t min_accelerator_micros;
  int64_t min_cpu_micros;
  int64_t min_params;
  int64_t min_float_ops;
  int64_t min_occurrence;
  int64_t step;
  string order_by;

  std::vector<string> account_type_regexes;
  std::vector<string> start_name_regexes;
  std::vector<string> trim_name_regexes;
  std::vector<string> show_name_regexes;
  std::vector<string> hide_name_regexes;
  bool account_displayed_op_only;

  std::set<string> select;

  string output_type;
  std::map<string, string> output_options;
};

// Parse the -output option.
// 'output_opt': User input string with format: output_type:key=value,key=value.
// 'output_type' and 'output_options' are extracted from 'output_opt'.
tensorflow::Status ParseOutput(const string& output_opt, string* output_type,
                               std::map<string, string>* output_options);

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_TFPROF_OPTIONS_H_
