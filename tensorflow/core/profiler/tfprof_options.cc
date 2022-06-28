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
class MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTcc() {
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

#include "tensorflow/core/profiler/tfprof_options.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/tfprof_options.pb.h"

namespace tensorflow {
namespace tfprof {
namespace {
string KeyValueToStr(const std::map<string, string>& kv_map) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/profiler/tfprof_options.cc", "KeyValueToStr");

  std::vector<string> kv_vec;
  kv_vec.reserve(kv_map.size());
  for (const auto& pair : kv_map) {
    kv_vec.push_back(absl::StrCat(pair.first, "=", pair.second));
  }
  return absl::StrJoin(kv_vec, ",");
}
}  // namespace

tensorflow::Status ParseOutput(const string& output_opt, string* output_type,
                               std::map<string, string>* output_options) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("output_opt: \"" + output_opt + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/profiler/tfprof_options.cc", "ParseOutput");

  // The default is to use stdout.
  if (output_opt.empty()) {
    *output_type = kOutput[1];
    return tensorflow::Status::OK();
  }

  std::set<string> output_types(kOutput,
                                kOutput + sizeof(kOutput) / sizeof(*kOutput));
  auto opt_split = output_opt.find(':');
  std::vector<string> kv_split;
  if (opt_split == output_opt.npos) {
    if (output_types.find(output_opt) == output_types.end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          absl::StrFormat("E.g. Unknown output type: %s, Valid types: %s\n",
                          output_opt, absl::StrJoin(output_types, ",")));
    }
    *output_type = output_opt;
  } else {
    *output_type = output_opt.substr(0, opt_split);
    if (output_types.find(*output_type) == output_types.end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          absl::StrFormat("E.g. Unknown output type: %s, Valid types: %s\n",
                          *output_type, absl::StrJoin(output_types, ",")));
    }
    kv_split = absl::StrSplit(output_opt.substr(opt_split + 1), ",",
                              absl::SkipEmpty());
  }

  std::set<string> valid_options;
  std::set<string> required_options;
  if (*output_type == kOutput[0]) {
    valid_options.insert(
        kTimelineOpts,
        kTimelineOpts + sizeof(kTimelineOpts) / sizeof(*kTimelineOpts));
    required_options.insert(
        kTimelineRequiredOpts,
        kTimelineRequiredOpts +
            sizeof(kTimelineRequiredOpts) / sizeof(*kTimelineRequiredOpts));
  } else if (*output_type == kOutput[2]) {
    valid_options.insert(kFileOpts,
                         kFileOpts + sizeof(kFileOpts) / sizeof(*kFileOpts));
    required_options.insert(kFileRequiredOpts,
                            kFileRequiredOpts + sizeof(kFileRequiredOpts) /
                                                    sizeof(*kFileRequiredOpts));
  } else if (*output_type == kOutput[3]) {
    valid_options.insert(kPprofOpts,
                         kPprofOpts + sizeof(kPprofOpts) / sizeof(*kPprofOpts));
    required_options.insert(
        kPprofRequiredOpts,
        kPprofRequiredOpts +
            sizeof(kPprofRequiredOpts) / sizeof(*kPprofRequiredOpts));
  }

  for (const string& kv_str : kv_split) {
    const std::vector<string> kv =
        absl::StrSplit(kv_str, "=", absl::SkipEmpty());
    if (kv.size() < 2) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          "Visualize format: -output timeline:key=value,key=value,...");
    }
    if (valid_options.find(kv[0]) == valid_options.end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          absl::StrFormat("Unrecognized options %s for output_type: %s\n",
                          kv[0], *output_type));
    }
    const std::vector<string> kv_without_key(kv.begin() + 1, kv.end());
    (*output_options)[kv[0]] = absl::StrJoin(kv_without_key, "=");
  }

  for (const string& opt : required_options) {
    if (output_options->find(opt) == output_options->end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          absl::StrFormat("Missing required output_options for %s\n"
                          "E.g. -output %s:%s=...\n",
                          *output_type, *output_type, opt));
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status Options::FromProtoStr(const string& opts_proto_str,
                                         Options* opts) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("opts_proto_str: \"" + opts_proto_str + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTcc mht_2(mht_2_v, 301, "", "./tensorflow/core/profiler/tfprof_options.cc", "Options::FromProtoStr");

  OptionsProto opts_pb;
  if (!opts_pb.ParseFromString(opts_proto_str)) {
    return tensorflow::Status(
        tensorflow::error::INTERNAL,
        absl::StrCat("Failed to parse option string from Python API: ",
                     opts_proto_str));
  }

  string output_type;
  std::map<string, string> output_options;
  tensorflow::Status s =
      ParseOutput(opts_pb.output(), &output_type, &output_options);
  if (!s.ok()) return s;

  if (!opts_pb.dump_to_file().empty()) {
    absl::FPrintF(stderr,
                  "-dump_to_file option is deprecated. "
                  "Please use -output file:outfile=<filename>\n");
    absl::FPrintF(stderr,
                  "-output %s is overwritten with -output file:outfile=%s\n",
                  opts_pb.output(), opts_pb.dump_to_file());
    output_type = kOutput[2];
    output_options.clear();
    output_options[kFileOpts[0]] = opts_pb.dump_to_file();
  }

  *opts = Options(
      opts_pb.max_depth(), opts_pb.min_bytes(), opts_pb.min_peak_bytes(),
      opts_pb.min_residual_bytes(), opts_pb.min_output_bytes(),
      opts_pb.min_micros(), opts_pb.min_accelerator_micros(),
      opts_pb.min_cpu_micros(), opts_pb.min_params(), opts_pb.min_float_ops(),
      opts_pb.min_occurrence(), opts_pb.step(), opts_pb.order_by(),
      std::vector<string>(opts_pb.account_type_regexes().begin(),
                          opts_pb.account_type_regexes().end()),
      std::vector<string>(opts_pb.start_name_regexes().begin(),
                          opts_pb.start_name_regexes().end()),
      std::vector<string>(opts_pb.trim_name_regexes().begin(),
                          opts_pb.trim_name_regexes().end()),
      std::vector<string>(opts_pb.show_name_regexes().begin(),
                          opts_pb.show_name_regexes().end()),
      std::vector<string>(opts_pb.hide_name_regexes().begin(),
                          opts_pb.hide_name_regexes().end()),
      opts_pb.account_displayed_op_only(),
      std::vector<string>(opts_pb.select().begin(), opts_pb.select().end()),
      output_type, output_options);
  return tensorflow::Status::OK();
}

std::string Options::ToString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPStfprof_optionsDTcc mht_3(mht_3_v, 353, "", "./tensorflow/core/profiler/tfprof_options.cc", "Options::ToString");

  // clang-format off
  const std::string s = absl::StrFormat(
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s:%s\n",
      kOptions[0],  max_depth,
      kOptions[1],  min_bytes,
      kOptions[2],  min_peak_bytes,
      kOptions[3],  min_residual_bytes,
      kOptions[4],  min_output_bytes,
      kOptions[5],  min_micros,
      kOptions[6],  min_accelerator_micros,
      kOptions[7],  min_cpu_micros,
      kOptions[8],  min_params,
      kOptions[9],  min_float_ops,
      kOptions[10], min_occurrence,
      kOptions[11], step,
      kOptions[12], order_by,
      kOptions[13], absl::StrJoin(account_type_regexes, ","),
      kOptions[14], absl::StrJoin(start_name_regexes, ","),
      kOptions[15], absl::StrJoin(trim_name_regexes, ","),
      kOptions[16], absl::StrJoin(show_name_regexes, ","),
      kOptions[17], absl::StrJoin(hide_name_regexes, ","),
      kOptions[18], (account_displayed_op_only ? "true" : "false"),
      kOptions[19], absl::StrJoin(select, ","),
      kOptions[20], output_type, KeyValueToStr(output_options));
  // clang-format on
  return s;
}

}  // namespace tfprof
}  // namespace tensorflow
