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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc() {
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

#include "tensorflow/core/profiler/internal/tfprof_utils.h"

#include <stdio.h>

#include <algorithm>
#include <memory>
#include <set>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace tfprof {
string FormatNumber(int64_t n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "FormatNumber");

  if (n < 1000) {
    return absl::StrFormat("%d", n);
  } else if (n < 1000000) {
    return absl::StrFormat("%.2fk", n / 1000.0);
  } else if (n < 1000000000) {
    return absl::StrFormat("%.2fm", n / 1000000.0);
  } else {
    return absl::StrFormat("%.2fb", n / 1000000000.0);
  }
}

string FormatTime(int64_t micros) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "FormatTime");

  if (micros < 1000) {
    return absl::StrFormat("%dus", micros);
  } else if (micros < 1000000) {
    return absl::StrFormat("%.2fms", micros / 1000.0);
  } else {
    return absl::StrFormat("%.2fsec", micros / 1000000.0);
  }
}

string FormatMemory(int64_t bytes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "FormatMemory");

  if (bytes < 1000) {
    return absl::StrFormat("%dB", bytes);
  } else if (bytes < 1000000) {
    return absl::StrFormat("%.2fKB", bytes / 1000.0);
  } else {
    return absl::StrFormat("%.2fMB", bytes / 1000000.0);
  }
}

string FormatShapes(const std::vector<int64_t>& shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_3(mht_3_v, 244, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "FormatShapes");

  return absl::StrJoin(shape, "x");
}

string StringReplace(const string& str, const string& oldsub,
                     const string& newsub) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("str: \"" + str + "\"");
   mht_4_v.push_back("oldsub: \"" + oldsub + "\"");
   mht_4_v.push_back("newsub: \"" + newsub + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_4(mht_4_v, 255, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "StringReplace");

  string out = str;
  RE2::GlobalReplace(&out, oldsub, newsub);
  return out;
}

namespace {
string StripQuote(const string& s) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_5(mht_5_v, 266, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "StripQuote");

  int start = s.find_first_not_of("\"\'");
  int end = s.find_last_not_of("\"\'");
  if (start == s.npos || end == s.npos) return "";

  return s.substr(start, end - start + 1);
}

tensorflow::Status ReturnError(const std::vector<string>& pieces, int idx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_6(mht_6_v, 277, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "ReturnError");

  string val;
  if (pieces.size() > idx + 1) {
    val = pieces[idx + 1];
  }
  return tensorflow::Status(
      tensorflow::error::INVALID_ARGUMENT,
      absl::StrCat("Invalid option '", pieces[idx], "' value: '", val, "'"));
}

bool CaseEqual(StringPiece s1, StringPiece s2) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_7(mht_7_v, 290, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "CaseEqual");

  if (s1.size() != s2.size()) return false;
  return absl::AsciiStrToLower(s1) == absl::AsciiStrToLower(s2);
}

bool StringToBool(StringPiece str, bool* value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_8(mht_8_v, 298, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "StringToBool");

  CHECK(value != nullptr) << "NULL output boolean given.";
  if (CaseEqual(str, "true") || CaseEqual(str, "t") || CaseEqual(str, "yes") ||
      CaseEqual(str, "y") || CaseEqual(str, "1")) {
    *value = true;
    return true;
  }
  if (CaseEqual(str, "false") || CaseEqual(str, "f") || CaseEqual(str, "no") ||
      CaseEqual(str, "n") || CaseEqual(str, "0")) {
    *value = false;
    return true;
  }
  return false;
}
}  // namespace

tensorflow::Status ParseCmdLine(const string& line, string* cmd,
                                tensorflow::tfprof::Options* opts) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("line: \"" + line + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_9(mht_9_v, 319, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "ParseCmdLine");

  std::vector<string> pieces = absl::StrSplit(line, ' ', absl::SkipEmpty());

  std::vector<string> cmds_str(kCmds, kCmds + sizeof(kCmds) / sizeof(*kCmds));
  if (std::find(cmds_str.begin(), cmds_str.end(), pieces[0]) ==
      cmds_str.end()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "First string must be a valid command.");
  }
  *cmd = pieces[0];

  for (int i = 1; i < pieces.size(); ++i) {
    if (pieces[i] == string(tensorflow::tfprof::kOptions[0])) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->max_depth)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[1]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[2]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_peak_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[3]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_residual_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[4]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_output_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[5]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_micros)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[6]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_accelerator_micros)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[7]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_cpu_micros)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[8]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_params)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[9]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_float_ops)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[10]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_occurrence)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[11]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->step)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[12]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      std::set<string> order_by_set(
          kOrderBy, kOrderBy + sizeof(kOrderBy) / sizeof(*kOrderBy));
      auto order_by = order_by_set.find(pieces[i + 1]);
      if (order_by == order_by_set.end()) {
        return ReturnError(pieces, i);
      }
      opts->order_by = *order_by;
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[13]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->account_type_regexes =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[14]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->start_name_regexes =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[15]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->trim_name_regexes =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[16]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->show_name_regexes =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[17]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->hide_name_regexes =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[18]) {
      if ((pieces.size() > i + 1 && absl::StartsWith(pieces[i + 1], "-")) ||
          pieces.size() == i + 1) {
        opts->account_displayed_op_only = true;
      } else if (!StringToBool(pieces[i + 1],
                               &opts->account_displayed_op_only)) {
        return ReturnError(pieces, i);
      } else {
        ++i;
      }
    } else if (pieces[i] == tensorflow::tfprof::kOptions[19]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      std::set<string> shown_set(kShown,
                                 kShown + sizeof(kShown) / sizeof(*kShown));
      std::vector<string> requested_vector =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      std::set<string> requested_set(requested_vector.begin(),
                                     requested_vector.end());
      for (const string& requested : requested_set) {
        if (shown_set.find(requested) == shown_set.end()) {
          return ReturnError(pieces, i);
        }
      }
      opts->select = requested_set;
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[20]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }

      tensorflow::Status s =
          ParseOutput(pieces[i + 1], &opts->output_type, &opts->output_options);
      if (!s.ok()) return s;
      ++i;
    } else {
      return ReturnError(pieces, i);
    }
  }
  return tensorflow::Status::OK();
}

void PrintHelp() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_10(mht_10_v, 496, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "PrintHelp");

  absl::PrintF(
      "See https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/"
      "README.md for profiler tutorial.\n");
  absl::PrintF(
      "See https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/"
      "g3doc/command_line.md for command line tool tutorial.\n");
  absl::PrintF(
      "profiler --profile_path=<ProfileProto binary file> # required\n"
      "\nOr:\n\n"
      "profiler --graph_path=<GraphDef proto file>  "
      "# Contains model graph info (no needed for eager execution)\n"
      "         --run_meta_path=<RunMetadata proto file>  "
      "# Contains runtime info. Optional.\n"
      "         --run_log_path=<OpLogProto proto file>  "
      "# Contains extra source code, flops, custom type info. Optional\n\n");
  absl::PrintF(
      "\nTo skip interactive mode, append one of the following commands:\n"
      "  scope: Organize profiles based on name scopes.\n"
      "  graph: Organize profiles based on graph node input/output.\n"
      "  op: Organize profiles based on operation type.\n"
      "  code: Organize profiles based on python codes (need op_log_path).\n"
      "  advise: Auto-profile and advise. (experimental)\n"
      "  set: Set options that will be default for follow up commands.\n"
      "  help: Show helps.\n");
  fflush(stdout);
}

static const char* const kTotalMicrosHelp =
    "total execution time: Sum of accelerator execution time and cpu execution "
    "time.";
static const char* const kAccMicrosHelp =
    "accelerator execution time: Time spent executing on the accelerator. "
    "This is normally measured by the actual hardware library.";
static const char* const kCPUHelp =
    "cpu execution time: The time from the start to the end of the operation. "
    "It's the sum of actual cpu run time plus the time that it spends waiting "
    "if part of computation is launched asynchronously.";
static const char* const kBytes =
    "requested bytes: The memory requested by the operation, accumulatively.";
static const char* const kPeakBytes =
    "peak bytes: The peak amount of memory that the operation is holding at "
    "some point.";
static const char* const kResidualBytes =
    "residual bytes: The memory not de-allocated after the operation finishes.";
static const char* const kOutputBytes =
    "output bytes: The memory that is output from the operation (not "
    "necessarily allocated by the operation)";
static const char* const kOccurrence =
    "occurrence: The number of times it occurs";
static const char* const kInputShapes =
    "input shape: The shape of input tensors";
static const char* const kDevice = "device: which device is placed on.";
static const char* const kFloatOps =
    "flops: Number of float operations. Note: Please read the implementation "
    "for the math behind it.";
static const char* const kParams =
    "param: Number of parameters (in the Variable).";
static const char* const kTensorValue = "tensor_value: Not supported now.";
static const char* const kOpTypes =
    "op_types: The attributes of the operation, includes the Kernel name "
    "device placed on and user-defined strings.";

static const char* const kScope =
    "scope: The nodes in the model graph are organized by their names, which "
    "is hierarchical like filesystem.";
static const char* const kCode =
    "code: When python trace is available, the nodes are python lines and "
    "their are organized by the python call stack.";
static const char* const kOp =
    "op: The nodes are operation kernel type, such as MatMul, Conv2D. Graph "
    "nodes belonging to the same type are aggregated together.";
static const char* const kAdvise =
    "advise: Automatically profile and discover issues. (Experimental)";
static const char* const kSet =
    "set: Set a value for an option for future use.";
static const char* const kHelp = "help: Print helping messages.";

string QueryDoc(const string& cmd, const Options& opts) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("cmd: \"" + cmd + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_utilsDTcc mht_11(mht_11_v, 578, "", "./tensorflow/core/profiler/internal/tfprof_utils.cc", "QueryDoc");

  string cmd_help = "";
  if (cmd == kCmds[0]) {
    cmd_help = kScope;
  } else if (cmd == kCmds[1]) {
    cmd_help = kScope;
  } else if (cmd == kCmds[2]) {
    cmd_help = kCode;
  } else if (cmd == kCmds[3]) {
    cmd_help = kOp;
  } else if (cmd == kCmds[4]) {
    cmd_help = kAdvise;
  } else if (cmd == kCmds[5]) {
    cmd_help = kSet;
  } else if (cmd == kCmds[6]) {
    cmd_help = kHelp;
  } else {
    cmd_help = "Unknown command: " + cmd;
  }

  std::vector<string> helps;
  for (const string& s : opts.select) {
    if (s == kShown[0]) {
      helps.push_back(kBytes);
    } else if (s == kShown[1]) {
      helps.push_back(
          absl::StrCat(kTotalMicrosHelp, "\n", kCPUHelp, "\n", kAccMicrosHelp));
    } else if (s == kShown[2]) {
      helps.push_back(kParams);
    } else if (s == kShown[3]) {
      helps.push_back(kFloatOps);
    } else if (s == kShown[4]) {
      helps.push_back(kTensorValue);
    } else if (s == kShown[5]) {
      helps.push_back(kDevice);
    } else if (s == kShown[6]) {
      helps.push_back(kOpTypes);
    } else if (s == kShown[7]) {
      helps.push_back(kOccurrence);
    } else if (s == kShown[8]) {
      helps.push_back(kInputShapes);
    } else if (s == kShown[9]) {
      helps.push_back(kAccMicrosHelp);
    } else if (s == kShown[10]) {
      helps.push_back(kCPUHelp);
    } else if (s == kShown[11]) {
      helps.push_back(kPeakBytes);
    } else if (s == kShown[12]) {
      helps.push_back(kResidualBytes);
    } else if (s == kShown[13]) {
      helps.push_back(kOutputBytes);
    } else {
      helps.push_back("Unknown select: " + s);
    }
  }
  return absl::StrCat("\nDoc:\n", cmd_help, "\n", absl::StrJoin(helps, "\n"),
                      "\n\n");
}

}  // namespace tfprof
}  // namespace tensorflow
