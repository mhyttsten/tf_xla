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
class MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc() {
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

#include "tensorflow/lite/tools/command_line_flags.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace {

template <typename T>
std::string ToString(T val) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/tools/command_line_flags.cc", "ToString");

  std::ostringstream stream;
  stream << val;
  return stream.str();
}

template <>
std::string ToString(bool val) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/tools/command_line_flags.cc", "ToString");

  return val ? "true" : "false";
}

template <>
std::string ToString(const std::string& val) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("val: \"" + val + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc mht_2(mht_2_v, 221, "", "./tensorflow/lite/tools/command_line_flags.cc", "ToString");

  return val;
}

bool ParseFlag(const std::string& arg, int argv_position,
               const std::string& flag, bool positional,
               const std::function<bool(const std::string&, int argv_position)>&
                   parse_func,
               bool* value_parsing_ok) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("arg: \"" + arg + "\"");
   mht_3_v.push_back("flag: \"" + flag + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc mht_3(mht_3_v, 234, "", "./tensorflow/lite/tools/command_line_flags.cc", "ParseFlag");

  if (positional) {
    *value_parsing_ok = parse_func(arg, argv_position);
    return true;
  }
  *value_parsing_ok = true;
  std::string flag_prefix = "--" + flag + "=";
  if (!absl::StartsWith(arg, flag_prefix)) {
    return false;
  }
  bool has_value = arg.size() >= flag_prefix.size();
  *value_parsing_ok = has_value;
  if (has_value) {
    *value_parsing_ok =
        parse_func(arg.substr(flag_prefix.size()), argv_position);
  }
  return true;
}

template <typename T>
bool ParseFlag(const std::string& flag_value, int argv_position,
               const std::function<void(const T&, int)>& hook) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("flag_value: \"" + flag_value + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc mht_4(mht_4_v, 259, "", "./tensorflow/lite/tools/command_line_flags.cc", "ParseFlag");

  std::istringstream stream(flag_value);
  T read_value;
  stream >> read_value;
  if (!stream.eof() && !stream.good()) {
    return false;
  }
  hook(read_value, argv_position);
  return true;
}

template <>
bool ParseFlag(const std::string& flag_value, int argv_position,
               const std::function<void(const bool&, int)>& hook) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("flag_value: \"" + flag_value + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc mht_5(mht_5_v, 276, "", "./tensorflow/lite/tools/command_line_flags.cc", "ParseFlag");

  if (flag_value != "true" && flag_value != "false" && flag_value != "0" &&
      flag_value != "1") {
    return false;
  }

  hook(flag_value == "true" || flag_value == "1", argv_position);
  return true;
}

template <typename T>
bool ParseFlag(const std::string& flag_value, int argv_position,
               const std::function<void(const std::string&, int)>& hook) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("flag_value: \"" + flag_value + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc mht_6(mht_6_v, 292, "", "./tensorflow/lite/tools/command_line_flags.cc", "ParseFlag");

  hook(flag_value, argv_position);
  return true;
}
}  // namespace

#define CONSTRUCTOR_IMPLEMENTATION(flag_T, default_value_T, flag_enum_val)     \
  Flag::Flag(const char* name,                                                 \
             const std::function<void(const flag_T& /*flag_val*/,              \
                                      int /*argv_position*/)>& hook,           \
             default_value_T default_value, const std::string& usage_text,     \
             FlagType flag_type)                                               \
      : name_(name),                                                           \
        type_(flag_enum_val),                                                  \
        value_hook_([hook](const std::string& flag_value, int argv_position) { \
          return ParseFlag<flag_T>(flag_value, argv_position, hook);           \
        }),                                                                    \
        default_for_display_(ToString<default_value_T>(default_value)),        \
        usage_text_(usage_text),                                               \
        flag_type_(flag_type) {}

CONSTRUCTOR_IMPLEMENTATION(int32_t, int32_t, TYPE_INT32)
CONSTRUCTOR_IMPLEMENTATION(int64_t, int64_t, TYPE_INT64)
CONSTRUCTOR_IMPLEMENTATION(float, float, TYPE_FLOAT)
CONSTRUCTOR_IMPLEMENTATION(bool, bool, TYPE_BOOL)
CONSTRUCTOR_IMPLEMENTATION(std::string, const std::string&, TYPE_STRING)

#undef CONSTRUCTOR_IMPLEMENTATION

bool Flag::Parse(const std::string& arg, int argv_position,
                 bool* value_parsing_ok) const {
  return ParseFlag(
      arg, argv_position, name_, flag_type_ == kPositional,
      [&](const std::string& read_value, int argv_position) {
        return value_hook_(read_value, argv_position);
      },
      value_parsing_ok);
}

std::string Flag::GetTypeName() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc mht_7(mht_7_v, 334, "", "./tensorflow/lite/tools/command_line_flags.cc", "Flag::GetTypeName");

  switch (type_) {
    case TYPE_INT32:
      return "int32";
    case TYPE_INT64:
      return "int64";
    case TYPE_FLOAT:
      return "float";
    case TYPE_BOOL:
      return "bool";
    case TYPE_STRING:
      return "string";
  }

  return "unknown";
}

/*static*/ bool Flags::Parse(int* argc, const char** argv,
                             const std::vector<Flag>& flag_list) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc mht_8(mht_8_v, 355, "", "./tensorflow/lite/tools/command_line_flags.cc", "Flags::Parse");

  bool result = true;
  std::vector<bool> unknown_argvs(*argc, true);
  // Record the list of flags that have been processed. key is the flag's name
  // and the value is the corresponding argv index if there's one, or -1 when
  // the argv list doesn't contain this flag.
  std::unordered_map<std::string, int> processed_flags;

  // Stores indexes of flag_list in a sorted order.
  std::vector<int> sorted_idx(flag_list.size());
  std::iota(std::begin(sorted_idx), std::end(sorted_idx), 0);
  std::sort(sorted_idx.begin(), sorted_idx.end(), [&flag_list](int a, int b) {
    return flag_list[a].GetFlagType() < flag_list[b].GetFlagType();
  });
  int positional_count = 0;

  for (int idx = 0; idx < sorted_idx.size(); ++idx) {
    const Flag& flag = flag_list[sorted_idx[idx]];

    const auto it = processed_flags.find(flag.name_);
    if (it != processed_flags.end()) {
#ifndef NDEBUG
      // Only log this in debug builds.
      TFLITE_LOG(WARN) << "Duplicate flags: " << flag.name_;
#endif
      if (it->second != -1) {
        bool value_parsing_ok;
        flag.Parse(argv[it->second], it->second, &value_parsing_ok);
        if (!value_parsing_ok) {
          TFLITE_LOG(ERROR) << "Failed to parse flag '" << flag.name_
                            << "' against argv '" << argv[it->second] << "'";
          result = false;
        }
        continue;
      } else if (flag.flag_type_ == Flag::kRequired) {
        TFLITE_LOG(ERROR) << "Required flag not provided: " << flag.name_;
        // If the required flag isn't found, we immediately stop the whole flag
        // parsing.
        result = false;
        break;
      }
    }

    // Parses positional flags.
    if (flag.flag_type_ == Flag::kPositional) {
      if (++positional_count >= *argc) {
        TFLITE_LOG(ERROR) << "Too few command line arguments.";
        return false;
      }
      bool value_parsing_ok;
      flag.Parse(argv[positional_count], positional_count, &value_parsing_ok);
      if (!value_parsing_ok) {
        TFLITE_LOG(ERROR) << "Failed to parse positional flag: " << flag.name_;
        return false;
      }
      unknown_argvs[positional_count] = false;
      processed_flags[flag.name_] = positional_count;
      continue;
    }

    // Parse other flags.
    bool was_found = false;
    for (int i = positional_count + 1; i < *argc; ++i) {
      if (!unknown_argvs[i]) continue;
      bool value_parsing_ok;
      was_found = flag.Parse(argv[i], i, &value_parsing_ok);
      if (!value_parsing_ok) {
        TFLITE_LOG(ERROR) << "Failed to parse flag '" << flag.name_
                          << "' against argv '" << argv[i] << "'";
        result = false;
      }
      if (was_found) {
        unknown_argvs[i] = false;
        processed_flags[flag.name_] = i;
        break;
      }
    }

    // If the flag is found from the argv (i.e. the flag name appears in argv),
    // continue to the next flag parsing.
    if (was_found) continue;

    // The flag isn't found, do some bookkeeping work.
    processed_flags[flag.name_] = -1;
    if (flag.flag_type_ == Flag::kRequired) {
      TFLITE_LOG(ERROR) << "Required flag not provided: " << flag.name_;
      result = false;
      // If the required flag isn't found, we immediately stop the whole flag
      // parsing by breaking the outer-loop (i.e. the 'sorted_idx'-iteration
      // loop).
      break;
    }
  }

  int dst = 1;  // Skip argv[0]
  for (int i = 1; i < *argc; ++i) {
    if (unknown_argvs[i]) {
      argv[dst++] = argv[i];
    }
  }
  *argc = dst;
  return result && (*argc < 2 || std::strcmp(argv[1], "--help") != 0);
}

/*static*/ std::string Flags::Usage(const std::string& cmdline,
                                    const std::vector<Flag>& flag_list) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("cmdline: \"" + cmdline + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc mht_9(mht_9_v, 464, "", "./tensorflow/lite/tools/command_line_flags.cc", "Flags::Usage");

  // Stores indexes of flag_list in a sorted order.
  std::vector<int> sorted_idx(flag_list.size());
  std::iota(std::begin(sorted_idx), std::end(sorted_idx), 0);
  std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(), [&flag_list](int a, int b) {
        return flag_list[a].GetFlagType() < flag_list[b].GetFlagType();
      });
  // Counts number of positional flags will be shown.
  int positional_count = 0;
  std::ostringstream usage_text;
  usage_text << "usage: " << cmdline;
  // Prints usage for positional flag.
  for (int i = 0; i < sorted_idx.size(); ++i) {
    const Flag& flag = flag_list[sorted_idx[i]];
    if (flag.flag_type_ == Flag::kPositional) {
      positional_count++;
      usage_text << " <" << flag.name_ << ">";
    } else {
      usage_text << " <flags>";
      break;
    }
  }
  usage_text << "\n";

  // Finds the max number of chars of the name column in the usage message.
  int max_name_width = 0;
  std::vector<std::string> name_column(flag_list.size());
  for (int i = 0; i < sorted_idx.size(); ++i) {
    const Flag& flag = flag_list[sorted_idx[i]];
    if (flag.flag_type_ != Flag::kPositional) {
      name_column[i] += "--";
      name_column[i] += flag.name_;
      name_column[i] += "=";
      name_column[i] += flag.default_for_display_;
    } else {
      name_column[i] += flag.name_;
    }
    if (name_column[i].size() > max_name_width) {
      max_name_width = name_column[i].size();
    }
  }

  if (positional_count > 0) {
    usage_text << "Where:\n";
  }
  for (int i = 0; i < sorted_idx.size(); ++i) {
    const Flag& flag = flag_list[sorted_idx[i]];
    if (i == positional_count) {
      usage_text << "Flags:\n";
    }
    auto type_name = flag.GetTypeName();
    usage_text << "\t";
    usage_text << std::left << std::setw(max_name_width) << name_column[i];
    usage_text << "\t" << type_name << "\t";
    usage_text << (flag.flag_type_ != Flag::kOptional ? "required"
                                                      : "optional");
    usage_text << "\t" << flag.usage_text_ << "\n";
  }
  return usage_text.str();
}

/*static*/ std::string Flags::ArgsToString(int argc, const char** argv) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTcc mht_10(mht_10_v, 529, "", "./tensorflow/lite/tools/command_line_flags.cc", "Flags::ArgsToString");

  std::string args;
  for (int i = 1; i < argc; ++i) {
    args.append(argv[i]);
    if (i != argc - 1) args.append(" ");
  }
  return args;
}

}  // namespace tflite
