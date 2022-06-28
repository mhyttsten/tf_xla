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
class MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <cinttypes>
#include <cstring>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {

bool ParseStringFlag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                     const std::function<bool(string)>& hook,
                     bool* value_parsing_ok) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/util/command_line_flags.cc", "ParseStringFlag");

  *value_parsing_ok = true;
  if (absl::ConsumePrefix(&arg, "--") && absl::ConsumePrefix(&arg, flag) &&
      absl::ConsumePrefix(&arg, "=")) {
    *value_parsing_ok = hook(string(arg));
    return true;
  }

  return false;
}

bool ParseInt32Flag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                    const std::function<bool(int32_t)>& hook,
                    bool* value_parsing_ok) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/util/command_line_flags.cc", "ParseInt32Flag");

  *value_parsing_ok = true;
  if (absl::ConsumePrefix(&arg, "--") && absl::ConsumePrefix(&arg, flag) &&
      absl::ConsumePrefix(&arg, "=")) {
    char extra;
    int32_t parsed_int32;
    if (sscanf(arg.data(), "%d%c", &parsed_int32, &extra) != 1) {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    } else {
      *value_parsing_ok = hook(parsed_int32);
    }
    return true;
  }

  return false;
}

bool ParseInt64Flag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                    const std::function<bool(int64_t)>& hook,
                    bool* value_parsing_ok) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/util/command_line_flags.cc", "ParseInt64Flag");

  *value_parsing_ok = true;
  if (absl::ConsumePrefix(&arg, "--") && absl::ConsumePrefix(&arg, flag) &&
      absl::ConsumePrefix(&arg, "=")) {
    char extra;
    int64_t parsed_int64;
    if (sscanf(arg.data(), "%" SCNd64 "%c", &parsed_int64, &extra) != 1) {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    } else {
      *value_parsing_ok = hook(parsed_int64);
    }
    return true;
  }

  return false;
}

bool ParseBoolFlag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                   const std::function<bool(bool)>& hook,
                   bool* value_parsing_ok) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_3(mht_3_v, 265, "", "./tensorflow/core/util/command_line_flags.cc", "ParseBoolFlag");

  *value_parsing_ok = true;
  if (absl::ConsumePrefix(&arg, "--") && absl::ConsumePrefix(&arg, flag)) {
    if (arg.empty()) {
      *value_parsing_ok = hook(true);
      return true;
    }

    if (arg == "=true") {
      *value_parsing_ok = hook(true);
      return true;
    } else if (arg == "=false") {
      *value_parsing_ok = hook(false);
      return true;
    } else {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
      return true;
    }
  }

  return false;
}

bool ParseFloatFlag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                    const std::function<bool(float)>& hook,
                    bool* value_parsing_ok) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_4(mht_4_v, 295, "", "./tensorflow/core/util/command_line_flags.cc", "ParseFloatFlag");

  *value_parsing_ok = true;
  if (absl::ConsumePrefix(&arg, "--") && absl::ConsumePrefix(&arg, flag) &&
      absl::ConsumePrefix(&arg, "=")) {
    char extra;
    float parsed_float;
    if (sscanf(arg.data(), "%f%c", &parsed_float, &extra) != 1) {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    } else {
      *value_parsing_ok = hook(parsed_float);
    }
    return true;
  }

  return false;
}

}  // namespace

Flag::Flag(const char* name, tensorflow::int32* dst, const string& usage_text,
           bool* dst_updated)
    : name_(name),
      type_(TYPE_INT32),
      int32_hook_([dst, dst_updated](int32_t value) {
        *dst = value;
        if (dst_updated) *dst_updated = true;
        return true;
      }),
      int32_default_for_display_(*dst),
      usage_text_(usage_text) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_5_v.push_back("usage_text: \"" + usage_text + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_5(mht_5_v, 331, "", "./tensorflow/core/util/command_line_flags.cc", "Flag::Flag");
}

Flag::Flag(const char* name, int64_t* dst, const string& usage_text,
           bool* dst_updated)
    : name_(name),
      type_(TYPE_INT64),
      int64_hook_([dst, dst_updated](int64_t value) {
        *dst = value;
        if (dst_updated) *dst_updated = true;
        return true;
      }),
      int64_default_for_display_(*dst),
      usage_text_(usage_text) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_6_v.push_back("usage_text: \"" + usage_text + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_6(mht_6_v, 348, "", "./tensorflow/core/util/command_line_flags.cc", "Flag::Flag");
}

Flag::Flag(const char* name, float* dst, const string& usage_text,
           bool* dst_updated)
    : name_(name),
      type_(TYPE_FLOAT),
      float_hook_([dst, dst_updated](float value) {
        *dst = value;
        if (dst_updated) *dst_updated = true;
        return true;
      }),
      float_default_for_display_(*dst),
      usage_text_(usage_text) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_7_v.push_back("usage_text: \"" + usage_text + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_7(mht_7_v, 365, "", "./tensorflow/core/util/command_line_flags.cc", "Flag::Flag");
}

Flag::Flag(const char* name, bool* dst, const string& usage_text,
           bool* dst_updated)
    : name_(name),
      type_(TYPE_BOOL),
      bool_hook_([dst, dst_updated](bool value) {
        *dst = value;
        if (dst_updated) *dst_updated = true;
        return true;
      }),
      bool_default_for_display_(*dst),
      usage_text_(usage_text) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_8_v.push_back("usage_text: \"" + usage_text + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_8(mht_8_v, 382, "", "./tensorflow/core/util/command_line_flags.cc", "Flag::Flag");
}

Flag::Flag(const char* name, string* dst, const string& usage_text,
           bool* dst_updated)
    : name_(name),
      type_(TYPE_STRING),
      string_hook_([dst, dst_updated](string value) {
        *dst = std::move(value);
        if (dst_updated) *dst_updated = true;
        return true;
      }),
      string_default_for_display_(*dst),
      usage_text_(usage_text) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_9_v.push_back("usage_text: \"" + usage_text + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_9(mht_9_v, 399, "", "./tensorflow/core/util/command_line_flags.cc", "Flag::Flag");
}

Flag::Flag(const char* name, std::function<bool(int32_t)> int32_hook,
           int32_t default_value_for_display, const string& usage_text)
    : name_(name),
      type_(TYPE_INT32),
      int32_hook_(std::move(int32_hook)),
      int32_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_10_v.push_back("usage_text: \"" + usage_text + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_10(mht_10_v, 412, "", "./tensorflow/core/util/command_line_flags.cc", "Flag::Flag");
}

Flag::Flag(const char* name, std::function<bool(int64_t)> int64_hook,
           int64_t default_value_for_display, const string& usage_text)
    : name_(name),
      type_(TYPE_INT64),
      int64_hook_(std::move(int64_hook)),
      int64_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_11_v.push_back("usage_text: \"" + usage_text + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_11(mht_11_v, 425, "", "./tensorflow/core/util/command_line_flags.cc", "Flag::Flag");
}

Flag::Flag(const char* name, std::function<bool(float)> float_hook,
           float default_value_for_display, const string& usage_text)
    : name_(name),
      type_(TYPE_FLOAT),
      float_hook_(std::move(float_hook)),
      float_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_12_v.push_back("usage_text: \"" + usage_text + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_12(mht_12_v, 438, "", "./tensorflow/core/util/command_line_flags.cc", "Flag::Flag");
}

Flag::Flag(const char* name, std::function<bool(bool)> bool_hook,
           bool default_value_for_display, const string& usage_text)
    : name_(name),
      type_(TYPE_BOOL),
      bool_hook_(std::move(bool_hook)),
      bool_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_13_v.push_back("usage_text: \"" + usage_text + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_13(mht_13_v, 451, "", "./tensorflow/core/util/command_line_flags.cc", "Flag::Flag");
}

Flag::Flag(const char* name, std::function<bool(string)> string_hook,
           string default_value_for_display, const string& usage_text)
    : name_(name),
      type_(TYPE_STRING),
      string_hook_(std::move(string_hook)),
      string_default_for_display_(std::move(default_value_for_display)),
      usage_text_(usage_text) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_14_v.push_back("default_value_for_display: \"" + default_value_for_display + "\"");
   mht_14_v.push_back("usage_text: \"" + usage_text + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_14(mht_14_v, 465, "", "./tensorflow/core/util/command_line_flags.cc", "Flag::Flag");
}

bool Flag::Parse(string arg, bool* value_parsing_ok) const {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("arg: \"" + arg + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_15(mht_15_v, 471, "", "./tensorflow/core/util/command_line_flags.cc", "Flag::Parse");

  bool result = false;
  if (type_ == TYPE_INT32) {
    result = ParseInt32Flag(arg, name_, int32_hook_, value_parsing_ok);
  } else if (type_ == TYPE_INT64) {
    result = ParseInt64Flag(arg, name_, int64_hook_, value_parsing_ok);
  } else if (type_ == TYPE_BOOL) {
    result = ParseBoolFlag(arg, name_, bool_hook_, value_parsing_ok);
  } else if (type_ == TYPE_STRING) {
    result = ParseStringFlag(arg, name_, string_hook_, value_parsing_ok);
  } else if (type_ == TYPE_FLOAT) {
    result = ParseFloatFlag(arg, name_, float_hook_, value_parsing_ok);
  }
  return result;
}

/*static*/ bool Flags::Parse(int* argc, char** argv,
                             const std::vector<Flag>& flag_list) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_16(mht_16_v, 491, "", "./tensorflow/core/util/command_line_flags.cc", "Flags::Parse");

  bool result = true;
  std::vector<char*> unknown_flags;
  for (int i = 1; i < *argc; ++i) {
    if (string(argv[i]) == "--") {
      while (i < *argc) {
        unknown_flags.push_back(argv[i]);
        ++i;
      }
      break;
    }

    bool was_found = false;
    for (const Flag& flag : flag_list) {
      bool value_parsing_ok;
      was_found = flag.Parse(argv[i], &value_parsing_ok);
      if (!value_parsing_ok) {
        result = false;
      }
      if (was_found) {
        break;
      }
    }
    if (!was_found) {
      unknown_flags.push_back(argv[i]);
    }
  }
  // Passthrough any extra flags.
  int dst = 1;  // Skip argv[0]
  for (char* f : unknown_flags) {
    argv[dst++] = f;
  }
  argv[dst++] = nullptr;
  *argc = unknown_flags.size() + 1;
  return result && (*argc < 2 || strcmp(argv[1], "--help") != 0);
}

/*static*/ string Flags::Usage(const string& cmdline,
                               const std::vector<Flag>& flag_list) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("cmdline: \"" + cmdline + "\"");
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTcc mht_17(mht_17_v, 533, "", "./tensorflow/core/util/command_line_flags.cc", "Flags::Usage");

  string usage_text;
  if (!flag_list.empty()) {
    strings::Appendf(&usage_text, "usage: %s\nFlags:\n", cmdline.c_str());
  } else {
    strings::Appendf(&usage_text, "usage: %s\n", cmdline.c_str());
  }
  for (const Flag& flag : flag_list) {
    const char* type_name = "";
    string flag_string;
    if (flag.type_ == Flag::TYPE_INT32) {
      type_name = "int32";
      flag_string = strings::Printf("--%s=%d", flag.name_.c_str(),
                                    flag.int32_default_for_display_);
    } else if (flag.type_ == Flag::TYPE_INT64) {
      type_name = "int64";
      flag_string = strings::Printf(
          "--%s=%lld", flag.name_.c_str(),
          static_cast<long long>(flag.int64_default_for_display_));
    } else if (flag.type_ == Flag::TYPE_BOOL) {
      type_name = "bool";
      flag_string =
          strings::Printf("--%s=%s", flag.name_.c_str(),
                          flag.bool_default_for_display_ ? "true" : "false");
    } else if (flag.type_ == Flag::TYPE_STRING) {
      type_name = "string";
      flag_string = strings::Printf("--%s=\"%s\"", flag.name_.c_str(),
                                    flag.string_default_for_display_.c_str());
    } else if (flag.type_ == Flag::TYPE_FLOAT) {
      type_name = "float";
      flag_string = strings::Printf("--%s=%f", flag.name_.c_str(),
                                    flag.float_default_for_display_);
    }
    strings::Appendf(&usage_text, "\t%-33s\t%s\t%s\n", flag_string.c_str(),
                     type_name, flag.usage_text_.c_str());
  }
  return usage_text;
}

}  // namespace tensorflow
