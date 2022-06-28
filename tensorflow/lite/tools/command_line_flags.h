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

#ifndef TENSORFLOW_LITE_TOOLS_COMMAND_LINE_FLAGS_H_
#define TENSORFLOW_LITE_TOOLS_COMMAND_LINE_FLAGS_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTh() {
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


#include <functional>
#include <string>
#include <vector>

namespace tflite {
// A simple command-line argument parsing module.
// Dependency free simplified port of core/util/command_line_flags.
// This class is written for benchmarks and uses inefficient string
// concatenation. This was written to avoid dependency on tensorflow/core/util
// which transitively brings in a lot of other dependencies that are not
// necessary for tflite benchmarking code.
// The recommended way of using it is with local variables and an initializer
// list of Flag objects, for example:
//
// int some_int = 10;
// bool some_switch = false;
// std::string some_name = "something";
//
// std::vector<tensorFlow::Flag> flag_list = {
//   Flag::CreateFlag("some_int", &some_int, "an integer that affects X"),
//   Flag::CreateFlag("some_switch", &some_switch, "a bool that affects Y"),
//   Flag::CreateFlag("some_name", &some_name, "a string that affects Z")
// };
// // Get usage message before ParseFlags() to capture default values.
// std::string usage = Flag::Usage(argv[0], flag_list);
// bool parsed_values_ok = Flags::Parse(&argc, argv, flag_list);
//
// tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
// if (argc != 1 || !parsed_values_ok) {
//    ...output usage and error message...
// }
//
// The argc and argv values are adjusted by the Parse function so all that
// remains is the program name (at argv[0]) and any unknown arguments fill the
// rest of the array. This means you can check for flags that weren't understood
// by seeing if argv is greater than 1.
// The result indicates if there were any errors parsing the values that were
// passed to the command-line switches. For example, --some_int=foo would return
// false because the argument is expected to be an integer.
//
// NOTE: Unlike gflags-style libraries, this library is intended to be
// used in the `main()` function of your binary. It does not handle
// flag definitions that are scattered around the source code.

// A description of a single command line flag, holding its name, type, usage
// text, and a pointer to the corresponding variable.
class Flag {
 public:
  enum FlagType {
    kPositional = 0,
    kRequired,
    kOptional,
  };

  // The order of the positional flags is the same as they are added.
  // Positional flags are supposed to be required.
  template <typename T>
  static Flag CreateFlag(const char* name, T* val, const char* usage,
                         FlagType flag_type = kOptional) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_0_v.push_back("usage: \"" + (usage == nullptr ? std::string("nullptr") : std::string((char*)usage)) + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTh mht_0(mht_0_v, 248, "", "./tensorflow/lite/tools/command_line_flags.h", "CreateFlag");

    return Flag(
        name, [val](const T& v) { *val = v; }, *val, usage, flag_type);
  }

// "flag_T" is same as "default_value_T" for trivial types, like int32, bool
// etc. But when it's a complex type, "default_value_T" is generally a const
// reference "flag_T".
#define CONSTRUCTOR_WITH_ARGV_INDEX(flag_T, default_value_T)         \
  Flag(const char* name,                                             \
       const std::function<void(const flag_T& /*flag_val*/,          \
                                int /*argv_position*/)>& hook,       \
       default_value_T default_value, const std::string& usage_text, \
       FlagType flag_type);

#define CONSTRUCTOR_WITHOUT_ARGV_INDEX(flag_T, default_value_T)            \
  Flag(const char* name, const std::function<void(const flag_T&)>& hook,   \
       default_value_T default_value, const std::string& usage_text,       \
       FlagType flag_type)                                                 \
      : Flag(                                                              \
            name, [hook](const flag_T& flag_val, int) { hook(flag_val); }, \
            default_value, usage_text, flag_type) {}

  CONSTRUCTOR_WITH_ARGV_INDEX(int32_t, int32_t)
  CONSTRUCTOR_WITHOUT_ARGV_INDEX(int32_t, int32_t)

  CONSTRUCTOR_WITH_ARGV_INDEX(int64_t, int64_t)
  CONSTRUCTOR_WITHOUT_ARGV_INDEX(int64_t, int64_t)

  CONSTRUCTOR_WITH_ARGV_INDEX(float, float)
  CONSTRUCTOR_WITHOUT_ARGV_INDEX(float, float)

  CONSTRUCTOR_WITH_ARGV_INDEX(bool, bool)
  CONSTRUCTOR_WITHOUT_ARGV_INDEX(bool, bool)

  CONSTRUCTOR_WITH_ARGV_INDEX(std::string, const std::string&)
  CONSTRUCTOR_WITHOUT_ARGV_INDEX(std::string, const std::string&)

#undef CONSTRUCTOR_WITH_ARGV_INDEX
#undef CONSTRUCTOR_WITHOUT_ARGV_INDEX

  FlagType GetFlagType() const { return flag_type_; }

  std::string GetFlagName() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPScommand_line_flagsDTh mht_1(mht_1_v, 294, "", "./tensorflow/lite/tools/command_line_flags.h", "GetFlagName");
 return name_; }

 private:
  friend class Flags;

  bool Parse(const std::string& arg, int argv_position,
             bool* value_parsing_ok) const;

  std::string name_;
  enum {
    TYPE_INT32,
    TYPE_INT64,
    TYPE_BOOL,
    TYPE_STRING,
    TYPE_FLOAT,
  } type_;

  std::string GetTypeName() const;

  std::function<bool(const std::string& /*read_value*/, int /*argv_position*/)>
      value_hook_;
  std::string default_for_display_;

  std::string usage_text_;
  FlagType flag_type_;
};

class Flags {
 public:
  // Parse the command line represented by argv[0, ..., (*argc)-1] to find flag
  // instances matching flags in flaglist[].  Update the variables associated
  // with matching flags, and remove the matching arguments from (*argc, argv).
  // Return true iff all recognized flag values were parsed correctly, and the
  // first remaining argument is not "--help".
  // Note:
  // 1. when there are duplicate args in argv for the same flag, the flag value
  // and the parse result will be based on the 1st arg.
  // 2. when there are duplicate flags in flag_list (i.e. two flags having the
  // same name), all of them will be checked against the arg list and the parse
  // result will be false if any of the parsing fails.
  // See *Duplicate* unit tests in command_line_flags_test.cc for the
  // illustration of such behaviors.
  static bool Parse(int* argc, const char** argv,
                    const std::vector<Flag>& flag_list);

  // Return a usage message with command line cmdline, and the
  // usage_text strings in flag_list[].
  static std::string Usage(const std::string& cmdline,
                           const std::vector<Flag>& flag_list);

  // Return a space separated string containing argv[1, ..., argc-1].
  static std::string ArgsToString(int argc, const char** argv);
};
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_COMMAND_LINE_FLAGS_H_
