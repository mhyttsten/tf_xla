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

#ifndef TENSORFLOW_CORE_UTIL_COMMAND_LINE_FLAGS_H
#define TENSORFLOW_CORE_UTIL_COMMAND_LINE_FLAGS_H
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
class MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTh() {
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
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// N.B. This library is for INTERNAL use only.
//
// This is a simple command-line argument parsing module to help us handle
// parameters for C++ binaries. The recommended way of using it is with local
// variables and an initializer list of Flag objects, for example:
//
// int some_int = 10;
// bool some_switch = false;
// string some_name = "something";
// std::vector<tensorFlow::Flag> flag_list = {
//   Flag("some_int", &some_int, "an integer that affects X"),
//   Flag("some_switch", &some_switch, "a bool that affects Y"),
//   Flag("some_name", &some_name, "a string that affects Z")
// };
// // Get usage message before ParseFlags() to capture default values.
// string usage = Flag::Usage(argv[0], flag_list);
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
  Flag(const char* name, int32* dst, const string& usage_text,
       bool* dst_updated = nullptr);
  Flag(const char* name, int64_t* dst, const string& usage_text,
       bool* dst_updated = nullptr);
  Flag(const char* name, bool* dst, const string& usage_text,
       bool* dst_updated = nullptr);
  Flag(const char* name, string* dst, const string& usage_text,
       bool* dst_updated = nullptr);
  Flag(const char* name, float* dst, const string& usage_text,
       bool* dst_updated = nullptr);

  // These constructors invoke a hook on a match instead of writing to a
  // specific memory location.  The hook may return false to signal a malformed
  // or illegal value, which will then fail the command line parse.
  //
  // "default_value_for_display" is shown as the default value of this flag in
  // Flags::Usage().
  Flag(const char* name, std::function<bool(int32_t)> int32_hook,
       int32_t default_value_for_display, const string& usage_text);
  Flag(const char* name, std::function<bool(int64_t)> int64_hook,
       int64_t default_value_for_display, const string& usage_text);
  Flag(const char* name, std::function<bool(float)> float_hook,
       float default_value_for_display, const string& usage_text);
  Flag(const char* name, std::function<bool(bool)> bool_hook,
       bool default_value_for_display, const string& usage_text);
  Flag(const char* name, std::function<bool(string)> string_hook,
       string default_value_for_display, const string& usage_text);

  bool is_default_initialized() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPScommand_line_flagsDTh mht_0(mht_0_v, 262, "", "./tensorflow/core/util/command_line_flags.h", "is_default_initialized");
 return default_initialized_; }

 private:
  friend class Flags;

  bool Parse(string arg, bool* value_parsing_ok) const;

  string name_;
  enum {
    TYPE_INT32,
    TYPE_INT64,
    TYPE_BOOL,
    TYPE_STRING,
    TYPE_FLOAT,
  } type_;

  std::function<bool(int32_t)> int32_hook_;
  int32 int32_default_for_display_;

  std::function<bool(int64_t)> int64_hook_;
  int64_t int64_default_for_display_;

  std::function<bool(float)> float_hook_;
  float float_default_for_display_;

  std::function<bool(bool)> bool_hook_;
  bool bool_default_for_display_;

  std::function<bool(string)> string_hook_;
  string string_default_for_display_;

  string usage_text_;
  bool default_initialized_ = true;
};

class Flags {
 public:
  // Parse the command line represented by argv[0, ..., (*argc)-1] to find flag
  // instances matching flags in flaglist[].  Update the variables associated
  // with matching flags, and remove the matching arguments from (*argc, argv).
  // Return true iff all recognized flag values were parsed correctly, and the
  // first remaining argument is not "--help".
  static bool Parse(int* argc, char** argv, const std::vector<Flag>& flag_list);

  // Return a usage message with command line cmdline, and the
  // usage_text strings in flag_list[].
  static string Usage(const string& cmdline,
                      const std::vector<Flag>& flag_list);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_COMMAND_LINE_FLAGS_H
