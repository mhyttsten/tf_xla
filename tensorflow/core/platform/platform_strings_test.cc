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
class MHTracer_DTPStensorflowPScorePSplatformPSplatform_strings_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSplatform_strings_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSplatform_strings_testDTcc() {
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

// Test for the platform_strings.h header file.

#include "tensorflow/core/platform/platform_strings.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/str_util.h"

// Embed the platform strings in this binary.
TF_PLATFORM_STRINGS()

// A vector of strings.
typedef std::vector<std::string> string_vec;

// Append to *found the strings within the named file with the platform_strings
// magic prefix, and return true; or return false on error.

// Print the platform strings embedded in the binary file_name and return 0,
// or on error return 2.
static int PrintStrings(const std::string file_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSplatform_strings_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/platform/platform_strings_test.cc", "PrintStrings");

  int rc = 0;
  string_vec str;
  if (!tensorflow::GetPlatformStrings(file_name, &str)) {
    for (int i = 0; i != str.size(); i++) {
      printf("%s\n", str[i].c_str());
    }
  } else {
    perror(file_name.c_str());
    rc = 2;
  }
  return rc;
}

// Return whether str[] contains a string with prefix "macro_name="; if so,
// set *pvalue to the suffix.
static bool GetValue(const string_vec &str, const std::string &macro_name,
                     std::string *pvalue) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("macro_name: \"" + macro_name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSplatform_strings_testDTcc mht_1(mht_1_v, 236, "", "./tensorflow/core/platform/platform_strings_test.cc", "GetValue");

  std::string nam_eq = macro_name + "=";
  int i = 0;
  while (i != str.size() && !absl::StartsWith(str[i], nam_eq)) {
    i++;
  }
  bool found = (i != str.size());
  if (found) {
    *pvalue = str[i].substr(nam_eq.size());
  }
  return found;
}

// If macro_name[] is not equal to value[], check that str[] contains the
// string "macro_name=value".  Otherwise, check that str[] does not contain any
// string starting with macro_name=".
static void CheckStr(const string_vec &str, const std::string &macro_name,
                     const std::string &value) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("macro_name: \"" + macro_name + "\"");
   mht_2_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSplatform_strings_testDTcc mht_2(mht_2_v, 258, "", "./tensorflow/core/platform/platform_strings_test.cc", "CheckStr");

  std::string value_from_str;
  if (GetValue(str, macro_name, &value_from_str)) {
    if (value != value_from_str) {
      // Output everything found, to aid debugging.
      LOG(ERROR) << "===== value=" << value
                 << "  value_from_str=" << value_from_str;
      for (int i = 0; i != str.size(); i++) {
        LOG(ERROR) << "% " << str[i];
      }
      LOG(ERROR) << "=====";
    }
    CHECK_EQ(value, value_from_str) << " " << macro_name << ": bad value";
  } else {
    // If the string is not found, we expect value to be macro_name.
    if (value != macro_name) {
      // Output everything found, to aid debugging.
      LOG(ERROR) << "===== value=" << value << "  macro_name=" << macro_name;
      for (int i = 0; i != str.size(); i++) {
        LOG(ERROR) << "% " << str[i];
      }
      LOG(ERROR) << "=====";
    }
    CHECK_EQ(value, macro_name) << " " << macro_name << ": not found in binary";
  }
}

// Helper for AS_STR(), below, to perform macro expansion.
#define AS_STR_1_(x) #x

// Yield x after macro expansion as a nul-terminated constant string.
#define AS_STR(x) AS_STR_1_(x)

// Run the test, and return 0 on success, 2 otherwise.
static int RunTest(const std::string &binary_name) {
  int rc = 0;
  string_vec str;

  if (!tensorflow::GetPlatformStrings(binary_name, &str)) {
    CheckStr(str, "__linux__", AS_STR(__linux__));
    CheckStr(str, "_WIN32", AS_STR(_WIN32));
    CheckStr(str, "__APPLE__", AS_STR(__APPLE__));
    CheckStr(str, "__x86_64__", AS_STR(__x86_64__));
    CheckStr(str, "__aarch64__", AS_STR(__aarch64__));
    CheckStr(str, "__powerpc64__", AS_STR(__powerpc64__));
    CheckStr(str, "TF_PLAT_STR_VERSION", TF_PLAT_STR_VERSION_);
  } else {
    perror(binary_name.c_str());
    rc = 2;
  }

  return rc;
}

int main(int argc, char *argv[]) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSplatform_strings_testDTcc mht_3(mht_3_v, 315, "", "./tensorflow/core/platform/platform_strings_test.cc", "main");

  tensorflow::Env *env = tensorflow::Env::Default();
  static const char usage[] = "usage: platform_strings_test [file...]";
  int rc = 0;
  tensorflow::port::InitMain(usage, &argc, &argv);
  if (argc == 1) {
    printf("rc=%d\n", PrintStrings(env->GetExecutablePath()));
    rc = RunTest(env->GetExecutablePath());
  } else {
    for (int argn = 1; argn != argc; argn++) {
      rc |= PrintStrings(argv[argn]);
    }
  }
  return rc;
}
