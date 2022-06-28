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
class MHTracer_DTPStensorflowPSlitePStestingPSnnapi_exampleDTcc {
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
   MHTracer_DTPStensorflowPSlitePStestingPSnnapi_exampleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPSnnapi_exampleDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
// NOTE: this is an example driver that converts a tflite model to TensorFlow.
// This is an example that will be integrated more tightly into tflite in
// the future.
//
// Usage: bazel run -c opt \
// tensorflow/lite/nnapi:nnapi_example -- <filename>
//
#include <dirent.h>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/testing/parse_testdata.h"
#include "tensorflow/lite/testing/tflite_driver.h"

std::string dirname(const std::string& s) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSnnapi_exampleDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/testing/nnapi_example.cc", "dirname");

  return s.substr(0, s.find_last_of('/'));
}

bool Interpret(const char* examples_filename, bool use_nnapi) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("examples_filename: \"" + (examples_filename == nullptr ? std::string("nullptr") : std::string((char*)examples_filename)) + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSnnapi_exampleDTcc mht_1(mht_1_v, 211, "", "./tensorflow/lite/testing/nnapi_example.cc", "Interpret");

  std::ifstream tflite_stream(examples_filename);
  if (!tflite_stream.is_open()) {
    fprintf(stderr, "Can't open input file.");
    return false;
  }

  printf("Use nnapi is set to: %d\n", use_nnapi);
  tflite::testing::TfLiteDriver test_driver(
      use_nnapi ? tflite::testing::TfLiteDriver::DelegateType::kNnapi
                : tflite::testing::TfLiteDriver::DelegateType::kNone);

  test_driver.SetModelBaseDir(dirname(examples_filename));
  if (!tflite::testing::ParseAndRunTests(&tflite_stream, &test_driver)) {
    fprintf(stderr, "Results from tflite don't match.");
    return false;
  }

  return true;
}

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStestingPSnnapi_exampleDTcc mht_2(mht_2_v, 235, "", "./tensorflow/lite/testing/nnapi_example.cc", "main");

  bool use_nnapi = true;
  if (argc == 4) {
    use_nnapi = strcmp(argv[3], "1") == 0 ? true : false;
  }
  if (argc < 3) {
    fprintf(stderr,
            "Compiled " __DATE__ __TIME__
            "\n"
            "Usage!!!: %s <tflite model> <examples to test> "
            "{ use nn api i.e. 0,1}\n",
            argv[0]);
    return 1;
  }

  std::string base_dir = dirname(argv[1]);
  DIR* dir = opendir(base_dir.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Can't open dir %s\n", base_dir.c_str());
    return 1;
  }
  while (struct dirent* ent = readdir(dir)) {
    std::string name = ent->d_name;
    if (name.rfind(".txt") == name.length() - 4) {
      printf("%s: ", name.c_str());
      if (Interpret((base_dir + "/" + name).c_str(), use_nnapi)) {
        printf(" %s\n", "OK");
      } else {
        printf(" %s\n", "FAIL");
      }
    }
  }
  closedir(dir);

  return 0;
}
