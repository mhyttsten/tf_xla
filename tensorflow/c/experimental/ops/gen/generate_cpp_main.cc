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
class MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPSgenerate_cpp_mainDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPSgenerate_cpp_mainDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPSgenerate_cpp_mainDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/ops/gen/cpp/cpp_generator.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::string;

namespace generator = tensorflow::generator;

namespace {
class MainConfig {
 public:
  void InitMain(int* argc, char*** argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPSgenerate_cpp_mainDTcc mht_0(mht_0_v, 196, "", "./tensorflow/c/experimental/ops/gen/generate_cpp_main.cc", "InitMain");

    std::vector<tensorflow::Flag> flags = Flags();

    // Parse known flags
    string usage = tensorflow::Flags::Usage(
        absl::StrCat(*argv[0], " Op1 [Op2 ...]"), flags);
    QCHECK(tensorflow::Flags::Parse(argc, *argv, flags)) << usage;  // Crash OK

    // Initialize any TensorFlow support, parsing boilerplate flags (e.g. logs)
    tensorflow::port::InitMain(usage.c_str(), argc, argv);

    // Validate flags
    if (help_) {
      LOG(QFATAL) << usage;  // Crash OK
    }

    QCHECK(!source_dir_.empty()) << usage;  // Crash OK
    QCHECK(!output_dir_.empty()) << usage;  // Crash OK
    QCHECK(!category_.empty()) << usage;    // Crash OK

    // Remaining arguments (i.e. the positional args) are the requested Op names
    op_names_.assign((*argv) + 1, (*argv) + (*argc));
  }

  generator::cpp::CppConfig CppConfig() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPSgenerate_cpp_mainDTcc mht_1(mht_1_v, 223, "", "./tensorflow/c/experimental/ops/gen/generate_cpp_main.cc", "CppConfig");

    return generator::cpp::CppConfig(category_);
  }

  generator::PathConfig PathConfig() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPSgenerate_cpp_mainDTcc mht_2(mht_2_v, 230, "", "./tensorflow/c/experimental/ops/gen/generate_cpp_main.cc", "PathConfig");

    return generator::PathConfig(output_dir_, source_dir_, api_dirs_,
                                 op_names_);
  }

 private:
  std::vector<tensorflow::Flag> Flags() {
    return {
        tensorflow::Flag("help", &help_, "Print this help message."),
        tensorflow::Flag("category", &category_,
                         "Category for generated ops (e.g. 'math', 'array')."),
        tensorflow::Flag(
            "namespace", &name_space_,
            "Compact C++ namespace, default is 'tensorflow::ops'."),
        tensorflow::Flag(
            "output_dir", &output_dir_,
            "Directory into which output files will be generated."),
        tensorflow::Flag(
            "source_dir", &source_dir_,
            "The tensorflow root directory, e.g. 'tensorflow/' for "
            "in-source include paths. Any path underneath the "
            "tensorflow root is also accepted."),
        tensorflow::Flag(
            "api_dirs", &api_dirs_,
            "Comma-separated list of directories containing API definitions.")};
  }

  bool help_ = false;
  string category_;
  string name_space_;
  string output_dir_;
  string source_dir_;
  string api_dirs_;
  std::vector<string> op_names_;
};

}  // namespace

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPSgenerate_cpp_mainDTcc mht_3(mht_3_v, 271, "", "./tensorflow/c/experimental/ops/gen/generate_cpp_main.cc", "main");

  MainConfig config;
  config.InitMain(&argc, &argv);
  generator::CppGenerator generator(config.CppConfig(), config.PathConfig());
  generator.WriteHeaderFile();
  generator.WriteSourceFile();
  return 0;
}
