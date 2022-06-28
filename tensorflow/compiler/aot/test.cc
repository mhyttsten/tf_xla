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
class MHTracer_DTPStensorflowPScompilerPSaotPStestDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSaotPStestDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSaotPStestDTcc() {
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

// Generated by the tf_library build rule.  DO NOT EDIT!
//
// This file contains a test and benchmark for the function generated by
// tfcompile.  All tokens of the form `{{TFCOMPILE_*}}` must be rewritten to
// real values before this file can be compiled.
//
//    TFCOMPILE_HEADER    : Path to the header file generated by tfcompile.
//    TFCOMPILE_CPP_CLASS : Name of the C++ class generated by tfcompile.
//    TFCOMPILE_NAME      : Name for tests and benchmarks.
//
// The tf_library bazel macro in tfcompile.bzl performs the token rewriting, and
// generates a cc_test rule for you.

// These macros must be defined before eigen files are included.
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

// clang-format off
#include "{{TFCOMPILE_HEADER}}"  // NOLINT(whitespace/braces)
// clang-format on

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

// Macros that expand to tokens based on the entry point name.
// clang-format off
#define CPP_CLASS {{TFCOMPILE_CPP_CLASS}}  // NOLINT(whitespace/braces)
#define TEST_NAME {{TFCOMPILE_NAME}}Test   // NOLINT(whitespace/braces)
#define BM_NAME   BM_{{TFCOMPILE_NAME}}    // NOLINT(whitespace/braces)
// clang-format on

namespace tensorflow {
namespace tfcompile {
namespace {

void zero_buffers(XlaCompiledCpuFunction* computation) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSaotPStestDTcc mht_0(mht_0_v, 223, "", "./tensorflow/compiler/aot/test.cc", "zero_buffers");

  for (int i = 0; i < computation->num_args(); ++i) {
    memset(computation->arg_data(i), 0, computation->arg_size(i));
  }
}

// Trivial test that runs the generated function to ensure it doesn't crash.
TEST(TEST_NAME, NoCrash) {
  Eigen::ThreadPool pool(port::MaxParallelism());
  Eigen::ThreadPoolDevice device(&pool, pool.NumThreads());

  CPP_CLASS computation;
  computation.set_thread_pool(&device);
  zero_buffers(&computation);

  EXPECT_TRUE(computation.Run());
}

// Simple benchmark that repeatedly runs the generated function.
void BM_NAME(benchmark::State& state) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSaotPStestDTcc mht_1(mht_1_v, 245, "", "./tensorflow/compiler/aot/test.cc", "BM_NAME");

  Eigen::ThreadPool pool(port::MaxParallelism());
  Eigen::ThreadPoolDevice device(&pool, pool.NumThreads());

  CPP_CLASS computation;
  computation.set_thread_pool(&device);
  zero_buffers(&computation);

  for (auto s : state) {
    computation.Run();
  }
}
BENCHMARK(BM_NAME);

}  // namespace
}  // namespace tfcompile
}  // namespace tensorflow
