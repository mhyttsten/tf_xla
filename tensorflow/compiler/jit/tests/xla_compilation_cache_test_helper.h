/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_JIT_TESTS_XLA_COMPILATION_CACHE_TEST_HELPER_H_
#define TENSORFLOW_COMPILER_JIT_TESTS_XLA_COMPILATION_CACHE_TEST_HELPER_H_
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
class MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTh {
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
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTh() {
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


#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// A listener to inspect the use of XLA's persistent compilation cache entries.
class JitCompilationListener : public XlaActivityListener {
 public:
  Status Listen(
      const XlaAutoClusteringActivity& auto_clustering_activity) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTh mht_0(mht_0_v, 202, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h", "Listen");

    return Status::OK();
  }

  Status Listen(
      const XlaJitCompilationActivity& jit_compilation_activity) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTh mht_1(mht_1_v, 210, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h", "Listen");

    used_persistent_cache_.push_back(
        jit_compilation_activity.used_persistent_cache());
    return Status::OK();
  }

  Status Listen(const XlaOptimizationRemark& optimization_remark) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTh mht_2(mht_2_v, 219, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h", "Listen");

    return Status::OK();
  }

  ~JitCompilationListener() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTh mht_3(mht_3_v, 226, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h", "~JitCompilationListener");
}

  Status VerifyListenerHistory(bool expect_persistent_cache_use) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTh mht_4(mht_4_v, 231, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h", "VerifyListenerHistory");

    for (bool used_persistent_cache : used_persistent_cache_) {
      if (used_persistent_cache != expect_persistent_cache_use) {
        return errors::FailedPrecondition("Unexpected listener history.");
      }
    }
    return Status::OK();
  }

  void ClearListenerHistory() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTh mht_5(mht_5_v, 243, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h", "ClearListenerHistory");
 used_persistent_cache_.clear(); }

 private:
  std::vector<bool> used_persistent_cache_;
};

// Fixture for testing XLA compilation cache serialization.
class XlaCompilationCacheSerializeTest : public ::testing::Test {
 protected:
  XlaCompilationCacheSerializeTest() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTh mht_6(mht_6_v, 255, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h", "XlaCompilationCacheSerializeTest");

    auto listener = absl::make_unique<JitCompilationListener>();
    listener_ = listener.get();
    RegisterXlaActivityListener(std::move(listener));
  }

  JitCompilationListener* listener() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTh mht_7(mht_7_v, 264, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h", "listener");
 return listener_; }

  // Returns a test graph that will split into two XLA clusters (due to a node
  // with _XlaCompile = false).
  GraphDef GetTestGraph(const PartialTensorShape& input_shape);

  // Runs the graph using specified batch size both with and without XLA JIT
  // compilation. Returns an error if the results between the two do not match.
  Status ExecuteWithBatch(const GraphDef& graph, int batch);

  // Adds the suffix "_altered" to the HLO module names of all of the persistent
  // XLA compilation cache entries found at the specified directory. If none are
  // found, returns NOT_FOUND error.
  Status AlterPersistentCacheEntryHloModuleNames(
      absl::string_view persistent_cache_dir_path,
      absl::string_view file_prefix = "xla_compile_cache");

 private:
  JitCompilationListener* listener_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_TESTS_XLA_COMPILATION_CACHE_TEST_HELPER_H_
