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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cache_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cache_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cache_testDTcc() {
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

#include "tensorflow/compiler/jit/xla_compilation_cache.h"

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

using SignatureHash = XlaCompilationCache::Signature::Hash;

TEST(XlaCompilationCacheTest, SignatureEquality) {
  NameAttrList fn;
  fn.set_name("afunction");
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({4, 0});
  args[0].constant_value = Tensor(DT_INT32, {4, 0});
  TF_ASSERT_OK_AND_ASSIGN(XlaCompilationCache::Signature s1,
                          XlaCompilationCache::BuildSignature(fn, args));

  args[0].type = DT_FLOAT;
  args[0].constant_value = Tensor(DT_FLOAT, {4, 0});
  TF_ASSERT_OK_AND_ASSIGN(XlaCompilationCache::Signature s2,
                          XlaCompilationCache::BuildSignature(fn, args));

  args[0].shape = TensorShape({0, 4});
  args[0].constant_value = Tensor(DT_FLOAT, {0, 4});
  TF_ASSERT_OK_AND_ASSIGN(XlaCompilationCache::Signature s3,
                          XlaCompilationCache::BuildSignature(fn, args));

  std::vector<XlaCompilationCache::Signature> signatures = {s1, s2, s3};
  for (int i = 0; i < signatures.size(); ++i) {
    for (int j = 0; j < signatures.size(); ++j) {
      EXPECT_EQ(i == j, signatures[i] == signatures[j])
          << "s1: " << signatures[i].HumanString() << "\n"
          << "s2: " << signatures[j].HumanString();
      EXPECT_EQ(i == j,
                signatures[i].HumanString() == signatures[j].HumanString())
          << "s1: " << signatures[i].HumanString() << "\n"
          << "s2: " << signatures[j].HumanString();
      EXPECT_EQ(i == j, SignatureHash()(signatures[i]) ==
                            SignatureHash()(signatures[j]))
          << "s1: " << signatures[i].HumanString() << "\n"
          << "s1_hash: " << SignatureHash()(signatures[i]) << "\n"
          << "s2: " << signatures[j].HumanString() << "\n"
          << "s2_hash: " << SignatureHash()(signatures[j]);
    }
  }
}

TEST(XlaCompilationCacheTest, SignatureUniqueness) {
  NameAttrList fn;
  fn.set_name("afunction");
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = DT_INT32;
  args[0].constant_value = Tensor(DT_INT32, {4, 0});

  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({4, 0});

  TF_ASSERT_OK_AND_ASSIGN(XlaCompilationCache::Signature s1,
                          XlaCompilationCache::BuildSignature(fn, args));

  using std::swap;  // go/using-std-swap
  swap(args[0], args[1]);
  TF_ASSERT_OK_AND_ASSIGN(XlaCompilationCache::Signature s2,
                          XlaCompilationCache::BuildSignature(fn, args));

  EXPECT_NE(s1.HumanString(), s2.HumanString());
  EXPECT_NE(SignatureHash()(s1), SignatureHash()(s2));
  EXPECT_FALSE(s1 == s2);
}

void BM_BuildSignature(::testing::benchmark::State& state) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cache_testDTcc mht_0(mht_0_v, 264, "", "./tensorflow/compiler/jit/xla_compilation_cache_test.cc", "BM_BuildSignature");

  const int n_args = state.range(0);

  NameAttrList fn;
  fn.set_name("afunction");
  for (int i = 0; i < n_args; i++) {
    (*fn.mutable_attr())[absl::StrCat("T", i)].set_type(DT_FLOAT);
  }
  std::vector<XlaCompiler::Argument> args(n_args);
  for (int i = 0; i < n_args; i++) {
    args[i].kind = (((i % 3) == 0) ? XlaCompiler::Argument::kConstant
                                   : XlaCompiler::Argument::kParameter);
    args[i].type = DT_INT32;
    args[i].shape = TensorShape({4, 0});
    args[i].constant_value = Tensor(DT_INT32, {4, 0});
  }

  for (auto i : state) {
    StatusOr<XlaCompilationCache::Signature> s =
        XlaCompilationCache::BuildSignature(fn, args);
    CHECK(s.ok());
    XlaCompilationCache::Signature sig = std::move(s.ValueOrDie());
  }
}
BENCHMARK(BM_BuildSignature)->Arg(0)->Arg(1)->Arg(2)->Arg(5)->Arg(10);

}  // namespace
}  // namespace tensorflow
