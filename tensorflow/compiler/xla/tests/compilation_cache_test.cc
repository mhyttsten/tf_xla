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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompilation_cache_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompilation_cache_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompilation_cache_testDTcc() {
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

#include <initializer_list>
#include <memory>
#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class CompilationCacheTest : public ClientLibraryTestBase {
 public:
  void ExecuteComputationR0F32(const XlaComputation& computation,
                               absl::Span<GlobalData* const> arguments,
                               float expected_result, bool expect_cache_hit) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompilation_cache_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/xla/tests/compilation_cache_test.cc", "ExecuteComputationR0F32");

    ExecutionProfile execution_profile;
    Literal result =
        client_
            ->ExecuteAndTransfer(computation, arguments,
                                 /*execution_options=*/&execution_options_,
                                 &execution_profile)
            .ConsumeValueOrDie();
    EXPECT_TRUE(LiteralTestUtil::Near(
        LiteralUtil::CreateR0<float>(expected_result), result, error_spec_));
    EXPECT_EQ(expect_cache_hit, execution_profile.compilation_cache_hit());
  }

  void ExecuteComputationR2F32(
      const XlaComputation& computation,
      absl::Span<GlobalData* const> arguments,
      std::initializer_list<std::initializer_list<float>> expected_result,
      bool expect_cache_hit) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompilation_cache_testDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/xla/tests/compilation_cache_test.cc", "ExecuteComputationR2F32");

    ExecutionProfile execution_profile;
    auto data_handle = client_
                           ->Execute(computation, arguments,
                                     &execution_options_, &execution_profile)
                           .ConsumeValueOrDie();
    Literal result = client_->Transfer(*data_handle).ConsumeValueOrDie();
    EXPECT_TRUE(LiteralTestUtil::Near(
        LiteralUtil::CreateR2<float>(expected_result), result, error_spec_));
    EXPECT_EQ(expect_cache_hit, execution_profile.compilation_cache_hit());
  }

  ErrorSpec error_spec_{0.0001};
};

// TODO(b/74197823): Disabled because there is no cache in the new design.
XLA_TEST_F(CompilationCacheTest, DISABLED_ComputationCalledMultipleTimes) {
  XlaBuilder builder(TestName());
  Neg(ConstantR0<float>(&builder, 42.0));
  XlaComputation computation = builder.Build().ConsumeValueOrDie();

  ExecuteComputationR0F32(computation, {}, -42.0, /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation, {}, -42.0, /*expect_cache_hit=*/true);
  ExecuteComputationR0F32(computation, {}, -42.0, /*expect_cache_hit=*/true);
}

// TODO(b/74197823): Disabled because there is no cache in the new design.
XLA_TEST_F(CompilationCacheTest,
           DISABLED_ComputationCalledWithDifferentParameters) {
  std::unique_ptr<GlobalData> data_42 =
      client_->TransferToServer(LiteralUtil::CreateR0<float>(42.0f))
          .ConsumeValueOrDie();
  std::unique_ptr<GlobalData> data_123 =
      client_->TransferToServer(LiteralUtil::CreateR0<float>(123.0f))
          .ConsumeValueOrDie();
  std::unique_ptr<GlobalData> data_456 =
      client_->TransferToServer(LiteralUtil::CreateR0<float>(456.0f))
          .ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  Neg(Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "param"));
  XlaComputation computation = builder.Build().ConsumeValueOrDie();

  ExecuteComputationR0F32(computation, {data_42.get()}, -42.0,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation, {data_123.get()}, -123.0,
                          /*expect_cache_hit=*/true);
  ExecuteComputationR0F32(computation, {data_456.get()}, -456.0,
                          /*expect_cache_hit=*/true);
  ExecuteComputationR0F32(computation, {data_42.get()}, -42.0,
                          /*expect_cache_hit=*/true);
}

// TODO(b/74197823): Disabled because there is no cache in the new design.
XLA_TEST_F(CompilationCacheTest, DISABLED_MultipleComputations) {
  XlaBuilder builder_neg(TestName() + "_neg");
  Neg(ConstantR0<float>(&builder_neg, 42.0));
  XlaComputation computation_neg = builder_neg.Build().ConsumeValueOrDie();

  XlaBuilder builder_exp(TestName() + "_exp");
  Exp(ConstantR0<float>(&builder_exp, 1.0));
  XlaComputation computation_exp = builder_exp.Build().ConsumeValueOrDie();

  XlaBuilder builder_add(TestName() + "_add");
  Add(ConstantR0<float>(&builder_add, 2.0),
      ConstantR0<float>(&builder_add, 3.0));
  XlaComputation computation_add = builder_add.Build().ConsumeValueOrDie();

  ExecuteComputationR0F32(computation_neg, {}, -42.0,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation_exp, {}, 2.7182817,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation_add, {}, 5.0,
                          /*expect_cache_hit=*/false);
  ExecuteComputationR0F32(computation_neg, {}, -42.0,
                          /*expect_cache_hit=*/true);
}

// TODO(b/74197823): Disabled because there is no cache in the new design.
XLA_TEST_F(CompilationCacheTest, DISABLED_DifferentParameterLayouts) {
  // Create two GlobalData arrays with the same shape but different
  // layouts. Use these arrays as parameters to a simple computation. If the
  // layout of the array changes then computation should be recompiled (cache
  // miss).
  auto rowmaj_array = LiteralUtil::CreateR2WithLayout(
      {{1.0f, 2.0f}, {3.0f, 4.0f}}, LayoutUtil::MakeLayout({1, 0}));
  auto rowmaj_handle =
      client_->TransferToServer(rowmaj_array).ConsumeValueOrDie();

  auto colmaj_array = LiteralUtil::CreateR2WithLayout(
      {{1.0f, 2.0f}, {3.0f, 4.0f}}, LayoutUtil::MakeLayout({0, 1}));
  auto colmaj_handle =
      client_->TransferToServer(colmaj_array).ConsumeValueOrDie();

  XlaBuilder builder(TestName());
  Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2, 2}), "param0");
  XlaComputation computation = builder.Build().ConsumeValueOrDie();

  ExecuteComputationR2F32(computation, {colmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/false);
  ExecuteComputationR2F32(computation, {colmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/true);
  ExecuteComputationR2F32(computation, {rowmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/false);
  ExecuteComputationR2F32(computation, {rowmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/true);
  ExecuteComputationR2F32(computation, {colmaj_handle.get()},
                          {{1.0f, 2.0f}, {3.0f, 4.0f}},
                          /*expect_cache_hit=*/true);
}

}  // namespace
}  // namespace xla
