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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScall_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScall_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScall_testDTcc() {
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

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class CallOpTest : public ClientLibraryTestBase {
 protected:
  XlaComputation CreateR0F32IdentityComputation() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScall_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/tests/call_test.cc", "CreateR0F32IdentityComputation");

    XlaBuilder builder("Identity");
    Parameter(&builder, 0, r0f32_, "x");
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR1S0F32AdditionComputation() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScall_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/tests/call_test.cc", "CreateR1S0F32AdditionComputation");

    XlaBuilder builder("Addition");
    auto x = Parameter(&builder, 0, r1s0f32_, "x");
    auto y = Parameter(&builder, 1, r1s0f32_, "y");
    Add(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR1S2F32AdditionComputation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScall_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/xla/tests/call_test.cc", "CreateR1S2F32AdditionComputation");

    XlaBuilder builder("Addition");
    auto x = Parameter(&builder, 0, r1s2f32_, "x");
    auto y = Parameter(&builder, 1, r1s2f32_, "y");
    Add(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0F32TupleComputation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScall_testDTcc mht_3(mht_3_v, 242, "", "./tensorflow/compiler/xla/tests/call_test.cc", "CreateR0F32TupleComputation");

    XlaBuilder builder("Tuple");
    Tuple(&builder, {Parameter(&builder, 0, r0f32_, "x")});
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
  Shape r1s0f32_ = ShapeUtil::MakeShape(F32, {0});
  Shape r1s2f32_ = ShapeUtil::MakeShape(F32, {2});
};

XLA_TEST_F(CallOpTest, CallR0F32IdentityScalar) {
  XlaBuilder builder(TestName());
  XlaComputation callee = CreateR0F32IdentityComputation();
  auto constant = ConstantLiteral(&builder, LiteralUtil::CreateR0<float>(42.0));
  Call(&builder, callee, {constant});

  ComputeAndCompareR0<float>(&builder, 42.0, {}, ErrorSpec(0.01f));
}

XLA_TEST_F(CallOpTest, CallR1S0F32AddArray) {
  XlaBuilder builder(TestName());
  XlaComputation callee = CreateR1S0F32AdditionComputation();
  auto x = ConstantLiteral(&builder, LiteralUtil::CreateR1<float>({}));
  auto y = ConstantLiteral(&builder, LiteralUtil::CreateR1<float>({}));
  Call(&builder, callee, {x, y});

  ComputeAndCompareR1<float>(&builder, {}, {}, ErrorSpec(0.01f));
}

XLA_TEST_F(CallOpTest, CallR1S2F32AddArray) {
  XlaBuilder builder(TestName());
  XlaComputation callee = CreateR1S2F32AdditionComputation();
  auto x =
      ConstantLiteral(&builder, LiteralUtil::CreateR1<float>({1.0f, 2.0f}));
  auto y =
      ConstantLiteral(&builder, LiteralUtil::CreateR1<float>({2.0f, 3.0f}));
  Call(&builder, callee, {x, y});

  ComputeAndCompareR1<float>(&builder, {3.0f, 5.0f}, {}, ErrorSpec(0.01f));
}

XLA_TEST_F(CallOpTest, CallTreeTwoDeepBranchFactorThree) {
  XlaBuilder builder("inner");
  {
    auto x = Parameter(&builder, 0, r0f32_, "x");
    Add(x, ConstantR0<float>(&builder, 1.0));
  }
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation inner, builder.Build());

  XlaBuilder builder2("outer");
  {
    auto x = Parameter(&builder2, 0, r0f32_, "x");
    x = Call(&builder2, inner, {x});
    x = Call(&builder2, inner, {x});
    x = Call(&builder2, inner, {x});
  }
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation outer, builder2.Build());

  XlaBuilder builder3("outermost");
  {
    auto x = Parameter(&builder3, 0, r0f32_, "x");
    x = Call(&builder3, outer, {x});
    x = Call(&builder3, outer, {x});
    x = Call(&builder3, outer, {x});
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> start,
      client_->TransferToServer(LiteralUtil::CreateR0<float>(1.0f)));
  ComputeAndCompareR0<float>(&builder3, 10.0f, {start.get()}, ErrorSpec(0.0f));
}

XLA_TEST_F(CallOpTest, CallR0F32Tuple) {
  XlaBuilder builder(TestName());
  XlaComputation callee = CreateR0F32TupleComputation();
  auto elem = LiteralUtil::CreateR0<float>(42.0);
  auto tuple = LiteralUtil::MakeTuple({&elem});
  Call(&builder, callee, {ConstantLiteral(&builder, elem)});

  ComputeAndCompareTuple(&builder, tuple, {}, ErrorSpec(0.01f));
}

}  // namespace
}  // namespace xla
