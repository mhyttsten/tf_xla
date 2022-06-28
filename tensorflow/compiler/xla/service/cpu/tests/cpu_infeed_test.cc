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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStestsPScpu_infeed_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStestsPScpu_infeed_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStestsPScpu_infeed_testDTcc() {
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

#include <unistd.h>

#include <memory>

#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class InfeedTest : public ClientLibraryTestBase {
 protected:
  // Transfers the given literal to the infeed interface of the device, and
  // check if the returned data from Infeed HLO is same as the literal.
  void TestInfeedRoundTrip(const Literal& literal) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStestsPScpu_infeed_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/xla/service/cpu/tests/cpu_infeed_test.cc", "TestInfeedRoundTrip");

    // TODO(b/31037751) Explicitly reset the Infeed state so that the
    // test is not affected by the state from the previous tests by
    // adding ClearInfeed if necessary when it is implemented. For now
    // don't use ResetDevice since it is not implemented on CPU.
    ASSERT_IS_OK(client_->TransferToInfeed(literal));
    XlaBuilder builder(TestName());
    Infeed(&builder, literal.shape());
    if (literal.shape().IsTuple()) {
      // TODO(b/30609564): Use ComputeAndCompareLiteral instead.
      ComputeAndCompareTuple(&builder, literal, {});
    } else {
      ComputeAndCompareLiteral(&builder, literal, {});
    }
  }
};

TEST_F(InfeedTest, SingleInfeedR0Bool) {
  TestInfeedRoundTrip(LiteralUtil::CreateR0<bool>(true));
}

TEST_F(InfeedTest, SingleInfeedR1U32) {
  TestInfeedRoundTrip(LiteralUtil::CreateR1<uint32_t>({1, 2, 3}));
}

TEST_F(InfeedTest, SingleInfeedR2F32) {
  TestInfeedRoundTrip(LiteralUtil::CreateR2F32Linspace(0.0, 1.0, 128, 64));
}

TEST_F(InfeedTest, SingleInfeedR3F32) {
  TestInfeedRoundTrip(
      LiteralUtil::CreateR3({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                             {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}}));
}

TEST_F(InfeedTest, SingleInfeedR3F32DifferentLayout) {
  const Layout r3_dim0minor = LayoutUtil::MakeLayout({0, 1, 2});
  const Layout r3_dim0major = LayoutUtil::MakeLayout({2, 1, 0});

  TestInfeedRoundTrip(LiteralUtil::CreateR3WithLayout(
      {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
       {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}},
      r3_dim0minor));

  TestInfeedRoundTrip(LiteralUtil::CreateR3WithLayout(
      {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
       {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}},
      r3_dim0major));
}

TEST_F(InfeedTest, SingleInfeedR4S32) {
  TestInfeedRoundTrip(LiteralUtil::CreateR4(
      {{{{1, -2}, {-4, 5}, {6, 7}}, {{8, 9}, {10, 11}, {12, 13}}},
       {{{10, 3}, {7, -2}, {3, 6}}, {{2, 5}, {-11, 5}, {-2, -5}}}}));
}

TEST_F(InfeedTest, SingleInfeedTuple) {
  TestInfeedRoundTrip(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<uint32_t>({1, 2, 3}),
       LiteralUtil::CreateR0<bool>(false)}));
}

TEST_F(InfeedTest, SingleInfeedEmptyTuple) {
  TestInfeedRoundTrip(LiteralUtil::MakeTuple({}));
}

// Tests Infeed operation used in a while loop, as in the code below. The
// computation is launched asynchronously, and then infeed data is transferred.
//
// float acc = 0.0f;
// while (acc < 40.0f) {
//   acc += reduce_add(Infeed());
// }
// return acc;
// TODO(b/30671675) enable this test once asynchronous execution is
// implemented for CPU.
TEST_F(InfeedTest, DISABLED_SingleInfeedInWhile) {
  XlaBuilder builder(TestName());
  const auto infeed_shape = ShapeUtil::MakeShape(F32, {3});
  const auto result_shape = ShapeUtil::MakeShape(F32, {});

  // Create a computation for the condition: repeat until (prev < 40.0f) holds.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    Gt(ConstantR0<float>(&builder, 40.0f), prev);
    condition = builder.Build().ConsumeValueOrDie();
  }
  // Create a computation for the body: add the reduced value of the Infeed
  // data to the result variable.
  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto infeed = Infeed(&builder, infeed_shape);
    auto addend = Reduce(infeed, ConstantR0<float>(&builder, 0.0f),
                         CreateScalarAddComputation(F32, &builder), {0});
    Add(prev, addend);
    body = builder.Build().ConsumeValueOrDie();
  }
  // Create a While node with computations for the condition and the body.
  auto init = ConstantR0<float>(&builder, 0.0f);
  While(condition, body, init);

  // Build and asynchronously launch the computation.
  auto computation = builder.Build().ConsumeValueOrDie();
  std::unique_ptr<GlobalData> result;
  tensorflow::Thread* computation_thread =
      tensorflow::Env::Default()->StartThread(
          tensorflow::ThreadOptions{}, "computation_thread", [&] {
            result = client_->Execute(computation, {}, &execution_options_)
                         .ValueOrDie();
          });

  // Send 5 Infeed data of shape F32[3].
  ASSERT_IS_OK(
      client_->TransferToInfeed(LiteralUtil::CreateR1<float>({1, 2, 3})));
  ASSERT_IS_OK(
      client_->TransferToInfeed(LiteralUtil::CreateR1<float>({4, 5, 6})));
  ASSERT_IS_OK(
      client_->TransferToInfeed(LiteralUtil::CreateR1<float>({7, 8, 9})));
  ASSERT_IS_OK(
      client_->TransferToInfeed(LiteralUtil::CreateR1<float>({10, 11, 12})));
  ASSERT_IS_OK(
      client_->TransferToInfeed(LiteralUtil::CreateR1<float>({13, 14, 15})));

  delete computation_thread;  // Joins the thread.
  auto result_literal = client_->Transfer(*result).ConsumeValueOrDie();

  // Only the first 3 infeed data should be added.
  LiteralTestUtil::ExpectR0Near<float>(45.0f, result_literal, ErrorSpec{1e-7});
}

// Tests two Infeed operations with a total order. The order is enforced by
// using the result of the first while loop as the initial value of the second
// while loop. The shapes of both Infeeds are Tuples, where the first tuple
// element (R1F32) is for the data to reduce and accumulate, and the second
// tuple element (PRED) to indicate whether the loop should continue. The
// computation is launched asynchronously, and then infeed data is transferred.
//
// float acc = 0.0f;
// continue = true;
// while (!continue) {
//   (data, continue) = Infeed(shape1);
//   acc += reduce_add(data)
// }
// continue = true;
// while(!continue) {
//   (data, continue) = Infeed(shape2);
//   acc += reduce_add(data)
// }
// return acc;
// TODO(b/30671675) enable this test once asynchronous execution is
// implemented for CPU.
TEST_F(InfeedTest, DISABLED_TwoInfeedsInTotalOrder) {
  XlaBuilder builder(TestName());
  const auto infeed1_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2}), ShapeUtil::MakeShape(PRED, {})});
  const auto infeed2_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3}), ShapeUtil::MakeShape(PRED, {})});
  const auto result_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(PRED, {})});

  // Create a computation for the condition: repeat until the second tuple
  // element is false.
  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    GetTupleElement(prev, 1);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // A lambda that builds the body computation of a while loop with the given
  // infeed shape, and returns the computation with the ownership.
  //
  // The body adds the reduced value of the Infeed data (first tuple element)
  // to the previous accumulator, and returns the accumulator and the continue
  // flag (second tuple element) as a tuple.
  const auto build_body = [&result_shape](const Shape& infeed_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStestsPScpu_infeed_testDTcc mht_1(mht_1_v, 395, "", "./tensorflow/compiler/xla/service/cpu/tests/cpu_infeed_test.cc", "lambda");

    XlaComputation body;
    XlaBuilder builder("body");
    auto prev = Parameter(&builder, 0, result_shape, "prev");
    auto infeed = Infeed(&builder, infeed_shape);
    auto addend =
        Reduce(GetTupleElement(infeed, 0), ConstantR0<float>(&builder, 0.0f),
               CreateScalarAddComputation(F32, &builder), {0});
    auto result = Add(GetTupleElement(prev, 0), addend);
    Tuple(&builder, {result, GetTupleElement(infeed, 1)});
    return builder.Build().ConsumeValueOrDie();
  };

  // Create the first while loop with infeed1_shape.
  auto init = Tuple(&builder, {ConstantR0<float>(&builder, 0.0f),
                               ConstantR0<bool>(&builder, true)});
  auto while1 = While(condition, build_body(infeed1_shape), init);
  auto result1 = Tuple(
      &builder, {GetTupleElement(while1, 0), ConstantR0<bool>(&builder, true)});

  // Create the second while loop with infeed2_shape. Note that the result from
  // the first while loop is used as the initial value.
  auto while2 = While(condition, build_body(infeed2_shape), result1);
  GetTupleElement(while2, 0);

  // Build the computation.
  auto computation = builder.Build().ConsumeValueOrDie();

  // Send the first 4 Infeed data of shape Tuple(F32[2], PRED).
  ASSERT_IS_OK(client_->TransferToInfeed(
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR1<float>({1, 2}),
                                        LiteralUtil::CreateR0<bool>(true)})));
  ASSERT_IS_OK(client_->TransferToInfeed(
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR1<float>({3, 4}),
                                        LiteralUtil::CreateR0<bool>(true)})));
  ASSERT_IS_OK(client_->TransferToInfeed(
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR1<float>({5, 6}),
                                        LiteralUtil::CreateR0<bool>(true)})));
  ASSERT_IS_OK(client_->TransferToInfeed(
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR1<float>({7, 8}),
                                        LiteralUtil::CreateR0<bool>(false)})));

  // Asynchronously launch the execution on the device.
  std::unique_ptr<GlobalData> result;
  tensorflow::Thread* computation_thread =
      tensorflow::Env::Default()->StartThread(
          tensorflow::ThreadOptions{}, "computation_thread", [&] {
            result = client_->Execute(computation, {}, &execution_options_)
                         .ValueOrDie();
          });

  // Wait for a second to ensure testing that the execution is waiting on the
  // Infeed data, and send the rest Infeed data of shape Tuple(F32[3], PRED).
  tensorflow::Env::Default()->SleepForMicroseconds(1000000);
  ASSERT_IS_OK(client_->TransferToInfeed(
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR1<float>({1, 2, 3}),
                                        LiteralUtil::CreateR0<bool>(true)})));
  ASSERT_IS_OK(client_->TransferToInfeed(
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR1<float>({7, 8, 9}),
                                        LiteralUtil::CreateR0<bool>(false)})));
  ASSERT_IS_OK(client_->TransferToInfeed(
      LiteralUtil::MakeTupleFromSlices({LiteralUtil::CreateR1<float>({4, 5, 6}),
                                        LiteralUtil::CreateR0<bool>(true)})));

  // Wait for the execution to be done, and transfer the result.
  delete computation_thread;  // Joins the thread.
  auto result_literal = client_->Transfer(*result).ConsumeValueOrDie();

  // Only the first 6 infeed data should be added.
  LiteralTestUtil::ExpectR0Near<float>(66.0f, result_literal, ErrorSpec{1e-7});
}

}  // namespace
}  // namespace xla
