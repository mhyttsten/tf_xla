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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSinfeed_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSinfeed_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSinfeed_testDTcc() {
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
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace {

class InfeedTest : public ClientLibraryTestBase {
 protected:
  // Transfers the given literal to the infeed interface of the device, and
  // check if the returned data from Infeed HLO is same as the literal.
  void TestInfeedRoundTrip(const Literal& literal) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSinfeed_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/xla/service/gpu/tests/infeed_test.cc", "TestInfeedRoundTrip");

    // TODO(b/30481585) Explicitly reset the Infeed state so that the
    // test is not affected by the state from the previous tests.
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

// Tests that a large infeed can be handled.
TEST_F(InfeedTest, LargeInfeed) {
  Array4D<float> array(80, 100, 8, 128);
  array.FillIota(1.0f);
  TestInfeedRoundTrip(LiteralUtil::CreateR4FromArray4D<float>(array));
}

TEST_F(InfeedTest, SingleInfeedTuple) {
  TestInfeedRoundTrip(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<uint32_t>({1, 2, 3}),
       LiteralUtil::CreateR0<bool>(false)}));
}

TEST_F(InfeedTest, SingleInfeedEmptyTuple) {
  TestInfeedRoundTrip(LiteralUtil::MakeTuple({}));
}

// Tests that a large tuple infeed can be handled.
TEST_F(InfeedTest, SingleInfeedLargeTuple) {
  Array4D<float> array(40, 100, 8, 128);
  array.FillIota(1.0f);
  TestInfeedRoundTrip(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR4FromArray4D<float>(array),
       LiteralUtil::CreateR0<int32_t>(5)}));
}

class BlockingInfeedTest : public ClientLibraryTestBase {};

TEST_F(BlockingInfeedTest, TestNoOoms) {
  Array3D<float> array(1024, 1024, 64);
  array.FillIota(1.0f);
  auto literal = LiteralUtil::CreateR3FromArray3D<float>(array);

  int64_t kMemoryPressure = 32ul * 1024 * 1024 * 1024;
  int64_t infeed_count =
      kMemoryPressure / (array.num_elements() * sizeof(float));

  auto transfer_infeeds = [&] {
    for (int i = 0; i < infeed_count; i++) {
      ASSERT_IS_OK(client_->TransferToInfeed(literal));
    }
  };

  auto* env = tensorflow::Env::Default();

  std::unique_ptr<tensorflow::Thread> thread{env->StartThread(
      tensorflow::ThreadOptions{}, "transfer_infeeds", transfer_infeeds)};

  // Sleep for 30s waiting for the infeed thread to "catch up".
  //
  // Without the fix accompanying this test, transfer_infeeds causes an OOM if
  // the consumer (XLA computation running on the main thread) is unable to keep
  // up with the producer (the transfer_infeeds thread).  When that happens, the
  // GPU buffers from the producer pile up and consume all of GPU memory.
  //
  // To reliably reproduce the issue we need to slow down the consumer, and we
  // do that by inserting a sleep here.
  //
  // The fix is to back TransferToInfeed by a blocking queue that does not allow
  // more than kMaxInfeedsInFlight infeeds in flight.
  env->SleepForMicroseconds(30ul * 1000 * 1000);

  XlaBuilder builder(TestName());
  for (int i = 0; i < infeed_count; i++) {
    Infeed(&builder, literal.shape());
  }

  ComputeAndCompareLiteral(&builder, literal, {});
}

}  // namespace
}  // namespace xla
