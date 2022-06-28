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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSfloor_ceil_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSfloor_ceil_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSfloor_ceil_testDTcc() {
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

#include <limits>
#include <string>

#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class FloorCeilTest : public ClientLibraryTestBase {
 public:
  enum Function {
    kFloor,
    kCeil,
  };

  // Runs a computation and comparison on expected vs f(input)
  void TestR1F32(absl::Span<const float> input,
                 absl::Span<const float> expected, Function f) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSfloor_ceil_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/tests/floor_ceil_test.cc", "TestR1F32");

    LOG(INFO) << "input: {" << absl::StrJoin(expected, ", ") << "}";
    XlaBuilder builder(TestName());
    auto c = ConstantR1<float>(&builder, input);
    if (f == kCeil) {
      Ceil(c);
    } else {
      ASSERT_EQ(kFloor, f);
      Floor(c);
    }
    ComputeAndCompareR1<float>(&builder, expected, /*arguments=*/{});
  }

  void TestR0F32(float input, float expected, Function f) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSfloor_ceil_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/compiler/xla/tests/floor_ceil_test.cc", "TestR0F32");

    LOG(INFO) << "input: " << expected;
    XlaBuilder builder(TestName());
    auto c = ConstantR0<float>(&builder, input);
    if (f == kCeil) {
      Ceil(c);
    } else {
      ASSERT_EQ(kFloor, f);
      Floor(c);
    }
    ComputeAndCompareR0<float>(&builder, expected, /*arguments=*/{});
  }

  const ErrorSpec error_spec_{0.0001};

  float infinity_ = std::numeric_limits<float>::infinity();
  float minus_infinity_ = -std::numeric_limits<float>::infinity();
};

// Interesting notes:
// * if you pass snan the CPU doesn't canonicalize it to qnan.
// * passing x86-based CPU's qnan to the GPU makes a different nan
//   "7fc00000=nan=nan vs 7fffffff=nan=nan"

XLA_TEST_F(FloorCeilTest, R1S0Floor) { TestR1F32({}, {}, kFloor); }

TEST_F(FloorCeilTest, R1Floor) {
  TestR1F32({0.0, -0.0, infinity_, minus_infinity_, 1.1, -0.1},
            {0.0, -0.0, infinity_, minus_infinity_, 1.0, -1.0}, kFloor);
}

TEST_F(FloorCeilTest, R1Ceil) {
  TestR1F32({0.0, -0.0, infinity_, minus_infinity_, 1.1, -0.1},
            {0.0, -0.0, infinity_, minus_infinity_, 2.0, -0.0}, kCeil);
}

TEST_F(FloorCeilTest, R0Floor) {
  TestR0F32(0.0, 0.0, kFloor);
  TestR0F32(-0.0, -0.0, kFloor);
  TestR0F32(infinity_, infinity_, kFloor);
  TestR0F32(minus_infinity_, minus_infinity_, kFloor);
  TestR0F32(1.1, 1.0, kFloor);
  TestR0F32(-0.1, -1.0, kFloor);
}

TEST_F(FloorCeilTest, R0Ceil) {
  TestR0F32(0.0, 0.0, kCeil);
  TestR0F32(-0.0, -0.0, kCeil);
  TestR0F32(infinity_, infinity_, kCeil);
  TestR0F32(minus_infinity_, minus_infinity_, kCeil);
  TestR0F32(1.1, 2.0, kCeil);
  TestR0F32(-0.1, -0.0, kCeil);
}

}  // namespace
}  // namespace xla
