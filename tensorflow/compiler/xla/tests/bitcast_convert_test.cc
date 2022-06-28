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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbitcast_convert_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbitcast_convert_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbitcast_convert_testDTcc() {
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

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class BitcastConvertTest : public ClientLibraryTestBase {
 public:
  explicit BitcastConvertTest(se::Platform* platform = nullptr)
      : ClientLibraryTestBase(platform) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbitcast_convert_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/tests/bitcast_convert_test.cc", "BitcastConvertTest");

    mutable_debug_options()->add_xla_disable_hlo_passes("algsimp");
    mutable_debug_options()->add_xla_disable_hlo_passes("inline");
  }
};

TEST_F(BitcastConvertTest, ConvertR1S32ToR1S32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {42, 64});
  BitcastConvertType(a, S32);

  std::vector<int32_t> expected = {42, 64};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertR1F32ToR1F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.0f, 64.0f});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, BitcastR1S32ToR1F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder,
                               {0, static_cast<int32_t>(0x80000000), 0x3F800000,
                                static_cast<int32_t>(0xBF800000), 0x3F000000,
                                static_cast<int32_t>(0xBF000000)});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(BitcastConvertTest, ConvertR1S0S32ToR1S0F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertR1F32ToR1S32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.6, 64.4});
  BitcastConvertType(a, S32);

  std::vector<int32_t> expected = {0x422a6666, 0x4280cccd};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertS32Extremes) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32_t>(&builder, {std::numeric_limits<int32_t>::min(),
                                          std::numeric_limits<int32_t>::max()});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {-0.0f, NAN};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0, 0));
}

TEST_F(BitcastConvertTest, ConvertMapToS32) {
  XlaBuilder builder(TestName());
  auto b = builder.CreateSubBuilder("convert");
  auto param = Parameter(b.get(), 0, ShapeUtil::MakeShape(F32, {}), "in");
  BitcastConvertType(param, S32);
  auto a = ConstantR1<float>(&builder, {42.0f, 64.0f});
  Map(&builder, {a}, b->BuildAndNoteError(), {0});

  std::vector<int32_t> expected = {0x42280000, 0x42800000};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertMapToF32) {
  XlaBuilder builder(TestName());
  auto b = builder.CreateSubBuilder("convert");
  auto param = Parameter(b.get(), 0, ShapeUtil::MakeShape(S32, {}), "in");
  BitcastConvertType(param, F32);
  auto a = ConstantR1<int32_t>(&builder, {0x42280000, 0x42800000});
  Map(&builder, {a}, b->BuildAndNoteError(), {0});

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

// Regression test for b/31758660. When ReshapeMover transforms
//   input -> reshape -> convert
// to
//   input -> convert -> reshape
// the new convert should have the same element type as the old convert.
TEST_F(BitcastConvertTest, ConvertReshape) {
  XlaBuilder builder(TestName());
  auto input = ConstantR1<int32_t>(&builder, {0x42280000});
  auto reshape = Reshape(input, /*dimensions=*/{0}, /*new_sizes=*/{});
  BitcastConvertType(reshape, F32);

  ComputeAndCompareR0<float>(&builder, 42.0f, {});
}

class BitcastConvertHloTest : public HloTestBase {};

XLA_TEST_F(BitcastConvertHloTest, S32to4S8) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_smaller

ENTRY main {
  p = s32[10] parameter(0)
  ROOT out = s8[10,4] bitcast-convert(p)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{0, 0}));
}

XLA_TEST_F(BitcastConvertHloTest, FourS8toS32) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_larger

ENTRY main {
  p = s8[10,4] parameter(0)
  ROOT out = s32[10] bitcast-convert(p)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{0, 0}));
}

XLA_TEST_F(BitcastConvertHloTest, F32to2F16) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_smaller

ENTRY main {
  p = f32[10] parameter(0)
  ROOT out = f16[10,2] bitcast-convert(p)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

XLA_TEST_F(BitcastConvertHloTest, TwoF16toF32) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_smaller

ENTRY main {
  p = f16[10,2] parameter(0)
  ROOT out = f32[10] bitcast-convert(p)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla
