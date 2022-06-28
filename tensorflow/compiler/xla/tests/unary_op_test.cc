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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc() {
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
#include <vector>

#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class UnaryOpTest : public ClientLibraryTestBase {
 protected:
  template <typename T>
  T inf() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/tests/unary_op_test.cc", "inf");

    return std::numeric_limits<T>::infinity();
  }
  template <typename T>
  void AbsSize0TestHelper() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/xla/tests/unary_op_test.cc", "AbsSize0TestHelper");

    XlaBuilder builder(TestName());
    auto arg = ConstantR1<T>(&builder, {});
    Abs(arg);

    if (primitive_util::NativeToPrimitiveType<T>() == C64) {
      ComputeAndCompareR1<float>(&builder, {}, {});
    } else {
      ComputeAndCompareR1<T>(&builder, {}, {});
    }
  }

  template <typename T>
  void AbsTestHelper() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc mht_2(mht_2_v, 226, "", "./tensorflow/compiler/xla/tests/unary_op_test.cc", "AbsTestHelper");

    XlaBuilder builder(TestName());
    auto arg = ConstantR1<T>(&builder, {-2, 25, 0, -123, inf<T>(), -inf<T>()});
    Abs(arg);

    ComputeAndCompareR1<T>(&builder, {2, 25, 0, 123, inf<T>(), inf<T>()}, {});
  }

  template <typename T>
  void SignTestHelper() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc mht_3(mht_3_v, 238, "", "./tensorflow/compiler/xla/tests/unary_op_test.cc", "SignTestHelper");

    XlaBuilder builder(TestName());
    auto arg = ConstantR1<T>(
        &builder, {-2, 25, 0, static_cast<T>(-0.0), -123, inf<T>(), -inf<T>()});
    Sign(arg);

    ComputeAndCompareR1<T>(
        &builder,
        {-1, 1, static_cast<T>(+0.0), static_cast<T>(-0.0), -1, 1, -1}, {});
  }

  template <typename T>
  void SignAbsTestHelper() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc mht_4(mht_4_v, 253, "", "./tensorflow/compiler/xla/tests/unary_op_test.cc", "SignAbsTestHelper");

    XlaBuilder builder(TestName());
    auto arg = ConstantR1<T>(&builder, {-2, 25, 0, -123});
    auto sign = Sign(arg);
    auto abs = Abs(arg);
    Sub(Mul(sign, abs), arg);

    ComputeAndCompareR1<T>(&builder, {0, 0, 0, 0}, {});
  }
};

template <>
int UnaryOpTest::inf<int>() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc mht_5(mht_5_v, 268, "", "./tensorflow/compiler/xla/tests/unary_op_test.cc", "UnaryOpTest::inf<int>");

  return 2147483647;
}

template <>
int64_t UnaryOpTest::inf<int64_t>() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc mht_6(mht_6_v, 276, "", "./tensorflow/compiler/xla/tests/unary_op_test.cc", "UnaryOpTest::inf<int64_t>");

  return 0x7FFFFFFFFFFFFFFFl;
}

template <>
void UnaryOpTest::AbsTestHelper<complex64>() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc mht_7(mht_7_v, 284, "", "./tensorflow/compiler/xla/tests/unary_op_test.cc", "UnaryOpTest::AbsTestHelper<complex64>");

  XlaBuilder builder(TestName());
  auto arg = ConstantR1<complex64>(&builder, {{-2, 0},
                                              {0, 25},
                                              {0, 0},
                                              {-0.3f, 0.4f},
                                              {0, inf<float>()},
                                              {-inf<float>(), 0}});
  Abs(arg);

  Literal expected =
      LiteralUtil::CreateR1<float>({2, 25, 0, 0.5, inf<float>(), inf<float>()});
  ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-6f));
}

template <>
void UnaryOpTest::SignTestHelper<complex64>() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc mht_8(mht_8_v, 303, "", "./tensorflow/compiler/xla/tests/unary_op_test.cc", "UnaryOpTest::SignTestHelper<complex64>");

  XlaBuilder builder(TestName());
  auto arg = ConstantR1<complex64>(
      &builder,
      {{-2, 0}, {0, 25}, {0, 0}, {static_cast<float>(-0.0), 0}, {-1, 1}});
  Sign(arg);

  Literal expected = LiteralUtil::CreateR1<complex64>(
      {{-1, 0}, {0, 1}, {0, 0}, {0, 0}, {-std::sqrt(0.5f), std::sqrt(0.5f)}});
  ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-6f));
}

template <>
void UnaryOpTest::SignAbsTestHelper<complex64>() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSunary_op_testDTcc mht_9(mht_9_v, 319, "", "./tensorflow/compiler/xla/tests/unary_op_test.cc", "UnaryOpTest::SignAbsTestHelper<complex64>");

  XlaBuilder builder(TestName());
  auto arg =
      ConstantR1<complex64>(&builder, {{-2, 0}, {0, 25}, {0, 0}, {-0.4, 0.3}});
  auto sign = Sign(arg);
  auto abs = Abs(arg);
  Sub(Mul(sign, ConvertElementType(abs, C64)), arg);

  Literal expected = LiteralUtil::CreateR1<complex64>({0, 0, 0, 0});
  ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-6f));
}

XLA_TEST_F(UnaryOpTest, AbsTestR1Size0) {
  AbsSize0TestHelper<int>();
  AbsSize0TestHelper<float>();
  AbsSize0TestHelper<complex64>();
}

XLA_TEST_F(UnaryOpTest, AbsTestR1) {
  AbsTestHelper<int>();
  AbsTestHelper<float>();
  AbsTestHelper<complex64>();
}

XLA_TEST_F(UnaryOpTest, AbsTestR0) {
  XlaBuilder builder(TestName());
  auto argi = ConstantR0<int>(&builder, -5);
  auto absi = Abs(argi);
  auto argf = ConstantR0<float>(&builder, -3.0f);
  auto absf = Abs(argf);
  auto argf0 = ConstantR0<float>(&builder, -0.0f);
  auto absf0 = Abs(argf0);
  auto argc = ConstantR0<complex64>(&builder, {-0.3f, 0.4f});
  auto absc = Abs(argc);
  Add(Add(absc, absf0), Add(absf, ConvertElementType(absi, F32)));

  ComputeAndCompareR0<float>(&builder, 8.5f, {});
}

XLA_TEST_F(UnaryOpTest, SignTestR0) {
  XlaBuilder builder(TestName());
  auto argi = ConstantR0<int>(&builder, -5);
  auto sgni = Sign(argi);  // -1
  auto argf = ConstantR0<float>(&builder, -4.0f);
  auto sgnf = Sign(argf);  // -1
  auto argf0 = ConstantR0<float>(&builder, -0.0f);
  auto sgnf0 = Sign(argf0);  // 0
  auto argc = ConstantR0<complex64>(&builder, {-.3, .4});
  auto sgnc = Sign(argc);  // (-.6, .8)
  Add(sgnc, ConvertElementType(
                Add(Add(sgnf0, sgnf), ConvertElementType(sgni, F32)), C64));

  Literal expected = LiteralUtil::CreateR0<complex64>({-2.6f, 0.8f});
  ComputeAndCompareLiteral(&builder, expected, {}, ErrorSpec(1e-6f));
}

XLA_TEST_F(UnaryOpTest, SignTestR1) {
  SignTestHelper<int>();
  SignTestHelper<int64_t>();
  SignTestHelper<float>();
  SignTestHelper<complex64>();
}

XLA_TEST_F(UnaryOpTest, SignAbsTestR1) {
  SignAbsTestHelper<int>();
  SignAbsTestHelper<float>();
  SignAbsTestHelper<complex64>();
}

XLA_TEST_F(UnaryOpTest, SignAbsTestR2) {
  XlaBuilder builder(TestName());
  auto arg = ConstantR2<float>(&builder, {{1.0, -2.0}, {-3.0, 4.0}});
  auto sign = Sign(arg);
  auto abs = Abs(arg);
  Sub(Mul(sign, abs), arg);

  ComputeAndCompareR2<float>(&builder, {{0, 0}, {0, 0}}, {});
}

XLA_TEST_F(UnaryOpTest, ConvertElementTypePredToS32) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<int32_t>(&builder, {0, 1});
  auto rhs = ConstantR1<int32_t>(&builder, {1, 1});
  ConvertElementType(Eq(lhs, rhs), S32);

  ComputeAndCompareR1<int32_t>(&builder, {0, 1}, {});
}

XLA_TEST_F(UnaryOpTest, ConvertElementTypePredToF32) {
  XlaBuilder builder(TestName());
  auto lhs = ConstantR1<int32_t>(&builder, {0, 1});
  auto rhs = ConstantR1<int32_t>(&builder, {1, 1});
  ConvertElementType(Eq(lhs, rhs), F32);

  ComputeAndCompareR1<float>(&builder, {0.0, 1.0}, {});
}

}  // namespace
}  // namespace xla
