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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc() {
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

#include <cmath>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

// Tests the handling of the basic mathematics operations with F16 operands.

namespace xla {
namespace {

class HalfTestBase : public ClientLibraryTestBase {
 protected:
  const ErrorSpec error_spec_{0.001, 0.001};
  // Number of elements in the input buffers.
  static constexpr int kNumElements = 4;
};

using UnaryBuildFuncTy = std::function<void(const xla::XlaOp& src)>;

struct UnaryOpTestParam {
  std::function<half(half)> compute_func;
  UnaryBuildFuncTy build_func;
};

class UnaryOpTest : public HalfTestBase,
                    public ::testing::WithParamInterface<UnaryOpTestParam> {};

XLA_TEST_P(UnaryOpTest, Ops) {
  std::vector<half> x({half(1.4), half(-2.3), half(3.2), half(-4.1), half(9.0),
                       half(42.0), half(-9.0), half(-100.0)});
  XlaBuilder builder(TestName());
  XlaOp x_opnd;
  auto x_data = CreateR1Parameter<half>(x, /*parameter_number=*/0, "x",
                                        &builder, &x_opnd);

  std::function<half(half)> compute_func = GetParam().compute_func;
  std::vector<half> expected;
  const int64_t n = x.size();
  expected.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    expected.push_back(compute_func(x[i]));
  }

  UnaryBuildFuncTy build_func = GetParam().build_func;
  build_func(x_opnd);

  ComputeAndCompareR1<half>(&builder, expected, {x_data.get()}, error_spec_);
}

half sign_imp(half value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_0(mht_0_v, 241, "", "./tensorflow/compiler/xla/tests/half_test.cc", "sign_imp");

  const float x(std::move(value));
  return half((x < .0) ? -1 : (x > .0));
}

half round_imp(half value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_1(mht_1_v, 249, "", "./tensorflow/compiler/xla/tests/half_test.cc", "round_imp");

  return half(std::round(static_cast<float>(std::move(value))));
}

INSTANTIATE_TEST_CASE_P(
    half, UnaryOpTest,
    ::testing::Values(
        UnaryOpTestParam{[](half x) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_2(mht_2_v, 259, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return abs(x); }, &Abs},
        UnaryOpTestParam{[](half x) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_3(mht_3_v, 263, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return round_imp(x); }, &Round},
        UnaryOpTestParam{[](half x) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_4(mht_4_v, 267, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return ceil(x); }, &Ceil},
        UnaryOpTestParam{[](half x) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_5(mht_5_v, 271, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return cos(x); }, &Cos},
        UnaryOpTestParam{[](half x) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_6(mht_6_v, 275, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return exp(x); }, &Exp},
        UnaryOpTestParam{[](half x) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_7(mht_7_v, 279, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return floor(x); }, &Floor},
        UnaryOpTestParam{[](half x) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_8(mht_8_v, 283, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return log(x); }, &Log},
        UnaryOpTestParam{[](half x) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_9(mht_9_v, 287, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return -x; }, &Neg},
        UnaryOpTestParam{[](half x) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_10(mht_10_v, 291, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return sign_imp(x); }, &Sign},
        UnaryOpTestParam{[](half x) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_11(mht_11_v, 295, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return sin(x); }, &Sin},
        UnaryOpTestParam{[](half x) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_12(mht_12_v, 299, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return tanh(x); }, &Tanh}

        ));

struct UnaryPredTestParam {
  std::function<bool(half)> compute_func;
  UnaryBuildFuncTy build_func;
};

class UnaryPredTest : public HalfTestBase,
                      public ::testing::WithParamInterface<UnaryPredTestParam> {
};

XLA_TEST_P(UnaryPredTest, Ops) {
  std::vector<half> x({half(1.4), half(-2.3), half(3.2), half(-4.1)});
  XlaBuilder builder(TestName());
  XlaOp x_opnd;
  auto x_data = CreateR1Parameter<half>(x, /*parameter_number=*/0, "x",
                                        &builder, &x_opnd);

  std::function<bool(half)> compute_func = GetParam().compute_func;
  CHECK_EQ(kNumElements, x.size());
  bool expected[kNumElements];
  for (int64_t i = 0; i < x.size(); ++i) {
    expected[i] = compute_func(x[i]);
  }

  UnaryBuildFuncTy build_func = GetParam().build_func;
  build_func(x_opnd);

  ComputeAndCompareR1<bool>(&builder, expected, {x_data.get()});
}

INSTANTIATE_TEST_CASE_P(half, UnaryPredTest,
                        ::testing::Values(UnaryPredTestParam{
                            [](half x) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_13(mht_13_v, 337, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return isfinite(x); }, &IsFinite}));

using BinaryBuildFuncTy = std::function<void(
    const xla::XlaOp& x, const xla::XlaOp& y, absl::Span<const int64_t>)>;

struct BinaryOpTestParam {
  std::function<half(half, half)> compute_func;
  BinaryBuildFuncTy build_func;
};

class BinaryOpTest : public HalfTestBase,
                     public ::testing::WithParamInterface<BinaryOpTestParam> {};

XLA_TEST_P(BinaryOpTest, Ops) {
  std::vector<half> x({half(1.0), half(2.0), half(3.0), half(-4.0)});
  std::vector<half> y({half(0.4), half(-0.3), half(0.2), half(0.1)});
  XlaBuilder builder(TestName());
  XlaOp x_opnd;
  auto x_data = CreateR1Parameter<half>(x, /*parameter_number=*/0, "x",
                                        &builder, &x_opnd);

  XlaOp y_opnd;
  auto y_data = CreateR1Parameter<half>(y, /*parameter_number=*/1, "y",
                                        &builder, &y_opnd);

  std::function<half(half, half)> compute_func = GetParam().compute_func;
  std::vector<half> expected;
  const int64_t n = x.size();
  expected.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    expected.push_back(compute_func(x[i], y[i]));
  }

  BinaryBuildFuncTy build_func = GetParam().build_func;
  build_func(x_opnd, y_opnd, {});

  ComputeAndCompareR1<half>(&builder, expected, {x_data.get(), y_data.get()},
                            error_spec_);
}

half atan2_imp(half x, half y) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_14(mht_14_v, 380, "", "./tensorflow/compiler/xla/tests/half_test.cc", "atan2_imp");

  return half(std::atan2(static_cast<float>(std::move(x)),
                         static_cast<float>(std::move(y))));
}

INSTANTIATE_TEST_CASE_P(
    half, BinaryOpTest,
    ::testing::Values(
        BinaryOpTestParam{[](half x, half y) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_15(mht_15_v, 391, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return x + y; }, &Add},
        BinaryOpTestParam{[](half x, half y) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_16(mht_16_v, 395, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return atan2_imp(x, y); },
                          &Atan2},
        BinaryOpTestParam{[](half x, half y) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_17(mht_17_v, 400, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return x / y; }, &Div},
        BinaryOpTestParam{[](half x, half y) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_18(mht_18_v, 404, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return max(x, y); }, &Max},
        BinaryOpTestParam{[](half x, half y) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_19(mht_19_v, 408, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return min(x, y); }, &Min},
        BinaryOpTestParam{[](half x, half y) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_20(mht_20_v, 412, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return x * y; }, &Mul},
        BinaryOpTestParam{[](half x, half y) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_21(mht_21_v, 416, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return pow(x, y); }, &Pow},
        BinaryOpTestParam{[](half x, half y) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_22(mht_22_v, 420, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return x - y; }, &Sub}

        ));

struct BinaryPredTestParam {
  std::function<bool(half, half)> compute_func;
  BinaryBuildFuncTy build_func;
};

class BinaryPredTest
    : public HalfTestBase,
      public ::testing::WithParamInterface<BinaryPredTestParam> {};

XLA_TEST_P(BinaryPredTest, Ops) {
  std::vector<half> x({half(1.0), half(2.0), half(0.2), half(-4.0)});
  std::vector<half> y({half(0.4), half(-0.3), half(0.2), half(0.1)});
  XlaBuilder builder(TestName());
  XlaOp x_opnd;
  auto x_data = CreateR1Parameter<half>(x, /*parameter_number=*/0, "x",
                                        &builder, &x_opnd);

  XlaOp y_opnd;
  auto y_data = CreateR1Parameter<half>(y, /*parameter_number=*/1, "y",
                                        &builder, &y_opnd);

  std::function<bool(half, half)> compute_func = GetParam().compute_func;
  CHECK_EQ(kNumElements, x.size());
  bool expected[kNumElements];
  for (int64_t i = 0; i < x.size(); ++i) {
    expected[i] = compute_func(x[i], y[i]);
  }

  BinaryBuildFuncTy build_func = GetParam().build_func;
  build_func(x_opnd, y_opnd, {});

  ComputeAndCompareR1<bool>(&builder, expected, {x_data.get(), y_data.get()});
}

INSTANTIATE_TEST_CASE_P(
    half, BinaryPredTest,
    ::testing::Values(
        BinaryPredTestParam{[](half x, half y) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_23(mht_23_v, 464, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return x == y; }, &Eq},
        BinaryPredTestParam{[](half x, half y) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_24(mht_24_v, 468, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return x != y; }, &Ne},
        BinaryPredTestParam{[](half x, half y) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_25(mht_25_v, 472, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return x >= y; }, &Ge},
        BinaryPredTestParam{[](half x, half y) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_26(mht_26_v, 476, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return x > y; }, &Gt},
        BinaryPredTestParam{[](half x, half y) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_27(mht_27_v, 480, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return x <= y; }, &Le},
        BinaryPredTestParam{[](half x, half y) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShalf_testDTcc mht_28(mht_28_v, 484, "", "./tensorflow/compiler/xla/tests/half_test.cc", "lambda");
 return x < y; }, &Lt}

        ));

}  // namespace
}  // namespace xla
