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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSvector_ops_simple_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSvector_ops_simple_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSvector_ops_simple_testDTcc() {
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

#include <cmath>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class VecOpsSimpleTest : public ClientLibraryTestBase {
 public:
  explicit VecOpsSimpleTest(se::Platform* platform = nullptr)
      : ClientLibraryTestBase(platform) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSvector_ops_simple_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/tests/vector_ops_simple_test.cc", "VecOpsSimpleTest");

    mutable_debug_options()->add_xla_disable_hlo_passes("algsimp");
    mutable_debug_options()->add_xla_disable_hlo_passes("inline");
  }

  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(VecOpsSimpleTest, ExpTenValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  Exp(x);

  std::vector<float> expected = {8.1662,     7.4274e-02, 13.4637,    1.8316e-02,
                                 8.1662,     9.9742,     6.7379e-03, 4.0657e-01,
                                 9.0718e-02, 4.9530};

  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, ExpManyValues) {
  for (int count : {63, 64, 65, 127, 128, 129, 17 * 4096}) {
    XlaBuilder builder(TestName());
    std::vector<float> exponents;
    exponents.reserve(count);
    for (int i = 0; i < count; ++i) {
      exponents.push_back(i / static_cast<float>(count));
    }
    auto x = ConstantR1<float>(&builder, exponents);
    Exp(x);

    std::vector<float> expected;
    expected.reserve(exponents.size());
    for (float exponent : exponents) {
      expected.push_back(std::exp(exponent));
    }

    ComputeAndCompareR1<float>(&builder, expected, {},
                               ErrorSpec(/*aabs=*/1e-2, /*arel=*/1e-3));
  }
}

XLA_TEST_F(VecOpsSimpleTest, ExpIn4D) {
  XlaBuilder builder(TestName());
  Array4D<float> exponents(2, 2, 2, 2);

  std::vector<float> exponents_vector;
  std::vector<float> expected_vector;
  const auto num_elements = exponents.num_elements();
  exponents_vector.reserve(num_elements);
  expected_vector.reserve(num_elements);
  for (int i = 0; i < exponents.num_elements(); ++i) {
    exponents_vector.push_back(static_cast<float>(i) /
                               exponents.num_elements());
    expected_vector.push_back(std::exp(exponents_vector.back()));
  }
  exponents.SetValues(exponents_vector);

  Array4D<float> expected(2, 2, 2, 2, expected_vector);

  auto x = ConstantR4FromArray4D<float>(&builder, exponents);
  Exp(x);

  ComputeAndCompareR4<float>(&builder, expected, {},
                             ErrorSpec(/*aabs=*/1e-2, /*arel=*/1e-3));
}

XLA_TEST_F(VecOpsSimpleTest, NegateTenFloatValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  Neg(x);

  std::vector<float> expected = {-2.1, 2.6, -2.6, 4.0, -2.1,
                                 -2.3, 5.0, 0.9,  2.4, -1.6};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, NegateTenInt32Values) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<int32_t>(&builder, {2, -2, 12, -4, 5, 20, -15, 0, -2, 1});
  Neg(x);

  std::vector<int> expected = {-2, 2, -12, 4, -5, -20, 15, 0, 2, -1};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, NegateUint32Values) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<uint32_t>(&builder, {0, 1, 42, static_cast<uint32_t>(-1),
                                           static_cast<uint32_t>(-12)});
  Neg(x);
  std::vector<uint32_t> expected = {0, static_cast<uint32_t>(-1),
                                    static_cast<uint32_t>(-42), 1, 12};
  ComputeAndCompareR1<uint32_t>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, InvSqrtSevenValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder,
                             {16.0, 1.0, 1024.0, 0.16, 0.2, 12345, 1.2345});
  Pow(x, ConstantR0<float>(&builder, -.5f));

  std::vector<float> expected = {.25,     1,       .03125, 2.5,
                                 2.23607, .009000, .900025};

  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, AddTenValuesViaMap) {
  XlaBuilder builder(TestName());
  auto add = CreateScalarAddComputation(F32, &builder);

  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = ConstantR1<float>(
      &builder, {-0.4, -0.6, -3.0, 0.2, 3.8, -2.2, -1.8, 4.9, 1.4, 0.6});
  Map(&builder, {x, y}, add, {0});

  std::vector<float> expected = {1.7, -3.2, -0.4, -3.8, 5.9,
                                 0.1, -6.8, 4.,   -1.,  2.2};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, MaxTenValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = ConstantR1<float>(
      &builder, {-0.4, -0.6, -3.0, 0.2, 3.8, -2.2, -1.8, 4.9, 1.4, 0.6});
  Max(x, y);

  std::vector<float> expected = {2.1, -0.6, 2.6, 0.2, 3.8,
                                 2.3, -1.8, 4.9, 1.4, 1.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MaxTenValuesFromParams) {
  // Similar to MaxTenValues, except that the inputs come from params rather
  // than constants.
  XlaBuilder builder(TestName());
  XlaOp v1, v2;
  std::unique_ptr<GlobalData> param0_data = CreateR1Parameter<float>(
      {41.0f, 2.0f, 3.0f, 84.0f}, /*parameter_number=*/0, /*name=*/"v1",
      /*builder=*/&builder, /*data_handle=*/&v1);
  std::unique_ptr<GlobalData> param1_data = CreateR1Parameter<float>(
      {21.0f, 22.0f, 23.0f, 24.0f}, /*parameter_number=*/1, /*name=*/"v2",
      /*builder=*/&builder, /*data_handle=*/&v2);

  Max(v1, v2);
  ComputeAndCompareR1<float>(&builder, {41.0f, 22.0f, 23.0f, 84.0f},
                             {param0_data.get(), param1_data.get()},
                             error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, Max15000ValuesFromParams) {
  // Similar to MaxTenValuesFromParams, except that the data size passed in and
  // out is large.
  XlaBuilder builder(TestName());

  // Number of floats in the data passed into and out of the computation.
  constexpr int datalen = 15 * 1000;

  // The inputs are initialized with a special pattern where in the first third
  // of the data v1[i] > v2[i] and elsewhere it's vice versa.
  std::vector<float> v1vec;
  std::vector<float> v2vec;
  std::vector<float> expected_vec;
  v1vec.reserve(datalen);
  v2vec.reserve(datalen);
  expected_vec.reserve(datalen);
  for (int i = 0; i < datalen; ++i) {
    float smaller = i;
    float larger = i * 2;
    if (i < datalen / 3) {
      v1vec.push_back(larger);
      v2vec.push_back(smaller);
    } else {
      v1vec.push_back(smaller);
      v2vec.push_back(larger);
    }
    expected_vec.push_back(larger);
  }

  XlaOp v1, v2;
  std::unique_ptr<GlobalData> param0_data =
      CreateR1Parameter<float>(v1vec, /*parameter_number=*/0, /*name=*/"v1",
                               /*builder=*/&builder, /*data_handle=*/&v1);
  std::unique_ptr<GlobalData> param1_data =
      CreateR1Parameter<float>(v2vec, /*parameter_number=*/1, /*name=*/"v2",
                               /*builder=*/&builder, /*data_handle=*/&v2);

  Max(v1, v2);
  ComputeAndCompareR1<float>(&builder, expected_vec,
                             {param0_data.get(), param1_data.get()},
                             error_spec_);
}

XLA_TEST_F(VecOpsSimpleTest, MaxTenValuesWithScalar) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = ConstantR0<float>(&builder, 0);
  Max(x, y);

  std::vector<float> expected = {2.1, 0.0, 2.6, 0.0, 2.1,
                                 2.3, 0.0, 0.0, 0.0, 1.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MinTenValues) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto y = ConstantR1<float>(
      &builder, {-0.4, -0.6, -3.0, 0.2, 3.8, -2.2, -1.8, 4.9, 1.4, 0.6});
  Min(x, y);

  std::vector<float> expected = {-0.4, -2.6, -3.0, -4.0, 2.1,
                                 -2.2, -5.0, -0.9, -2.4, 0.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MinMaxTenValues) {
  XlaBuilder builder(TestName());
  auto zero = ConstantR0<float>(&builder, 0);
  auto one = ConstantR0<float>(&builder, 1);
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, 0.3, 3.1, 0.9, -5.0, 0.1, -2.4, 0.6});
  Min(Max(x, zero), one);

  std::vector<float> expected = {1.0, 0.0, 1.0, 0.3, 1.0,
                                 0.9, 0.0, 0.1, 0.0, 0.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampTenValuesConstant) {
  XlaBuilder builder(TestName());
  auto zero = ConstantR0<float>(&builder, 0);
  auto one = ConstantR0<float>(&builder, 1);
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, 0.3, 3.1, 0.9, -5.0, 0.1, -2.4, 0.6});
  Clamp(zero, x, one);

  std::vector<float> expected = {1.0, 0.0, 1.0, 0.3, 1.0,
                                 0.9, 0.0, 0.1, 0.0, 0.6};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampTwoValuesConstant) {
  XlaBuilder builder(TestName());
  auto zero = ConstantR1<float>(&builder, {0.0f, 0.0f});
  auto one = ConstantR1<float>(&builder, {1.0f, 1.0f});
  auto x = ConstantR1<float>(&builder, {2.1, -2.6});
  Clamp(zero, x, one);

  std::vector<float> expected = {1.0, 0.0};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampTenValuesConstantNonzeroLower) {
  XlaBuilder builder(TestName());
  auto one = ConstantR0<float>(&builder, 1);
  auto two = ConstantR0<float>(&builder, 2);
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, 0.3, 3.1, 0.9, -5.0, 0.1, -2.4, 0.6});
  Clamp(one, x, two);

  std::vector<float> expected = {2.0, 1.0, 2.0, 1.0, 2.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampFloatEdgeCases) {
  XlaBuilder builder(TestName());
  SetFastMathDisabled(true);
  auto low = ConstantR1<float>(&builder, {NAN, 1, 1});
  auto high = ConstantR1<float>(&builder, {3, NAN, 3});
  auto x = ConstantR1<float>(&builder, {2, 2, NAN});
  Clamp(low, x, high);

  std::vector<float> expected = {NAN, NAN, NAN};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, ClampValuesConstantS64) {
  XlaBuilder builder(TestName());
  auto zero = ConstantR0<int64_t>(&builder, 0);
  auto one = ConstantR0<int64_t>(&builder, 10);
  auto x = ConstantR1<int64_t>(&builder, {-3, 3, 9, 13});
  Clamp(zero, x, one);

  std::vector<int64_t> expected = {0, 3, 9, 10};
  ComputeAndCompareR1<int64_t>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, MapTenValues) {
  XlaComputation add_half;
  {
    // add_half(x) = x + 0.5
    XlaBuilder builder("add_half");
    auto x_value =
        Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "x_value");
    auto half = ConstantR0<float>(&builder, 0.5);
    Add(x_value, half);
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    add_half = computation_status.ConsumeValueOrDie();
  }

  XlaComputation clamp;
  {
    // clamp(y) = clamp<0,5>(y)
    XlaBuilder builder("clamp");
    auto y_value =
        Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "y_value");
    auto zero = ConstantR0<float>(&builder, 0.0);
    Clamp(zero, y_value, ConstantR0<float>(&builder, 5));
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    clamp = computation_status.ConsumeValueOrDie();
  }

  XlaComputation mult_relu_add;
  {
    // mult_relu_add(z) = clamp(add_half(2 * max(z, 0)))
    XlaBuilder builder("mult_relu_add");
    auto z_value =
        Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "z_value");
    auto zero = ConstantR0<float>(&builder, 0.0);
    auto two = ConstantR0<float>(&builder, 2.0);
    auto max = Max(z_value, zero);
    auto mult = Mul(two, max);
    auto inner = Map(&builder, {mult}, add_half, {});
    Map(&builder, {inner}, clamp, {});
    auto computation_status = builder.Build();
    ASSERT_IS_OK(computation_status.status());
    mult_relu_add = computation_status.ConsumeValueOrDie();
  }

  XlaBuilder builder("map10");
  {
    auto x = ConstantR1<float>(
        &builder, {2.1, -21.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
    Map(&builder, {x}, mult_relu_add, {0});
  }

  std::vector<float> expected = {4.7, 0.5, 5.0, 0.5, 4.7,
                                 5.0, 0.5, 0.5, 0.5, 3.7};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, RemainderTenValuesS32) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<int32_t>(&builder, {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4});
  auto y = ConstantR0<int32_t>(&builder, 3);
  Rem(x, y);

  std::vector<int32_t> expected = {-2, -1, 0, -2, -1, 0, 1, 2, 0, 1};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, VectorPredicateEqual) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<bool>(&builder, {false, true});
  auto y = ConstantR1<bool>(&builder, {true, false});
  Eq(x, y);

  std::array<bool, 2> expected = {{false, false}};
  ComputeAndCompareR1<bool>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, VectorPredicateNotEqual) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<bool>(&builder, {false, true});
  auto y = ConstantR1<bool>(&builder, {true, false});
  Ne(x, y);

  std::array<bool, 2> expected = {{true, true}};
  ComputeAndCompareR1<bool>(&builder, expected, {});
}

XLA_TEST_F(VecOpsSimpleTest, CbrtSevenValues) {
  XlaBuilder builder(TestName());
  std::vector<float> expected = {16.0, 1888.0, -102.0, 0.16, 0.2, 0., 1.23};
  std::vector<float> cube = {4096.0, 6729859072., -1061208, .004096,
                             0.008,  0.,          1.860867};
  auto x = ConstantR1<float>(&builder, cube);
  Cbrt(x);
  ComputeAndCompareR1<float>(&builder, expected, {},
                             ErrorSpec(/*aabs=*/1e-7, /*arel=*/3e-7));
}

}  // namespace
}  // namespace xla
