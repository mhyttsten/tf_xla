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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompute_constant_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompute_constant_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompute_constant_testDTcc() {
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
#include <vector>

#include "absl/strings/match.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

// An enumerator for the client types that we want to iterate over in
// the various tests.
enum class ClientType { kLocal, kCompileOnly };
ClientType client_types[] = {ClientType::kLocal, ClientType::kCompileOnly};

class ComputeConstantTest : public ::testing::Test {
 public:
  explicit ComputeConstantTest(se::Platform* platform = nullptr)
      : platform_(platform) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompute_constant_testDTcc mht_0(mht_0_v, 217, "", "./tensorflow/compiler/xla/tests/compute_constant_test.cc", "ComputeConstantTest");
}

  std::string TestName() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompute_constant_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/xla/tests/compute_constant_test.cc", "TestName");

    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  Client* ClientOrDie(se::Platform* platform, ClientType client_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompute_constant_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/xla/tests/compute_constant_test.cc", "ClientOrDie");

    if (client_type == ClientType::kLocal) {
      StatusOr<Client*> result =
          ClientLibrary::GetOrCreateLocalClient(platform);
      TF_CHECK_OK(result.status())
          << "could not create LocalClient for testing";
      return result.ValueOrDie();
    } else if (client_type == ClientType::kCompileOnly) {
      StatusOr<Client*> result =
          ClientLibrary::GetOrCreateCompileOnlyClient(platform);
      TF_CHECK_OK(result.status())
          << "could not create CompileOnlyClient for testing";
      return result.ValueOrDie();
    }
    LOG(FATAL) << "invalid client_type value";
  }

  StatusOr<Literal> ComputeConstantLiteral(Client* client, const XlaOp operand,
                                           XlaBuilder* builder,
                                           Layout* output_layout = nullptr) {
    TF_ASSIGN_OR_RETURN(auto subgraph, builder->BuildConstantSubGraph(operand));
    TF_ASSIGN_OR_RETURN(auto computed,
                        client->ComputeConstant(subgraph, output_layout));
    return std::move(computed);
  }

  template <class Scalar>
  StatusOr<Scalar> ComputeConstantScalar(Client* client, const XlaOp operand,
                                         XlaBuilder* builder) {
    TF_ASSIGN_OR_RETURN(auto literal, ComputeConstantLiteral(client, operand,
                                                             builder, nullptr));
    return literal.Get<Scalar>({});
  }

  bool IsConstant(const XlaOp operand, XlaBuilder* builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPScompute_constant_testDTcc mht_3(mht_3_v, 266, "", "./tensorflow/compiler/xla/tests/compute_constant_test.cc", "IsConstant");

    StatusOr<bool> result = builder->IsConstant(operand);
    EXPECT_TRUE(result.ok()) << result.status();
    return result.ok() ? result.ValueOrDie() : false;
  }

  se::Platform* platform_;
};

TEST_F(ComputeConstantTest, ScalarInt32Literal) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation = ConstantR0<int32_t>(&b, 42);
    EXPECT_TRUE(IsConstant(computation, &b));

    auto value = ComputeConstantScalar<int32_t>(client, computation, &b);
    ASSERT_TRUE(value.ok()) << value.status();
    EXPECT_EQ(value.ValueOrDie(), 42);
  }
}

TEST_F(ComputeConstantTest, ScalarFloatAdd) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation =
        Add(ConstantR0<float>(&b, 42.5f), ConstantR0<float>(&b, 1.5f));
    EXPECT_TRUE(IsConstant(computation, &b));

    auto value = ComputeConstantScalar<float>(client, computation, &b);
    ASSERT_TRUE(value.ok()) << value.status();
    EXPECT_EQ(value.ValueOrDie(), 44.0f);
  }
}

TEST_F(ComputeConstantTest, ScalarRng) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation =
        RngUniform(ConstantR0<float>(&b, 1.1f), ConstantR0<float>(&b, 2.1f),
                   ShapeUtil::MakeShape(F32, {}));
    EXPECT_FALSE(IsConstant(computation, &b));

    auto value = ComputeConstantScalar<float>(client, computation, &b);
    ASSERT_FALSE(value.ok())
        << "computing a RNG value should not be considered a constant";
  }
}

TEST_F(ComputeConstantTest, DirectParamMissing) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "param");
    EXPECT_FALSE(IsConstant(computation, &b));

    auto value = ComputeConstantScalar<float>(client, computation, &b);
    EXPECT_TRUE(
        absl::StrContains(value.status().ToString(), "depends on a parameter"))
        << value.status();
  }
}

TEST_F(ComputeConstantTest, GetDimensionSize) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto add =
        Add(ConstantR1<float>(&b, {1.0f}), ConstantR1<float>(&b, {1.0f}));
    auto get_dimension_size = GetDimensionSize(add, 0);
    EXPECT_TRUE(IsConstant(get_dimension_size, &b));

    TF_ASSERT_OK_AND_ASSIGN(auto value, ComputeConstantScalar<int32_t>(
                                            client, get_dimension_size, &b));
    EXPECT_EQ(value, 1);
  }
}

TEST_F(ComputeConstantTest, MultipleGetDimensionSize) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto add =
        Add(ConstantR2<float>(&b, {{1.0f}}), ConstantR2<float>(&b, {{1.0f}}));
    auto get_dimension_size = GetDimensionSize(add, 0);
    auto get_dimension_size_2 = GetDimensionSize(add, 0);
    auto add_2 = Add(get_dimension_size, get_dimension_size_2);
    EXPECT_TRUE(IsConstant(add_2, &b));

    TF_ASSERT_OK_AND_ASSIGN(auto value,
                            ComputeConstantScalar<int32_t>(client, add_2, &b));
    EXPECT_EQ(value, 2);
  }
}

// Test computation of an expression interspersed with param nodes but
// the expression does not depend on the param nodes.
TEST_F(ComputeConstantTest, UnrelatedParam) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());

    auto param_a = Parameter(&b, 10, ShapeUtil::MakeShape(F32, {}), "param0");
    auto constant_4 =
        Add(ConstantR0<float>(&b, 2.5f), ConstantR0<float>(&b, 1.5f));
    auto not_constant_a = Add(constant_4, param_a);

    auto param_b = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {}), "param1");
    auto constant_9 =
        Mul(ConstantR0<float>(&b, 2.0f), ConstantR0<float>(&b, 4.5f));
    auto not_constant_b = Add(param_b, constant_9);

    auto constant_13 = Add(constant_4, constant_9);
    Add(not_constant_b, Add(constant_13, not_constant_a));

    EXPECT_TRUE(IsConstant(constant_13, &b));

    TF_ASSERT_OK_AND_ASSIGN(
        auto value, ComputeConstantScalar<float>(client, constant_13, &b));
    EXPECT_EQ(value, 13.0f);
  }
}

TEST_F(ComputeConstantTest, NonScalarAdd) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());

    auto computation =
        Add(ConstantR1<int32_t>(&b, {1, 2}), ConstantR1<int32_t>(&b, {3, 4}));
    EXPECT_TRUE(IsConstant(computation, &b));

    TF_ASSERT_OK_AND_ASSIGN(auto computed,
                            ComputeConstantLiteral(client, computation, &b));
    Literal expected_literal = LiteralUtil::CreateR1<int32_t>({4, 6});
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, computed));
  }
}

TEST_F(ComputeConstantTest, IntegerDivide) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation =
        Div(ConstantR0<int32_t>(&b, 15), ConstantR0<int32_t>(&b, 3));
    EXPECT_TRUE(IsConstant(computation, &b));

    TF_ASSERT_OK_AND_ASSIGN(auto computed,
                            ComputeConstantLiteral(client, computation, &b));
    Literal expected_literal = LiteralUtil::CreateR0<int32_t>(5);
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, computed));
  }
}

XLA_TEST_F(ComputeConstantTest, Layout) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());

    std::vector<std::vector<int64_t>> layouts = {{0, 1}, {1, 0}};
    for (const std::vector<int64_t>& layout : layouts) {
      auto layout_proto = LayoutUtil::MakeLayout(layout);
      TF_ASSERT_OK_AND_ASSIGN(
          auto computed, ComputeConstantLiteral(
                             client,
                             Add(ConstantR2<int32_t>(&b, {{1, 2}, {3, 4}}),
                                 ConstantR2<int32_t>(&b, {{10, 20}, {30, 40}})),
                             &b, &layout_proto));

      Literal expected_literal = LiteralUtil::CreateR2WithLayout<int32_t>(
          {{11, 22}, {33, 44}}, LayoutUtil::MakeLayout(layout));
      ASSERT_TRUE(LiteralTestUtil::EqualShapesAndLayouts(
          expected_literal.shape(), computed.shape()));
      EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, computed));
    }
  }
}

}  // namespace
}  // namespace xla
