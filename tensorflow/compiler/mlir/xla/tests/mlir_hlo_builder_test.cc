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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStestsPSmlir_hlo_builder_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStestsPSmlir_hlo_builder_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStestsPSmlir_hlo_builder_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.h"

#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

namespace {

static void ExpectHasSubstr(absl::string_view s, absl::string_view expected) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("s: \"" + std::string(s.data(), s.size()) + "\"");
   mht_0_v.push_back("expected: \"" + std::string(expected.data(), expected.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStestsPSmlir_hlo_builder_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/mlir/xla/tests/mlir_hlo_builder_test.cc", "ExpectHasSubstr");

  EXPECT_TRUE(absl::StrContains(s, expected))
      << s << " does not contain " << expected;
}

class XlaBuilderTest : public ::testing::Test {
 protected:
  XlaBuilderTest()
      : name_(SetupTest()),
        module_(mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_))),
        builder_(&module_->getBodyRegion()),
        xla_builder_(name_, builder_, module_->getLoc()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStestsPSmlir_hlo_builder_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/mlir/xla/tests/mlir_hlo_builder_test.cc", "XlaBuilderTest");

    context_.loadDialect<mlir::mhlo::MhloDialect>();
  }

  std::string SetupTest() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStestsPSmlir_hlo_builder_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/mlir/xla/tests/mlir_hlo_builder_test.cc", "SetupTest");

    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  // Retuns the MLIR op string representation of the given XlaOp.
  std::string GetMlirOpString(XlaOp xla_op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStestsPSmlir_hlo_builder_testDTcc mht_3(mht_3_v, 237, "", "./tensorflow/compiler/mlir/xla/tests/mlir_hlo_builder_test.cc", "GetMlirOpString");

    std::string str;
    llvm::raw_string_ostream ostream{str};
    xla_builder_.GetValue(xla_op).print(ostream);
    ostream.flush();
    return str;
  }

  std::string name_;
  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  mlir::OpBuilder builder_;
  MlirHloBuilder xla_builder_;
};

TEST_F(XlaBuilderTest, CreateToken) {
  auto token = CreateToken(&xla_builder_);
  auto str = GetMlirOpString(token);

  TF_ASSERT_OK(xla_builder_.GetCurrentStatus());

  ExpectHasSubstr(GetMlirOpString(token),
                  R"("mhlo.create_token"() : () -> !mhlo.token)");
}

TEST_F(XlaBuilderTest, Infeed) {
  auto token = CreateToken(&xla_builder_);
  auto infeed = InfeedWithToken(token, ShapeUtil::MakeShape(F32, {4, 8}), "");

  TF_ASSERT_OK(xla_builder_.GetCurrentStatus());
  ExpectHasSubstr(
      GetMlirOpString(infeed),
      R"("mhlo.tuple"(%1#0, %1#1) : (tensor<4x8xf32>, !mhlo.token) -> tuple<tensor<4x8xf32>)");
}

TEST_F(XlaBuilderTest, Outfeed) {
  auto outfeed_shape = ShapeUtil::MakeShape(F32, {4, 8});
  auto data = ConstantLiteral(
      &xla_builder_,
      LiteralUtil::CreateFromDimensions(F32, outfeed_shape.dimensions()));
  auto token = CreateToken(&xla_builder_);
  auto outfeed = OutfeedWithToken(data, token, outfeed_shape, "");

  TF_ASSERT_OK(xla_builder_.GetCurrentStatus());
  ExpectHasSubstr(
      GetMlirOpString(outfeed),
      R"("mhlo.outfeed"(%0, %1) {outfeed_config = ""} : (tensor<4x8xf32>, !mhlo.token) -> !mhlo.token)");
}

TEST_F(XlaBuilderTest, ConcatInDim) {
  auto data0 = ConstantLiteral(
      &xla_builder_, LiteralUtil::CreateFromDimensions(F32, {2, 4, 5}));
  auto data1 = ConstantLiteral(
      &xla_builder_, LiteralUtil::CreateFromDimensions(F32, {2, 6, 5}));
  auto concat = ConcatInDim(&xla_builder_, {data0, data1}, 1);

  TF_ASSERT_OK(xla_builder_.GetCurrentStatus());
  ExpectHasSubstr(
      GetMlirOpString(concat),
      R"("mhlo.concatenate"(%0, %1) {dimension = 1 : i64} : (tensor<2x4x5xf32>, tensor<2x6x5xf32>) -> tensor<2x10x5xf32>)");
}

TEST_F(XlaBuilderTest, Tuple) {
  auto data0 = ConstantLiteral(&xla_builder_,
                               LiteralUtil::CreateFromDimensions(F32, {3, 7}));
  auto data1 = ConstantLiteral(&xla_builder_,
                               LiteralUtil::CreateFromDimensions(F32, {}));
  auto tuple = Tuple(&xla_builder_, {data0, data1});

  TF_ASSERT_OK(xla_builder_.GetCurrentStatus());
  ExpectHasSubstr(
      GetMlirOpString(tuple),
      R"("mhlo.tuple"(%0, %1) : (tensor<3x7xf32>, tensor<f32>) -> tuple<tensor<3x7xf32>, tensor<f32>>)");
}

TEST_F(XlaBuilderTest, GetTupleElement) {
  auto data0 = ConstantLiteral(&xla_builder_,
                               LiteralUtil::CreateFromDimensions(F32, {3, 7}));
  auto data1 = ConstantLiteral(&xla_builder_,
                               LiteralUtil::CreateFromDimensions(F32, {}));
  auto tuple_data = Tuple(&xla_builder_, {data0, data1});
  auto gte = GetTupleElement(tuple_data, 1);

  TF_ASSERT_OK(xla_builder_.GetCurrentStatus());
  ExpectHasSubstr(
      GetMlirOpString(gte),
      R"("mhlo.get_tuple_element"(%2) {index = 1 : i32} : (tuple<tensor<3x7xf32>, tensor<f32>>) -> tensor<f32>)");
}

TEST_F(XlaBuilderTest, Slice) {
  auto data = ConstantLiteral(&xla_builder_,
                              LiteralUtil::CreateFromDimensions(F32, {3, 7}));
  auto slice = Slice(data, {0, 1}, {2, 5}, {1, 1});

  TF_ASSERT_OK(xla_builder_.GetCurrentStatus());
  ExpectHasSubstr(
      GetMlirOpString(slice),
      R"("mhlo.slice"(%0) {limit_indices = dense<[2, 5]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<3x7xf32>) -> tensor<2x4xf32>)");
}

TEST_F(XlaBuilderTest, Pad) {
  auto data = ConstantLiteral(&xla_builder_,
                              LiteralUtil::CreateFromDimensions(F32, {3, 7}));
  auto zero = ConstantLiteral(&xla_builder_, LiteralUtil::Zero(F32));

  PaddingConfig padding_config;
  auto* dims0 = padding_config.add_dimensions();
  dims0->set_edge_padding_low(1);
  dims0->set_interior_padding(0);
  dims0->set_edge_padding_high(2);
  auto* dims1 = padding_config.add_dimensions();
  dims1->set_edge_padding_low(3);
  dims1->set_interior_padding(1);
  dims1->set_edge_padding_high(0);
  auto pad = Pad(data, zero, padding_config);

  TF_ASSERT_OK(xla_builder_.GetCurrentStatus());
  ExpectHasSubstr(
      GetMlirOpString(pad),
      R"("mhlo.pad"(%0, %1) {edge_padding_high = dense<[2, 0]> : tensor<2xi64>, edge_padding_low = dense<[1, 3]> : tensor<2xi64>, interior_padding = dense<[0, 1]> : tensor<2xi64>} : (tensor<3x7xf32>, tensor<f32>) -> tensor<6x16xf32>)");
}

}  // namespace
}  // namespace xla
