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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSconvert_tensor_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSconvert_tensor_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSconvert_tensor_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"

#include <cstring>
#include <initializer_list>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {

using ::testing::Eq;
using ::testing::IsFalse;
using ::testing::IsTrue;

static void RegisterDialects(mlir::MLIRContext &context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSconvert_tensor_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/mlir/tensorflow/utils/convert_tensor_test.cc", "RegisterDialects");

  context.loadDialect<mlir::TF::TensorFlowDialect>();
}

TEST(ConvertTypeToTensorTypeTest, UnrankedTensorType) {
  mlir::MLIRContext context;
  RegisterDialects(context);
  mlir::Builder b(&context);

  PartialTensorShape output_shape =
      ConvertTypeToTensorShape(mlir::UnrankedTensorType::get(b.getF32Type()));
  EXPECT_TRUE(output_shape.IsIdenticalTo(PartialTensorShape()));
}

TEST(ConvertTypeToTensorTypeTest, NonFullyDefinedRankedTensorType) {
  mlir::MLIRContext context;
  RegisterDialects(context);
  mlir::Builder b(&context);

  PartialTensorShape output_shape = ConvertTypeToTensorShape(
      mlir::RankedTensorType::get({-1, 2, 3}, b.getF32Type()));
  EXPECT_TRUE(output_shape.IsIdenticalTo(PartialTensorShape({-1, 2, 3})));
}

TEST(ConvertTypeToTensorTypeTest, FullyDefinedRankedTensorType) {
  mlir::MLIRContext context;
  RegisterDialects(context);
  mlir::Builder b(&context);

  PartialTensorShape output_shape = ConvertTypeToTensorShape(
      mlir::RankedTensorType::get({1, 2, 3}, b.getF32Type()));
  EXPECT_TRUE(output_shape.IsIdenticalTo(PartialTensorShape({1, 2, 3})));
}

TEST(ConvertTypeToTensorTypeTest, ScalarTensorType) {
  mlir::MLIRContext context;
  mlir::Builder b(&context);

  PartialTensorShape output_shape = ConvertTypeToTensorShape(b.getF32Type());
  EXPECT_TRUE(output_shape.IsIdenticalTo(TensorShape()));
}

TEST(ConvertTypeToTensorTypeTest, ConvertStringTensor) {
  mlir::MLIRContext context;
  RegisterDialects(context);
  mlir::Builder b(&context);

  // Create the sample tensor to convert.
  Tensor tensor(DT_STRING, TensorShape({1, 2, 2, 1}));
  EXPECT_EQ(4, tensor.NumElements());
  auto Tt = tensor.flat<tstring>();
  Tt.setValues({"one", "two", "three", "four"});
  auto value_or_status = ConvertTensor(tensor, &b);
  ASSERT_TRUE(value_or_status.ok());
  auto attr = value_or_status.ValueOrDie();

  EXPECT_TRUE(attr.isa<mlir::DenseStringElementsAttr>());
  auto string_attr = attr.cast<mlir::DenseStringElementsAttr>();
  auto string_values = string_attr.getRawStringData();
  ASSERT_EQ(string_values.size(), 4);
  EXPECT_EQ(string_values[0], mlir::StringRef("one"));
  EXPECT_EQ(string_values[1], mlir::StringRef("two"));
  EXPECT_EQ(string_values[2], mlir::StringRef("three"));
  EXPECT_EQ(string_values[3], mlir::StringRef("four"));
}

class ConvertTensorTest : public ::testing::Test {
 protected:
  template <typename T>
  void VerifyConversion(std::initializer_list<T> values, DataType dtype,
                        mlir::Type expected_ty) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSconvert_tensor_testDTcc mht_1(mht_1_v, 284, "", "./tensorflow/compiler/mlir/tensorflow/utils/convert_tensor_test.cc", "VerifyConversion");

    mlir::Builder b(expected_ty.getContext());
    Tensor tensor(dtype, TensorShape({static_cast<int64_t>(values.size())}));
    tensor.flat<T>().setValues(values);

    auto value_or = ConvertTensor(tensor, &b);
    TF_ASSERT_OK(value_or.status());
    auto attr = value_or.ValueOrDie();

    EXPECT_EQ(attr.getType().getElementType(), expected_ty);

    Tensor out;
    TF_ASSERT_OK(ConvertToTensor(attr, &out));

    test::ExpectTensorEqual<T>(tensor, out);
  }
};

TEST_F(ConvertTensorTest, Simple) {
  mlir::MLIRContext context;
  RegisterDialects(context);
  ASSERT_NO_FATAL_FAILURE(VerifyConversion<Eigen::half>(
      {Eigen::half(1.0)}, DT_HALF, mlir::FloatType::getF16(&context)));
  ASSERT_NO_FATAL_FAILURE(
      VerifyConversion<bfloat16>({bfloat16(1.0), bfloat16(-1.0)}, DT_BFLOAT16,
                                 mlir::FloatType::getBF16(&context)));
  ASSERT_NO_FATAL_FAILURE(VerifyConversion<float>(
      {1.0, -1.0}, DT_FLOAT, mlir::FloatType::getF32(&context)));
  ASSERT_NO_FATAL_FAILURE(VerifyConversion<double>(
      {1.0, -1.0}, DT_DOUBLE, mlir::FloatType::getF64(&context)));

  ASSERT_NO_FATAL_FAILURE(VerifyConversion<int8>(
      {1, -1}, DT_INT8, mlir::IntegerType::get(&context, 8)));
  ASSERT_NO_FATAL_FAILURE(VerifyConversion<int16>(
      {1, -1}, DT_INT16, mlir::IntegerType::get(&context, 16)));
  ASSERT_NO_FATAL_FAILURE(VerifyConversion<int32>(
      {1, -1}, DT_INT32, mlir::IntegerType::get(&context, 32)));
  ASSERT_NO_FATAL_FAILURE(VerifyConversion<int64_t>(
      {1, -1}, DT_INT64, mlir::IntegerType::get(&context, 64)));

  ASSERT_NO_FATAL_FAILURE(VerifyConversion<uint8>(
      {1, 2}, DT_UINT8,
      mlir::IntegerType::get(
          &context, 8, mlir::IntegerType::SignednessSemantics::Unsigned)));
  ASSERT_NO_FATAL_FAILURE(VerifyConversion<uint16>(
      {1, 2}, DT_UINT16,
      mlir::IntegerType::get(
          &context, 16, mlir::IntegerType::SignednessSemantics::Unsigned)));
  ASSERT_NO_FATAL_FAILURE(VerifyConversion<uint32>(
      {1, 2}, DT_UINT32,
      mlir::IntegerType::get(
          &context, 32, mlir::IntegerType::SignednessSemantics::Unsigned)));
  ASSERT_NO_FATAL_FAILURE(VerifyConversion<uint64>(
      {1, 2}, DT_UINT64,
      mlir::IntegerType::get(
          &context, 64, mlir::IntegerType::SignednessSemantics::Unsigned)));

  ASSERT_NO_FATAL_FAILURE(VerifyConversion<std::complex<float>>(
      {{0.0, 1.0}, {1.0, 0.0}}, DT_COMPLEX64,
      mlir::ComplexType::get(mlir::FloatType::getF32(&context))));
  ASSERT_NO_FATAL_FAILURE(VerifyConversion<std::complex<double>>(
      {{0.0, 1.0}, {1.0, 0.0}}, DT_COMPLEX128,
      mlir::ComplexType::get(mlir::FloatType::getF64(&context))));
}

bool IsSplat(mlir::ElementsAttr attr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSconvert_tensor_testDTcc mht_2(mht_2_v, 352, "", "./tensorflow/compiler/mlir/tensorflow/utils/convert_tensor_test.cc", "IsSplat");

  return attr.cast<mlir::DenseElementsAttr>().isSplat();
}

TEST(ConvertTensorProtoTest, SplatTensor) {
  // We construct a sparse TensorProto representing 2^35 float elements, all of
  // them 42. Our conversion routine should not materialize these elements when
  // creating the Attribute. If it tries to, we'll crash OOM here.
  TensorProto tensor;
  tensor.set_dtype(DT_FLOAT);
  tensor.mutable_tensor_shape()->add_dim()->set_size(1ULL << 35);
  tensor.add_float_val(42.0);

  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  TF_ASSERT_OK_AND_ASSIGN(mlir::ElementsAttr attribute,
                          ConvertTensorProto(tensor, &builder));
  EXPECT_THAT(
      attribute,
      AllOf(Eq(mlir::DenseElementsAttr::get(
                mlir::RankedTensorType::get({1ULL << 35}, builder.getF32Type()),
                42.0f)),
            ResultOf(IsSplat, IsTrue())));
}

TEST(ConvertTensorProtoTest, NonSplatTensor) {
  TensorProto proto = tensor::CreateTensorProto<float>(
      /*values=*/{1.0f, 2.0f, 3.0f, 4.0f}, /*shape=*/{2, 2});
  mlir::MLIRContext context;
  mlir::Builder builder(&context);

  TF_ASSERT_OK_AND_ASSIGN(mlir::ElementsAttr attribute,
                          ConvertTensorProto(proto, &builder));
  EXPECT_THAT(
      attribute,
      AllOf(Eq(mlir::DenseElementsAttr::get(
                mlir::RankedTensorType::get({2, 2}, builder.getF32Type()),
                {1.0f, 2.0f, 3.0f, 4.0f})),
            ResultOf(IsSplat, IsFalse())));
}

}  // namespace
}  // namespace tensorflow
