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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shape_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shape_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shape_testDTcc() {
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

#include "tensorflow/compiler/mlir/xla/type_to_shape.h"

#include <iostream>

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"

using mlir::Builder;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::RankedTensorType;
using mlir::UnrankedTensorType;
using mlir::VectorType;

namespace xla {
namespace {

// Simple implementation of a proto matcher comparing string representations.
// Only works as ShapeProto's textual representation is deterministic.
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const tensorflow::protobuf::Message& expected)
      : expected_(expected.SerializeAsString()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shape_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/xla/type_to_shape_test.cc", "ProtoStringMatcher");
}

  template <typename Message>
  bool MatchAndExplain(const Message& p, testing::MatchResultListener*) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shape_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/mlir/xla/type_to_shape_test.cc", "MatchAndExplain");

    return p.SerializeAsString() == expected_;
  }

  void DescribeTo(::std::ostream* os) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shape_testDTcc mht_2(mht_2_v, 226, "", "./tensorflow/compiler/mlir/xla/type_to_shape_test.cc", "DescribeTo");
 *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shape_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/compiler/mlir/xla/type_to_shape_test.cc", "DescribeNegationTo");

    *os << "not equal to expected message: " << expected_;
  }

 private:
  const std::string expected_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const tensorflow::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

TEST(TypeToShapeTest, ConvertPrimitiveTypes) {
  MLIRContext context;
  Builder b(&context);

  EXPECT_EQ(TypeToPrimitiveType(b.getF32Type()), PrimitiveType::F32);
  EXPECT_EQ(TypeToPrimitiveType(b.getIntegerType(1)), PrimitiveType::PRED);
  EXPECT_EQ(TypeToPrimitiveType(b.getIntegerType(17)),
            PrimitiveType::PRIMITIVE_TYPE_INVALID);
}

TEST(TypeToShapeTest, ConvertBasicTypesToTypes) {
  MLIRContext context;
  Builder b(&context);

  EXPECT_TRUE(
      ShapeUtil::IsScalarWithElementType(TypeToShape(b.getF32Type()), F32));
  EXPECT_THAT(
      TypeToShape(VectorType::get({8, 128}, b.getIntegerType(32))).ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::S32, {8, 128}).ToProto()));
  EXPECT_THAT(
      TypeToShape(VectorType::get({8, 128}, b.getF32Type())).ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {8, 128}).ToProto()));

  // MLIR Type that is not representable as XLA Shape.
  EXPECT_THAT(
      TypeToShape(VectorType::get({8, 128}, b.getIntegerType(17))).ToProto(),
      EqualsProto(Shape().ToProto()));
}

TEST(TypeToShapeTest, ConvertMemRefTypeToTypes) {
  MLIRContext context;
  Builder b(&context);

  // Memref without any affine map. Note: memory space is ignored for shape.
  EXPECT_THAT(
      TypeToShape(MemRefType::get({8, 128}, b.getF32Type())).ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {8, 128}).ToProto()));
  EXPECT_THAT(
      TypeToShape(MemRefType::get({100, 13, 210}, b.getF32Type())).ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {100, 13, 210}).ToProto()));

  // Vector types are "flattened" into the end of the shape.
  EXPECT_THAT(
      TypeToShape(MemRefType::get({100, 13, 210},
                                  VectorType::get({8, 128}, b.getF32Type())))
          .ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {100, 13, 210, 8, 128})
              .ToProto()));
}

TEST(TypeToShapeTest, ConvertTensorTypeToTypes) {
  MLIRContext context;
  Builder b(&context);

  EXPECT_THAT(
      TypeToShape(RankedTensorType::get({8, 128}, b.getF32Type())).ToProto(),
      EqualsProto(
          ShapeUtil::MakeShape(PrimitiveType::F32, {8, 128}).ToProto()));

  // Shape cannot represent dynamic shapes.
  // TODO(b/115638799): Update once Shape can support dynamic shapes.
  EXPECT_THAT(TypeToShape(UnrankedTensorType::get(b.getF32Type())).ToProto(),
              EqualsProto(Shape().ToProto()));

  // TODO(jpienaar): Expand to handle more complicated tensor types.
  EXPECT_THAT(
      TypeToShape(RankedTensorType::get(
                      {8, 128}, VectorType::get({16, 16}, b.getF32Type())))
          .ToProto(),
      EqualsProto(Shape().ToProto()));
}

TEST(TypeToShapeTest, ConvertWithShapeRepresentationFn) {
  tensorflow::DataType captured_dtype;
  tensorflow::TensorShape captured_tensor_shape;

  // A dummy shape representation function that does nothing other than
  // capturing arguments passed to it.
  auto test_shape_representation_fn = [&](const tensorflow::TensorShape& shape,
                                          tensorflow::DataType dtype) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shape_testDTcc mht_4(mht_4_v, 330, "", "./tensorflow/compiler/mlir/xla/type_to_shape_test.cc", "lambda");

    captured_tensor_shape = shape;
    captured_dtype = dtype;
    return xla::Shape();
  };

  MLIRContext context;
  Builder b(&context);
  StatusOr<Shape> status_or_shape;

  // Non-fully-defined shape.
  status_or_shape =
      TypeToShape(RankedTensorType::get({-1, 2, 3}, b.getF32Type()),
                  test_shape_representation_fn);
  EXPECT_EQ(status_or_shape.status().code(),
            tensorflow::errors::Code::INVALID_ARGUMENT);

  // Scalar Int32 Tensor, using fast memory.
  status_or_shape =
      TypeToShape(b.getIntegerType(32), test_shape_representation_fn);
  EXPECT_TRUE(status_or_shape.ok());
  EXPECT_EQ(captured_dtype, tensorflow::DataType::DT_INT32);
  EXPECT_EQ(captured_tensor_shape, tensorflow::TensorShape());

  // Ranked Float32 Tensor, not using fast memory.
  status_or_shape =
      TypeToShape(RankedTensorType::get({1, 2, 3}, b.getF32Type()),
                  test_shape_representation_fn);
  EXPECT_TRUE(status_or_shape.ok());
  EXPECT_EQ(captured_dtype, tensorflow::DataType::DT_FLOAT);
  EXPECT_EQ(captured_tensor_shape, tensorflow::TensorShape({1, 2, 3}));
}

TEST(TypeToShapeTest, ConvertMemRefToShape) {
  Shape shape = ShapeUtil::MakeShapeWithLayout(PrimitiveType::F32, {10, 20, 30},
                                               {2, 0, 1});
  MLIRContext context;
  mlir::Builder builder(&context);

  StatusOr<mlir::Type> mlir_type =
      ConvertShapeToType<MemRefType>(shape, builder);
  ASSERT_TRUE(mlir_type.ok());
  mlir::Type type = mlir_type.ConsumeValueOrDie();
  Shape converted = TypeToShape(type);
  EXPECT_TRUE(ShapeUtil::Equal(
      converted, ShapeUtil::MakeShapeWithLayout(PrimitiveType::F32,
                                                {10, 20, 30}, {2, 0, 1})));
  EXPECT_TRUE(ShapeUtil::Equal(converted, shape));
}

TEST(TypeToShapeTest, ConvertMemRefToShape2) {
  Shape shape = ShapeUtil::MakeShapeWithLayout(PrimitiveType::C64, {2, 4, 3, 3},
                                               {2, 3, 1, 0});
  MLIRContext context;
  mlir::Builder builder(&context);

  StatusOr<mlir::Type> mlir_type =
      ConvertShapeToType<MemRefType>(shape, builder);
  ASSERT_TRUE(mlir_type.ok());
  mlir::Type type = mlir_type.ConsumeValueOrDie();
  Shape converted = TypeToShape(type);
  EXPECT_TRUE(ShapeUtil::Equal(
      converted, ShapeUtil::MakeShapeWithLayout(PrimitiveType::C64,
                                                {2, 4, 3, 3}, {2, 3, 1, 0})));
  EXPECT_TRUE(ShapeUtil::Equal(converted, shape));
}

}  // namespace
}  // namespace xla
