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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utils_testDTcc() {
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
#include "tensorflow/compiler/mlir/lite/utils/perception_ops_utils.h"

#include <memory>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace TFL {
namespace {

template <int NInput, int NOutput>
FuncOp createMaxUnpoolingFunc(
    mlir::Builder* builder, const SmallVector<mlir::Type, NInput>& input_types,
    const SmallVector<mlir::Type, NOutput>& output_types) {
  auto func_type = builder->getFunctionType(input_types, output_types);
  auto func =
      FuncOp::create(mlir::NameLoc::get(builder->getStringAttr("fused_func")),
                     "fused_func", func_type, {});

  func.addEntryBlock();
  mlir::StringAttr attr_value = builder->getStringAttr("MaxUnpooling2D");
  func->setAttr("tf._implements", attr_value);
  return func;
}

FuncOp createMaxUnpoolingFunc(mlir::Builder* builder,
                              const SmallVector<int64_t, 4>& input_shape,
                              const SmallVector<int64_t, 4>& output_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utils_testDTcc mht_0(mht_0_v, 222, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils_test.cc", "createMaxUnpoolingFunc");

  auto input_type = RankedTensorType::get(input_shape, builder->getF32Type());
  auto indices_type = RankedTensorType::get(input_shape, builder->getI64Type());
  auto output_type = RankedTensorType::get(output_shape, builder->getF32Type());
  SmallVector<mlir::Type, 2> input_types{input_type, indices_type};
  SmallVector<mlir::Type, 1> output_types{output_type};
  return createMaxUnpoolingFunc<2, 1>(builder, input_types, output_types);
}

template <int N>
ArrayAttr createInt32Array(mlir::Builder* builder, mlir::MLIRContext* context,
                           const SmallVector<int32_t, N>& values) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utils_testDTcc mht_1(mht_1_v, 236, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils_test.cc", "createInt32Array");

  SmallVector<Attribute, N> ret;
  for (int32_t value : values) {
    ret.push_back(builder->getI32IntegerAttr(value));
  }
  return ArrayAttr::get(context, ret);
}

template <int N>
ArrayAttr createInt64Array(mlir::Builder* builder, mlir::MLIRContext* context,
                           const SmallVector<int64_t, N>& values) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utils_testDTcc mht_2(mht_2_v, 249, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils_test.cc", "createInt64Array");

  SmallVector<Attribute, N> ret;
  for (int64_t value : values) {
    ret.push_back(builder->getI64IntegerAttr(value));
  }
  return ArrayAttr::get(context, ret);
}

mlir::TF::FuncAttr createMaxUnpoolingAttr(mlir::MLIRContext* context,
                                          const std::string& padding,
                                          const ArrayAttr& pool_size,
                                          const ArrayAttr& strides) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("padding: \"" + padding + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utils_testDTcc mht_3(mht_3_v, 264, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils_test.cc", "createMaxUnpoolingAttr");

  SmallVector<::mlir::NamedAttribute, 3> fields;

  auto padding_id = ::mlir::StringAttr::get(context, "padding");
  fields.emplace_back(padding_id, StringAttr::get(context, padding));

  auto pool_size_id = ::mlir::StringAttr::get(context, "pool_size");
  fields.emplace_back(pool_size_id, pool_size);

  auto strides_id = ::mlir::StringAttr::get(context, "strides");
  fields.emplace_back(strides_id, strides);

  DictionaryAttr dict = DictionaryAttr::get(context, fields);
  return TF::FuncAttr::get(context, "MaxUnpooling2D", dict);
}

}  // namespace

class PerceptionUtilsTest : public ::testing::Test {
 protected:
  PerceptionUtilsTest() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utils_testDTcc mht_4(mht_4_v, 287, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils_test.cc", "PerceptionUtilsTest");
}

  void SetUp() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utils_testDTcc mht_5(mht_5_v, 292, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils_test.cc", "SetUp");

    context_ = std::make_unique<mlir::MLIRContext>();
    context_
        ->loadDialect<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                      mlir::TF::TensorFlowDialect, TensorFlowLiteDialect>();
    builder_ = std::unique_ptr<mlir::Builder>(new Builder(context_.get()));

    fused_max_unpooling_func_ =
        createMaxUnpoolingFunc(builder_.get(), {2, 4, 4, 2}, {2, 2, 2, 2});

    func_attr_ = createMaxUnpoolingAttr(
        context_.get(), "SAME",
        createInt32Array<2>(builder_.get(), context_.get(), {2, 2}),
        createInt32Array<2>(builder_.get(), context_.get(), {2, 2}));
  }

  void TearDown() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSperception_ops_utils_testDTcc mht_6(mht_6_v, 311, "", "./tensorflow/compiler/mlir/lite/utils/perception_ops_utils_test.cc", "TearDown");

    fused_max_unpooling_func_.erase();
    builder_.reset();
  }

  FuncOp fused_max_unpooling_func_;
  mlir::TF::FuncAttr func_attr_;
  std::unique_ptr<mlir::MLIRContext> context_;
  std::unique_ptr<mlir::Builder> builder_;
};

TEST_F(PerceptionUtilsTest, VerifySignatureValid) {
  mlir::TFL::ConvertMaxUnpoolingFunc convert(fused_max_unpooling_func_,
                                             func_attr_);

  EXPECT_FALSE(failed(convert.VerifySignature()));
}

TEST_F(PerceptionUtilsTest, VerifySignatureInvalid) {
  auto input_type = RankedTensorType::get({1, 2, 2, 1}, builder_->getF32Type());
  auto output_type =
      RankedTensorType::get({1, 2, 1, 1}, builder_->getF32Type());
  SmallVector<mlir::Type, 1> input_types{input_type};
  SmallVector<mlir::Type, 1> output_types{output_type};

  auto max_unpooling_func =
      createMaxUnpoolingFunc<1, 1>(builder_.get(), input_types, output_types);
  mlir::TFL::ConvertMaxUnpoolingFunc convert(max_unpooling_func, func_attr_);

  EXPECT_TRUE(failed(convert.VerifySignature()));
  max_unpooling_func->erase();
}

TEST_F(PerceptionUtilsTest, RewriteValid) {
  mlir::TFL::ConvertMaxUnpoolingFunc convert(fused_max_unpooling_func_,
                                             func_attr_);

  EXPECT_FALSE(failed(convert.RewriteFunc()));
}

TEST_F(PerceptionUtilsTest, RewriteWrongPadding) {
  auto func_attr = createMaxUnpoolingAttr(
      context_.get(), "INVALID",
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2}),
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2}));
  mlir::TFL::ConvertMaxUnpoolingFunc convert(fused_max_unpooling_func_,
                                             func_attr);

  EXPECT_TRUE(failed(convert.RewriteFunc()));
}

TEST_F(PerceptionUtilsTest, RewriteWrongFilter) {
  auto func_attr = createMaxUnpoolingAttr(
      context_.get(), "VALID",
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2, 2}),
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2}));
  mlir::TFL::ConvertMaxUnpoolingFunc convert(fused_max_unpooling_func_,
                                             func_attr);

  EXPECT_TRUE(failed(convert.RewriteFunc()));
}

TEST_F(PerceptionUtilsTest, RewriteWrongStrides) {
  auto func_attr = createMaxUnpoolingAttr(
      context_.get(), "VALID",
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2}),
      createInt32Array<2>(builder_.get(), context_.get(), {2, 2, 0}));
  mlir::TFL::ConvertMaxUnpoolingFunc convert(fused_max_unpooling_func_,
                                             func_attr);

  EXPECT_TRUE(failed(convert.RewriteFunc()));
}

}  // namespace TFL
}  // namespace mlir
