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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utils_testDTcc() {
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

#include "tensorflow/compiler/mlir/lite/utils/lstm_utils.h"

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace TFL {

FuncOp createLstmCompositeFunc(mlir::Builder* builder, bool ln, bool cifg) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utils_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/mlir/lite/utils/lstm_utils_test.cc", "createLstmCompositeFunc");

  SmallVector<int64_t, 2> input_shape{1, 2};
  SmallVector<int64_t, 2> weight_shape{3, 12};
  SmallVector<int64_t, 1> bias_shape{2};
  SmallVector<int64_t, 2> projection_shape{1, 2};
  SmallVector<int64_t, 1> layer_norm_scale{4};
  SmallVector<int64_t, 2> output_shape{1, 2};
  auto input_type = RankedTensorType::get(input_shape, builder->getF32Type());
  auto weight_type = RankedTensorType::get(weight_shape, builder->getF32Type());
  auto bias_type = RankedTensorType::get(bias_shape, builder->getF32Type());
  auto projection_type =
      RankedTensorType::get(projection_shape, builder->getF32Type());
  auto layer_norm_scale_type =
      RankedTensorType::get(layer_norm_scale, builder->getF32Type());
  auto output_type = RankedTensorType::get(output_shape, builder->getF32Type());
  SmallVector<mlir::Type, 4> input_types{input_type, weight_type, bias_type,
                                         projection_type,
                                         layer_norm_scale_type};
  auto func_type = builder->getFunctionType(input_types, output_type);

  auto func =
      FuncOp::create(mlir::NameLoc::get(builder->getStringAttr("fused_func")),
                     "fused_func", func_type, {});
  func.addEntryBlock();

  std::vector<std::string> attributes;
  if (ln) {
    attributes.push_back(kLayerNormalizedLstmCellSimple);
  } else {
    attributes.push_back(kLstmCellSimple);
  }

  if (cifg) {
    attributes.push_back(kCoupleInputForgetGates);
  }

  mlir::StringAttr attr_values =
      builder->getStringAttr(llvm::join(attributes, ","));

  func->setAttr(kTFImplements, attr_values);
  return func;
}

class LstmUtilsTest : public ::testing::Test {
 protected:
  LstmUtilsTest() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utils_testDTcc mht_1(mht_1_v, 263, "", "./tensorflow/compiler/mlir/lite/utils/lstm_utils_test.cc", "LstmUtilsTest");
}

  void SetUp() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utils_testDTcc mht_2(mht_2_v, 268, "", "./tensorflow/compiler/mlir/lite/utils/lstm_utils_test.cc", "SetUp");

    context_ = std::make_unique<mlir::MLIRContext>();
    context_->loadDialect<arith::ArithmeticDialect, mlir::func::FuncDialect,
                          tensor::TensorDialect, mlir::TF::TensorFlowDialect,
                          TensorFlowLiteDialect>();
    builder_ = std::unique_ptr<mlir::Builder>(new Builder(context_.get()));
    fused_lstm_func_ = createLstmCompositeFunc(builder_.get(), false, false);
    fused_lstm_func_cifg_ =
        createLstmCompositeFunc(builder_.get(), false, true);
    fused_ln_lstm_func_ = createLstmCompositeFunc(builder_.get(), true, false);
  }

  void TearDown() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utils_testDTcc mht_3(mht_3_v, 283, "", "./tensorflow/compiler/mlir/lite/utils/lstm_utils_test.cc", "TearDown");

    fused_lstm_func_.erase();
    fused_lstm_func_cifg_.erase();
    fused_ln_lstm_func_.erase();
    builder_.reset();
  }

  FuncOp fused_lstm_func_;
  FuncOp fused_lstm_func_cifg_;
  FuncOp fused_ln_lstm_func_;
  std::unique_ptr<mlir::MLIRContext> context_;
  std::unique_ptr<mlir::Builder> builder_;
};

TEST_F(LstmUtilsTest, ConvertLSTMCellSimple) {
  mlir::TFL::ConvertLSTMCellSimpleToFusedLSTM convert(fused_lstm_func_);

  auto result = convert.RewriteFunc();
  EXPECT_FALSE(failed(result));
  fused_lstm_func_.dump();

  // verify transpose
  EXPECT_EQ(
      fused_lstm_func_->getAttrOfType<StringAttr>(kTFImplements).getValue(),
      convert.GetCompositeOpName());
  EXPECT_EQ(fused_lstm_func_.getNumArguments(), 5);
  EXPECT_EQ(fused_lstm_func_.getFunctionType().getNumResults(), 1);

  auto transpose_op = fused_lstm_func_.getBody().front().begin();
  transpose_op++;
  EXPECT_EQ(
      transpose_op->getOperand(0).getType().cast<RankedTensorType>().getDimSize(
          0),
      3);
  EXPECT_EQ(
      transpose_op->getOperand(0).getType().cast<RankedTensorType>().getDimSize(
          1),
      12);
  EXPECT_EQ(
      transpose_op->getResult(0).getType().cast<RankedTensorType>().getDimSize(
          0),
      12);
  EXPECT_EQ(
      transpose_op->getResult(0).getType().cast<RankedTensorType>().getDimSize(
          1),
      3);

  auto it = fused_lstm_func_.getBody().back().rbegin();
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::func::ReturnOp::getOperationName());
  it++;  // tensor_cast
  it++;  // lstm
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::TFL::LSTMOp::getOperationName());
  EXPECT_EQ(it->getNumOperands(), 24);
  EXPECT_EQ(it->getNumResults(), 1);
  // cifg = false, so input2input is not None.
  EXPECT_FALSE(it->getOperand(1).getType().isa<NoneType>());
  // input layer norm is None
  EXPECT_TRUE(it->getOperand(20).getType().isa<NoneType>());
  // proj_bias is F32
  EXPECT_TRUE(it->getOperand(17)
                  .getType()
                  .cast<RankedTensorType>()
                  .getElementType()
                  .isF32());

  // output gate bias is 0 since it is out of bounds of the bias tensor, so
  // we set its value as a const tensor of specified size and value 0.
  EXPECT_TRUE(mlir::cast<mlir::arith::ConstantOp>(
                  it->getOpOperand(15).get().getDefiningOp())
                  .getValue()
                  .cast<ElementsAttr>()
                  .getValues<FloatAttr>()[0]
                  .getValue()
                  .isExactlyValue(0.0f));

  EXPECT_EQ(fused_lstm_func_.getFunctionType().getNumResults(), 1);
  auto output_types = fused_lstm_func_.getFunctionType().getResults();
  SmallVector<int64_t, 2> output_shape{1, -1};
  EXPECT_EQ(output_types[0].cast<RankedTensorType>().getShape().size(),
            output_shape.size());
  for (int i = 0; i < output_shape.size(); i++) {
    EXPECT_EQ(output_types[0].cast<RankedTensorType>().getDimSize(i),
              output_shape[i]);
  }
}

TEST_F(LstmUtilsTest, ConvertLSTMCellSimpleToFusedLSTMCoupleInputForget) {
  mlir::TFL::ConvertLSTMCellSimpleToFusedLSTM convert(fused_lstm_func_cifg_);

  auto result = convert.RewriteFunc();
  EXPECT_FALSE(failed(result));
  fused_lstm_func_cifg_.dump();

  llvm::SmallVector<std::string, 2> attributes{kLstmCellSimple,
                                               kCoupleInputForgetGates};
  EXPECT_EQ(fused_lstm_func_cifg_->getAttrOfType<StringAttr>(kTFImplements)
                .getValue(),
            llvm::join(attributes, ","));

  auto it = fused_lstm_func_cifg_.getBody().back().rbegin();
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::func::ReturnOp::getOperationName());
  it++;
  it++;
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::TFL::LSTMOp::getOperationName());
  EXPECT_EQ(it->getNumOperands(), 24);
  EXPECT_EQ(it->getNumResults(), 1);
  // cifg = true, so input2input is None.
  EXPECT_TRUE(it->getOperand(1).getType().isa<NoneType>());
}

TEST_F(LstmUtilsTest, ConvertLayerNormLSTMCellSimpleToFusedLSTM) {
  mlir::TFL::ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM convert(
      fused_ln_lstm_func_);

  auto result = convert.RewriteFunc();
  EXPECT_FALSE(failed(result));
  fused_ln_lstm_func_.dump();

  EXPECT_EQ(
      fused_ln_lstm_func_->getAttrOfType<StringAttr>(kTFImplements).getValue(),
      convert.GetCompositeOpName());
  EXPECT_EQ(fused_ln_lstm_func_.getNumArguments(), 5);
  EXPECT_EQ(fused_ln_lstm_func_.getFunctionType().getNumResults(), 1);

  auto it = fused_ln_lstm_func_.getBody().back().rbegin();
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::func::ReturnOp::getOperationName());
  it++;
  it++;
  EXPECT_EQ(it->getName().getStringRef(),
            mlir::TFL::LSTMOp::getOperationName());
  EXPECT_EQ(it->getNumOperands(), 24);
  EXPECT_EQ(it->getNumResults(), 1);
  // cifg = false, so input2input is not None.
  EXPECT_FALSE(it->getOperand(1).getType().isa<NoneType>());

  // input layer norm
  EXPECT_FALSE(it->getOperand(20).getType().isa<NoneType>());
  EXPECT_EQ(
      it->getOperand(20).getType().cast<RankedTensorType>().getShape().size(),
      1);
  EXPECT_EQ(it->getOperand(20).getType().cast<RankedTensorType>().getDimSize(0),
            3);

  EXPECT_EQ(fused_ln_lstm_func_.getFunctionType().getNumResults(), 1);
  auto output_types = fused_ln_lstm_func_.getFunctionType().getResults();
  SmallVector<int64_t, 2> output_shape{1, -1};
  EXPECT_EQ(output_types[0].cast<RankedTensorType>().getShape().size(),
            output_shape.size());
  for (int i = 0; i < output_shape.size(); i++) {
    EXPECT_EQ(output_types[0].cast<RankedTensorType>().getDimSize(i),
              output_shape[i]);
  }
}

}  // namespace TFL
}  // namespace mlir
