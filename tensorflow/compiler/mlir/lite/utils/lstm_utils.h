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

// This header file defines common utils used by TFLite transformation
// passes to work with op attributes.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_LSTM_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_LSTM_UTILS_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utilsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utilsDTh() {
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


#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

constexpr char kTFImplements[] = "tf._implements";
constexpr char kLstmCellSimple[] = "LSTMCellSimple";
constexpr char kLayerNormalizedLstmCellSimple[] =
    "LayerNormalizedLstmCellSimple";
constexpr char kCoupleInputForgetGates[] = "CoupleInputForgetGates";

// A utility class that enables the conversion of the LSTMCellSimple composite
// op into a fused TFL LSTM op. The fused op is contained within a FuncOp
// that also contains other supporting ops needed to construct the operands for
// the fused op. The caller provides the containing FuncOp as input with
// arguments specifying the input, weight, projection and bias.
// The weight, projection, bias and layer norm scale all need to be
// RankedTensorType.
// This class sets the layer norm coefficients to NoneType.
class ConvertLSTMCellSimpleToFusedLSTM {
 public:
  explicit ConvertLSTMCellSimpleToFusedLSTM(mlir::func::FuncOp fused_func_op)
      : fused_func_op_(fused_func_op),
        couple_input_forget_gates_(false),
        builder_(fused_func_op.getBody()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utilsDTh mht_0(mht_0_v, 223, "", "./tensorflow/compiler/mlir/lite/utils/lstm_utils.h", "ConvertLSTMCellSimpleToFusedLSTM");
}

  // not copyable.
  ConvertLSTMCellSimpleToFusedLSTM(const ConvertLSTMCellSimpleToFusedLSTM&) =
      delete;
  ConvertLSTMCellSimpleToFusedLSTM& operator=(
      const ConvertLSTMCellSimpleToFusedLSTM&) = delete;
  virtual ~ConvertLSTMCellSimpleToFusedLSTM() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utilsDTh mht_1(mht_1_v, 233, "", "./tensorflow/compiler/mlir/lite/utils/lstm_utils.h", "~ConvertLSTMCellSimpleToFusedLSTM");
}

  virtual llvm::StringRef GetCompositeOpName() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utilsDTh mht_2(mht_2_v, 238, "", "./tensorflow/compiler/mlir/lite/utils/lstm_utils.h", "GetCompositeOpName");
 return kLstmCellSimple; }

  // Rewrite the func body with constructed fused lstm.
  LogicalResult RewriteFunc();

  int GetNumInputs() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utilsDTh mht_3(mht_3_v, 246, "", "./tensorflow/compiler/mlir/lite/utils/lstm_utils.h", "GetNumInputs");
 return n_input_; }

 protected:
  // verify input func op arguments/attributes and initialize internal state.
  virtual LogicalResult InitializeFromFuncAttributes();
  virtual LogicalResult Initialize();

  void UpdateFuncSignature();
  void GenerateFusedOpOperands();

  void SetWeightForInputToCellGate();
  void SetWeightForInputToInputGate();
  void SetWeightForInputToForgetGate();
  void SetWeightForInputToOutputGate();

  void SetWeightForRecurrentToCellGate();
  void SetWeightForRecurrentToInputGate();
  void SetWeightForRecurrentToForgetGate();
  void SetWeightForRecurrentToOutputGate();

  void SetBiasToCellGate();
  void SetBiasToInputGate();
  void SetBiasToForgetGate();
  void SetBiasToOutputGate();

  void SetProjection();
  void SetProjectionBias();

  void SetInputActivationState();
  void SetInputCellState();

  virtual void SetCellLayerNormCoefficients();
  virtual void SetInputLayerNormCoefficients();
  virtual void SetForgetLayerNormCoefficients();
  virtual void SetOutputLayerNormCoefficients();

  // specified state
  FuncOp fused_func_op_;
  Value input_;
  Value weight_;
  Value bias_;
  Value projection_;
  bool couple_input_forget_gates_;

  // internal state
  Value weight_transposed_;
  Value projection_transposed_;
  RankedTensorType weight_type_;
  RankedTensorType projection_type_;
  int num_gates_;
  int n_cell_;
  int n_output_;
  int n_input_;
  int num_cols_weight_transposed_;
  int num_cols_projection_transposed_;

  // input -> cifg
  Value input2input_;
  Value input2forget_;
  Value input2cell_;
  Value input2output_;

  // recurrent -> cifg
  Value rec2input_;
  Value rec2forget_;
  Value rec2cell_;
  Value rec2output_;

  // bias -> cifg
  Value bias2input_;
  Value bias2forget_;
  Value bias2cell_;
  Value bias2output_;

  // projection
  Value proj_weight_;
  Value proj_bias_;

  // state
  Value input_activation_state_;
  Value input_cell_state_;

  // layer norm coefficients
  Value input_layer_norm_coefficients_;
  Value forget_layer_norm_coefficients_;
  Value cell_layer_norm_coefficients_;
  Value output_layer_norm_coefficients_;

  mlir::TFL::LSTMOp lstm_;

  Value none_;
  SmallVector<int64_t, 1> bias_slice_shape_;
  SmallVector<int64_t, 1> bias_size_values_;
  SmallVector<int64_t, 2> weight_slice_shape_;
  SmallVector<int64_t, 2> weight_slice_size_input_values_;
  SmallVector<int64_t, 2> weight_slice_size_recurrent_values_;
  OpBuilder builder_;
};

// A utility class that enables the conversion of the
// LayerNormalizedLSTMCellSimple composite op into a fused TFL LSTM op. The
// fused op is contained within a FuncOp that also contains other supporting ops
// needed to construct the operands for the fused op. The caller provides the
// containing FuncOp as input with arguments specifying the input, weight,
// projection, bias and layer norm scale. The weight, projection, bias and
// layer norm scale all need to be RankedTensorType.
// This class overrides the layer norm coefficient setters from the base class.
class ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM
    : public ConvertLSTMCellSimpleToFusedLSTM {
 public:
  explicit ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM(
      mlir::func::FuncOp fused_func_op)
      : ConvertLSTMCellSimpleToFusedLSTM(fused_func_op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utilsDTh mht_4(mht_4_v, 361, "", "./tensorflow/compiler/mlir/lite/utils/lstm_utils.h", "ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM");
}

  // not copyable.
  ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM(
      const ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM&) = delete;
  ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM& operator=(
      const ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM&) = delete;
  ~ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utilsDTh mht_5(mht_5_v, 371, "", "./tensorflow/compiler/mlir/lite/utils/lstm_utils.h", "~ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM");
}

  llvm::StringRef GetCompositeOpName() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSutilsPSlstm_utilsDTh mht_6(mht_6_v, 376, "", "./tensorflow/compiler/mlir/lite/utils/lstm_utils.h", "GetCompositeOpName");

    return kLayerNormalizedLstmCellSimple;
  }

 protected:
  LogicalResult Initialize() override;

  void SetCellLayerNormCoefficients() override;
  void SetInputLayerNormCoefficients() override;
  void SetForgetLayerNormCoefficients() override;
  void SetOutputLayerNormCoefficients() override;

 private:
  // specified state
  Value layer_norm_scale_;

  // internal state
  RankedTensorType layer_norm_scale_type_;
  SmallVector<int64_t, 1> layer_norm_slice_shape_;
  SmallVector<int64_t, 1> layer_norm_size_values_;
};

LogicalResult ConvertKerasLSTMLayer(mlir::func::FuncOp func_op,
                                    OpBuilder* builder);

}  // end namespace TFL
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_LSTM_UTILS_H_
