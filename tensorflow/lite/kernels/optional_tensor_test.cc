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
class MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc() {
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
// Unit test for TFLite LSTM op.

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

class LSTMOpModel : public SingleOpModel {
 public:
  LSTMOpModel(int n_batch, int n_input, int n_cell, int n_output, bool use_cifg,
              bool use_peephole, bool use_projection_weights,
              bool use_projection_bias, float cell_clip, float proj_clip,
              const std::vector<std::vector<int>>& input_shapes)
      : n_batch_(n_batch),
        n_input_(n_input),
        n_cell_(n_cell),
        n_output_(n_output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "LSTMOpModel");

    input_ = AddInput(TensorType_FLOAT32);

    if (use_cifg) {
      input_to_input_weights_ = AddNullInput();
    } else {
      input_to_input_weights_ = AddInput(TensorType_FLOAT32);
    }

    input_to_forget_weights_ = AddInput(TensorType_FLOAT32);
    input_to_cell_weights_ = AddInput(TensorType_FLOAT32);
    input_to_output_weights_ = AddInput(TensorType_FLOAT32);

    if (use_cifg) {
      recurrent_to_input_weights_ = AddNullInput();
    } else {
      recurrent_to_input_weights_ = AddInput(TensorType_FLOAT32);
    }

    recurrent_to_forget_weights_ = AddInput(TensorType_FLOAT32);
    recurrent_to_cell_weights_ = AddInput(TensorType_FLOAT32);
    recurrent_to_output_weights_ = AddInput(TensorType_FLOAT32);

    if (use_peephole) {
      if (use_cifg) {
        cell_to_input_weights_ = AddNullInput();
      } else {
        cell_to_input_weights_ = AddInput(TensorType_FLOAT32);
      }
      cell_to_forget_weights_ = AddInput(TensorType_FLOAT32);
      cell_to_output_weights_ = AddInput(TensorType_FLOAT32);
    } else {
      cell_to_input_weights_ = AddNullInput();
      cell_to_forget_weights_ = AddNullInput();
      cell_to_output_weights_ = AddNullInput();
    }

    if (use_cifg) {
      input_gate_bias_ = AddNullInput();
    } else {
      input_gate_bias_ = AddInput(TensorType_FLOAT32);
    }
    forget_gate_bias_ = AddInput(TensorType_FLOAT32);
    cell_gate_bias_ = AddInput(TensorType_FLOAT32);
    output_gate_bias_ = AddInput(TensorType_FLOAT32);

    if (use_projection_weights) {
      projection_weights_ = AddInput(TensorType_FLOAT32);
      if (use_projection_bias) {
        projection_bias_ = AddInput(TensorType_FLOAT32);
      } else {
        projection_bias_ = AddNullInput();
      }
    } else {
      projection_weights_ = AddNullInput();
      projection_bias_ = AddNullInput();
    }

    // Adding the 2 input state tensors.
    input_activation_state_ = AddVariableInput(
        TensorData{TensorType_FLOAT32, {n_output_ * n_batch_}});
    input_cell_state_ =
        AddVariableInput(TensorData{TensorType_FLOAT32, {n_cell_ * n_batch_}});

    output_ = AddOutput(TensorType_FLOAT32);

    SetBuiltinOp(BuiltinOperator_LSTM, BuiltinOptions_LSTMOptions,
                 CreateLSTMOptions(builder_, ActivationFunctionType_TANH,
                                   cell_clip, proj_clip)
                     .Union());
    BuildInterpreter(input_shapes);
  }

  void SetInputToInputWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_1(mht_1_v, 283, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetInputToInputWeights");

    PopulateTensor(input_to_input_weights_, f);
  }

  void SetInputToForgetWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_2(mht_2_v, 290, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetInputToForgetWeights");

    PopulateTensor(input_to_forget_weights_, f);
  }

  void SetInputToCellWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_3(mht_3_v, 297, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetInputToCellWeights");

    PopulateTensor(input_to_cell_weights_, f);
  }

  void SetInputToOutputWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_4(mht_4_v, 304, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetInputToOutputWeights");

    PopulateTensor(input_to_output_weights_, f);
  }

  void SetRecurrentToInputWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_5(mht_5_v, 311, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetRecurrentToInputWeights");

    PopulateTensor(recurrent_to_input_weights_, f);
  }

  void SetRecurrentToForgetWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_6(mht_6_v, 318, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetRecurrentToForgetWeights");

    PopulateTensor(recurrent_to_forget_weights_, f);
  }

  void SetRecurrentToCellWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_7(mht_7_v, 325, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetRecurrentToCellWeights");

    PopulateTensor(recurrent_to_cell_weights_, f);
  }

  void SetRecurrentToOutputWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_8(mht_8_v, 332, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetRecurrentToOutputWeights");

    PopulateTensor(recurrent_to_output_weights_, f);
  }

  void SetCellToInputWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_9(mht_9_v, 339, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetCellToInputWeights");

    PopulateTensor(cell_to_input_weights_, f);
  }

  void SetCellToForgetWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_10(mht_10_v, 346, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetCellToForgetWeights");

    PopulateTensor(cell_to_forget_weights_, f);
  }

  void SetCellToOutputWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_11(mht_11_v, 353, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetCellToOutputWeights");

    PopulateTensor(cell_to_output_weights_, f);
  }

  void SetInputGateBias(std::initializer_list<float> f) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_12(mht_12_v, 360, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetInputGateBias");

    PopulateTensor(input_gate_bias_, f);
  }

  void SetForgetGateBias(std::initializer_list<float> f) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_13(mht_13_v, 367, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetForgetGateBias");

    PopulateTensor(forget_gate_bias_, f);
  }

  void SetCellBias(std::initializer_list<float> f) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_14(mht_14_v, 374, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetCellBias");

    PopulateTensor(cell_gate_bias_, f);
  }

  void SetOutputGateBias(std::initializer_list<float> f) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_15(mht_15_v, 381, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetOutputGateBias");

    PopulateTensor(output_gate_bias_, f);
  }

  void SetProjectionWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_16(mht_16_v, 388, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetProjectionWeights");

    PopulateTensor(projection_weights_, f);
  }

  void SetProjectionBias(std::initializer_list<float> f) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_17(mht_17_v, 395, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetProjectionBias");

    PopulateTensor(projection_bias_, f);
  }

  void SetInput(int offset, float* begin, float* end) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_18(mht_18_v, 402, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "SetInput");

    PopulateTensor(input_, offset, begin, end);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  void Verify() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_19(mht_19_v, 410, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "Verify");

    auto model = tflite::UnPackModel(builder_.GetBufferPointer());
    EXPECT_NE(model, nullptr);
  }

  int num_inputs() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_20(mht_20_v, 418, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "num_inputs");
 return n_input_; }
  int num_outputs() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_21(mht_21_v, 422, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "num_outputs");
 return n_output_; }
  int num_cells() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_22(mht_22_v, 426, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "num_cells");
 return n_cell_; }
  int num_batches() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSoptional_tensor_testDTcc mht_23(mht_23_v, 430, "", "./tensorflow/lite/kernels/optional_tensor_test.cc", "num_batches");
 return n_batch_; }

 private:
  int input_;
  int input_to_input_weights_;
  int input_to_forget_weights_;
  int input_to_cell_weights_;
  int input_to_output_weights_;

  int recurrent_to_input_weights_;
  int recurrent_to_forget_weights_;
  int recurrent_to_cell_weights_;
  int recurrent_to_output_weights_;

  int cell_to_input_weights_;
  int cell_to_forget_weights_;
  int cell_to_output_weights_;

  int input_gate_bias_;
  int forget_gate_bias_;
  int cell_gate_bias_;
  int output_gate_bias_;

  int projection_weights_;
  int projection_bias_;
  int input_activation_state_;
  int input_cell_state_;

  int output_;

  int n_batch_;
  int n_input_;
  int n_cell_;
  int n_output_;
};

TEST(LSTMOpTest, BlackBoxTestWithCifgWithPeepholeNoProjectionNoClipping) {
  const int n_batch = 1;
  const int n_input = 2;
  // n_cell and n_output have the same size when there is no projection.
  const int n_cell = 4;
  const int n_output = 4;

  LSTMOpModel lstm(n_batch, n_input, n_cell, n_output,
                   /*use_cifg=*/true, /*use_peephole=*/true,
                   /*use_projection_weights=*/false,
                   /*use_projection_bias=*/false,
                   /*cell_clip=*/0.0, /*proj_clip=*/0.0,
                   {
                       {n_batch, n_input},  // input tensor

                       {0, 0},             // input_to_input_weight tensor
                       {n_cell, n_input},  // input_to_forget_weight tensor
                       {n_cell, n_input},  // input_to_cell_weight tensor
                       {n_cell, n_input},  // input_to_output_weight tensor

                       {0, 0},              // recurrent_to_input_weight tensor
                       {n_cell, n_output},  // recurrent_to_forget_weight tensor
                       {n_cell, n_output},  // recurrent_to_cell_weight tensor
                       {n_cell, n_output},  // recurrent_to_output_weight tensor

                       {0},       // cell_to_input_weight tensor
                       {n_cell},  // cell_to_forget_weight tensor
                       {n_cell},  // cell_to_output_weight tensor

                       {0},       // input_gate_bias tensor
                       {n_cell},  // forget_gate_bias tensor
                       {n_cell},  // cell_gate_bias tensor
                       {n_cell},  // output_gate_bias tensor

                       {0, 0},  // projection_weight tensor
                       {0},     // projection_bias tensor
                   });

  lstm.SetInputToCellWeights({-0.49770179, -0.27711356, -0.09624726, 0.05100781,
                              0.04717243, 0.48944736, -0.38535351,
                              -0.17212132});

  lstm.SetInputToForgetWeights({-0.55291498, -0.42866567, 0.13056988,
                                -0.3633365, -0.22755712, 0.28253698, 0.24407166,
                                0.33826375});

  lstm.SetInputToOutputWeights({0.10725588, -0.02335852, -0.55932593,
                                -0.09426838, -0.44257352, 0.54939759,
                                0.01533556, 0.42751634});

  lstm.SetCellBias({0., 0., 0., 0.});

  lstm.SetForgetGateBias({1., 1., 1., 1.});

  lstm.SetOutputGateBias({0., 0., 0., 0.});

  lstm.SetRecurrentToCellWeights(
      {0.54066205, -0.32668582, -0.43562764, -0.56094903, 0.42957711,
       0.01841056, -0.32764608, -0.33027974, -0.10826075, 0.20675004,
       0.19069612, -0.03026325, -0.54532051, 0.33003211, 0.44901288,
       0.21193194});

  lstm.SetRecurrentToForgetWeights(
      {-0.13832897, -0.0515101, -0.2359007, -0.16661474, -0.14340827,
       0.36986142, 0.23414481, 0.55899, 0.10798943, -0.41174671, 0.17751795,
       -0.34484994, -0.35874045, -0.11352962, 0.27268326, 0.54058349});

  lstm.SetRecurrentToOutputWeights(
      {0.41613156, 0.42610586, -0.16495961, -0.5663873, 0.30579174, -0.05115908,
       -0.33941799, 0.23364776, 0.11178309, 0.09481031, -0.26424935, 0.46261835,
       0.50248802, 0.26114327, -0.43736315, 0.33149987});

  lstm.SetCellToForgetWeights(
      {0.47485286, -0.51955009, -0.24458408, 0.31544167});
  lstm.SetCellToOutputWeights(
      {-0.17135078, 0.82760304, 0.85573703, -0.77109635});

  // Verify the model by unpacking it.
  lstm.Verify();
}

}  // namespace
}  // namespace tflite
