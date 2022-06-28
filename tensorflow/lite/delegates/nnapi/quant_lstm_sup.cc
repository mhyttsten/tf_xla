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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_supDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_supDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_supDTcc() {
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
#include "tensorflow/lite/delegates/nnapi/quant_lstm_sup.h"

#include <algorithm>

#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegate {
namespace nnapi {

// The function extracts a submatrix of the weights at a given row
// and column offsets from  a 2D matrix
void ExtractQuantLstmWeightsSubmatrix(const TfLiteIntArray* submatrix_dims,
                                      const int32_t offset_row,
                                      const int32_t offset_column,
                                      const TfLiteIntArray* weight_dims,
                                      const uint8_t* weights,
                                      std::vector<uint8_t>* submatrix) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_supDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/delegates/nnapi/quant_lstm_sup.cc", "ExtractQuantLstmWeightsSubmatrix");

  auto const& submatrix_rows = submatrix_dims->data[0];
  auto const& submatrix_cols = submatrix_dims->data[1];
  auto const& weight_cols = weight_dims->data[1];

  submatrix->resize(NumElements(submatrix_dims));

  for (uint32_t i = 0, end = submatrix_rows * submatrix_cols; i < end; ++i) {
    const uint32_t row = i / submatrix_cols;
    const uint32_t column = i % submatrix_cols;
    (*submatrix)[i] =
        weights[(row + offset_row) * weight_cols + column + offset_column];
  }
}

inline int OutputDepth(const TfLiteIntArray* weight_dims) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_supDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/delegates/nnapi/quant_lstm_sup.cc", "OutputDepth");

  return weight_dims->data[0] / 4;
}

inline int InputDepth(const TfLiteIntArray* weight_dims) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_supDTcc mht_2(mht_2_v, 226, "", "./tensorflow/lite/delegates/nnapi/quant_lstm_sup.cc", "InputDepth");

  return weight_dims->data[1] - OutputDepth(weight_dims);
}

void SetWeightSubmatrixDims(const TfLiteIntArray* weight_dims,
                            TfLiteIntArray* recurrent_submatrix_dims,
                            TfLiteIntArray* input_submatrix_dims) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_supDTcc mht_3(mht_3_v, 235, "", "./tensorflow/lite/delegates/nnapi/quant_lstm_sup.cc", "SetWeightSubmatrixDims");

  const auto input_depth = InputDepth(weight_dims);
  const auto output_depth = OutputDepth(weight_dims);

  recurrent_submatrix_dims->data[0] = output_depth;
  recurrent_submatrix_dims->data[1] = output_depth;

  input_submatrix_dims->data[0] = output_depth;
  input_submatrix_dims->data[1] = input_depth;
}

// Doing exactly the opposite work of QuantizedLSTMCell::concatenateWeights
// in NNAPI, decomposing the concat_weights tensor data into its 8 components
// according to the following diagram
//
// +-----------------------------------+
// | recurrentToInput  | inputToInput  |
// |-------------------+---------------|
// | recurrentToCell   | inputToCell   |
// |-------------------+---------------|
// | recurrentToForget | inputToForget |
// |-------------------+---------------|
// | recurrentToOutput | inputToOutput |
// +-----------------------------------+
void DecomposeQuantLstmWeightsTensor(const uint8_t* concat_weights,
                                     const TfLiteIntArray* weight_dims,
                                     std::vector<uint8_t>* recurrent_to_input,
                                     std::vector<uint8_t>* input_to_input,
                                     std::vector<uint8_t>* recurrent_to_cell,
                                     std::vector<uint8_t>* input_to_cell,
                                     std::vector<uint8_t>* recurrent_to_forget,
                                     std::vector<uint8_t>* input_to_forget,
                                     std::vector<uint8_t>* recurrent_to_output,
                                     std::vector<uint8_t>* input_to_output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_supDTcc mht_4(mht_4_v, 271, "", "./tensorflow/lite/delegates/nnapi/quant_lstm_sup.cc", "DecomposeQuantLstmWeightsTensor");

  const auto output_depth = OutputDepth(weight_dims);

  TfLiteIntArray* recurrent_submatrix_dims = TfLiteIntArrayCreate(2);
  TfLiteIntArray* input_submatrix_dims = TfLiteIntArrayCreate(2);
  SetWeightSubmatrixDims(weight_dims, recurrent_submatrix_dims,
                         input_submatrix_dims);

  ExtractQuantLstmWeightsSubmatrix(recurrent_submatrix_dims, 0 * output_depth,
                                   0, weight_dims, concat_weights,
                                   recurrent_to_input);
  ExtractQuantLstmWeightsSubmatrix(input_submatrix_dims, 0 * output_depth,
                                   output_depth, weight_dims, concat_weights,
                                   input_to_input);

  ExtractQuantLstmWeightsSubmatrix(recurrent_submatrix_dims, 1 * output_depth,
                                   0, weight_dims, concat_weights,
                                   recurrent_to_cell);
  ExtractQuantLstmWeightsSubmatrix(input_submatrix_dims, 1 * output_depth,
                                   output_depth, weight_dims, concat_weights,
                                   input_to_cell);

  ExtractQuantLstmWeightsSubmatrix(recurrent_submatrix_dims, 2 * output_depth,
                                   0, weight_dims, concat_weights,
                                   recurrent_to_forget);
  ExtractQuantLstmWeightsSubmatrix(input_submatrix_dims, 2 * output_depth,
                                   output_depth, weight_dims, concat_weights,
                                   input_to_forget);

  ExtractQuantLstmWeightsSubmatrix(recurrent_submatrix_dims, 3 * output_depth,
                                   0, weight_dims, concat_weights,
                                   recurrent_to_output);
  ExtractQuantLstmWeightsSubmatrix(input_submatrix_dims, 3 * output_depth,
                                   output_depth, weight_dims, concat_weights,
                                   input_to_output);

  TfLiteIntArrayFree(recurrent_submatrix_dims);
  TfLiteIntArrayFree(input_submatrix_dims);
}

void DecomposeBiasTensor(const int32_t* biases, int bias_size,
                         std::vector<int32_t>* input_bias,
                         std::vector<int32_t>* cell_bias,
                         std::vector<int32_t>* forget_bias,
                         std::vector<int32_t>* output_bias) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSquant_lstm_supDTcc mht_5(mht_5_v, 318, "", "./tensorflow/lite/delegates/nnapi/quant_lstm_sup.cc", "DecomposeBiasTensor");

  input_bias->resize(bias_size);
  std::copy(biases, biases + bias_size, input_bias->begin());

  cell_bias->resize(bias_size);
  std::copy(biases + bias_size, biases + 2 * bias_size, cell_bias->begin());

  forget_bias->resize(bias_size);
  std::copy(biases + 2 * bias_size, biases + 3 * bias_size,
            forget_bias->begin());

  output_bias->resize(bias_size);
  std::copy(biases + 3 * bias_size, biases + 4 * bias_size,
            output_bias->begin());
}

}  // namespace nnapi
}  // namespace delegate
}  // namespace tflite
