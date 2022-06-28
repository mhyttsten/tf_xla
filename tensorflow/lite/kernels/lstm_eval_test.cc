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
class MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc() {
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
#include "tensorflow/lite/kernels/lstm_eval.h"

#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"

namespace tflite {
namespace {

// Validate result.
template <typename T>
bool ArrayEq(const T* result, const T* expected_result, int size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "ArrayEq");

  for (int i = 0; i < size; ++i) {
    if (result[i] != expected_result[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool ArrayFloatNear(const T* result, const T* expected_result, int size,
                    double threshold) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "ArrayFloatNear");

  for (int i = 0; i < size; ++i) {
    if (std::abs(result[i] - expected_result[i]) > threshold) {
      return false;
    }
  }
  return true;
}

// Base class that holds input parameters for quantized and hybrid lstm.
class BaseLstmParam {
 public:
  TfLiteTensor* Geti2i() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "Geti2i");

    PackWeightToTensor(&i2i_tensor_, i2i_, i2i_size_);
    i2i_tensor_.data.int8 = i2i_.data();
    return &i2i_tensor_;
  }
  TfLiteTensor* Geti2f() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_3(mht_3_v, 240, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "Geti2f");

    PackWeightToTensor(&i2f_tensor_, i2f_, i2f_size_);
    i2f_tensor_.data.int8 = i2f_.data();
    return &i2f_tensor_;
  }
  TfLiteTensor* Geti2c() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_4(mht_4_v, 248, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "Geti2c");

    PackWeightToTensor(&i2c_tensor_, i2c_, i2c_size_);
    i2c_tensor_.data.int8 = i2c_.data();
    return &i2c_tensor_;
  }
  TfLiteTensor* Geti2o() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_5(mht_5_v, 256, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "Geti2o");

    PackWeightToTensor(&i2o_tensor_, i2o_, i2o_size_);
    i2o_tensor_.data.int8 = i2o_.data();
    return &i2o_tensor_;
  }
  TfLiteTensor* Getr2i() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_6(mht_6_v, 264, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "Getr2i");

    PackWeightToTensor(&r2i_tensor_, r2i_, r2i_size_);
    r2i_tensor_.data.int8 = r2i_.data();
    return &r2i_tensor_;
  }
  TfLiteTensor* Getr2f() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_7(mht_7_v, 272, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "Getr2f");

    PackWeightToTensor(&r2f_tensor_, r2f_, r2f_size_);
    r2f_tensor_.data.int8 = r2f_.data();
    return &r2f_tensor_;
  }
  TfLiteTensor* Getr2c() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_8(mht_8_v, 280, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "Getr2c");

    PackWeightToTensor(&r2c_tensor_, r2c_, r2c_size_);
    r2c_tensor_.data.int8 = r2c_.data();
    return &r2c_tensor_;
  }
  TfLiteTensor* Getr2o() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_9(mht_9_v, 288, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "Getr2o");

    PackWeightToTensor(&r2o_tensor_, r2o_, r2o_size_);
    r2o_tensor_.data.int8 = r2o_.data();
    return &r2o_tensor_;
  }
  TfLiteTensor* GetProjection() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_10(mht_10_v, 296, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetProjection");

    PackWeightToTensor(&projection_tensor_, projection_, projection_size_);
    projection_tensor_.data.int8 = projection_.data();
    return &projection_tensor_;
  }
  ~BaseLstmParam() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_11(mht_11_v, 304, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "~BaseLstmParam");

    TfLiteIntArrayFree(input_tensor_.dims);
    TfLiteIntArrayFree(i2i_tensor_.dims);
    TfLiteIntArrayFree(i2f_tensor_.dims);
    TfLiteIntArrayFree(i2c_tensor_.dims);
    TfLiteIntArrayFree(i2o_tensor_.dims);
    TfLiteIntArrayFree(r2i_tensor_.dims);
    TfLiteIntArrayFree(r2f_tensor_.dims);
    TfLiteIntArrayFree(r2c_tensor_.dims);
    TfLiteIntArrayFree(r2o_tensor_.dims);
    TfLiteIntArrayFree(layer_norm_input_tensor_.dims);
    TfLiteIntArrayFree(layer_norm_forget_tensor_.dims);
    TfLiteIntArrayFree(layer_norm_cell_tensor_.dims);
    TfLiteIntArrayFree(layer_norm_output_tensor_.dims);
    TfLiteIntArrayFree(input_gate_bias_tensor_.dims);
    TfLiteIntArrayFree(forget_gate_bias_tensor_.dims);
    TfLiteIntArrayFree(cell_gate_bias_tensor_.dims);
    TfLiteIntArrayFree(output_gate_bias_tensor_.dims);
    TfLiteIntArrayFree(projection_tensor_.dims);
    TfLiteIntArrayFree(projection_bias_tensor_.dims);
    TfLiteIntArrayFree(activation_tensor_.dims);
    TfLiteIntArrayFree(cell_tensor_.dims);
    TfLiteIntArrayFree(output_tensor_.dims);
  }

 protected:
  template <typename T>
  void PackWeightToTensor(TfLiteTensor* tensor, std::vector<T>& data,
                          std::vector<int32_t> dims) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_12(mht_12_v, 335, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "PackWeightToTensor");

    if (data.empty()) {
      int total = 1;
      for (int i = 0; i < dims.size(); ++i) {
        total *= dims[i];
      }
      for (int i = 0; i < total; ++i) {
        data.push_back(0);
      }
    }
    tensor->dims = TfLiteIntArrayCreate(dims.size());
    for (int i = 0; i < dims.size(); ++i) {
      tensor->dims->data[i] = dims[i];
    }
  }
  // Dimensions. Need proper size to trigger neon code.
  const int n_batch_ = 2;
  const int n_input_ = 18;
  const int n_cell_ = 10;
  const int n_output_ = 6;

  std::vector<int32_t> input_size_ = {n_batch_, n_input_};
  TfLiteTensor input_tensor_;

  // input_to_input_weights.
  std::vector<int8_t> i2i_ = {
      18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
      1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  0,   //
      8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6, 1, 2, 3, -4, 5,  6,   //
      1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6, 1, 7, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
      1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 8,  5,  -6,  //
      8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
      1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6, 1, 2, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  6, 1, 2, 3, 14, 5,  6,   //
      1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6, 1, 2, 3, 4,  5,  6,   //
  };
  std::vector<int32_t> i2i_size_ = {n_cell_, n_input_};
  TfLiteTensor i2i_tensor_;

  // input_to_forget_weights.
  std::vector<int8_t> i2f_ = {
      1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  0,   //
      8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6,  1,  2, 3, -4, 5,  6,   //
      1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6,  1,  7, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
      1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6,  11, 2, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  -6, 1,  2, 3, 14, 5,  6,   //
      1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
      18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
      8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6,  13, 2, 3, 4,  5,  6,   //
      1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 8,  5,  -6,  //
  };
  std::vector<int32_t> i2f_size_ = {n_cell_, n_input_};
  TfLiteTensor i2f_tensor_;

  // input_to_cell_weights.
  std::vector<int8_t> i2c_ = {
      1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  0,   //
      1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6,  1, 2, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  16, 1, 2, 3, 14, 5,  6,   //
      1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6,  7, 2, 3, 4,  5,  6,   //
      18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  6,   //
      8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  6,   //
      1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1, 2, 3, 8,  5,  -6,  //
      8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6,  1, 2, 3, -4, 5,  6,   //
      1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6,  1, 7, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1, 2, 3, 4,  5,  6,   //
  };
  std::vector<int32_t> i2c_size_ = {n_cell_, n_input_};
  TfLiteTensor i2c_tensor_;

  // input_to_output_weights.
  std::vector<int8_t> i2o_ = {
      1,  2,  3, 4,  5, 6, 1, 2,  3, 4, -5, 6,  1,  7, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  6,  -1, 2, 3, 4,  5,  6,   //
      1,  2,  3, 4,  3, 6, 1, 2,  6, 4, 5,  6,  1,  2, 3, 4,  -5, 6,   //
      8,  2,  3, 4,  5, 6, 7, 2,  3, 4, 5,  6,  1,  2, 3, 14, 5,  6,   //
      18, 2,  3, 4,  5, 6, 1, 2,  3, 4, 5,  -6, 1,  2, 3, 4,  5,  6,   //
      8,  2,  3, 4,  5, 6, 3, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  6,   //
      1,  2,  3, 4,  5, 6, 5, 2,  3, 4, 5,  6,  1,  2, 3, 4,  5,  0,   //
      8,  2,  3, 4,  3, 6, 1, -2, 3, 4, 5,  6,  1,  2, 3, -4, 5,  6,   //
      1,  2,  3, -4, 5, 6, 1, 2,  3, 4, 5,  6,  -1, 2, 3, 4,  5,  6,   //
      1,  -2, 2, 4,  5, 6, 1, 2,  3, 4, 5,  6,  1,  2, 3, 8,  5,  -6,  //
  };
  std::vector<int32_t> i2o_size_ = {n_cell_, n_input_};
  TfLiteTensor i2o_tensor_;

  // recurrent_to_input_weights.
  std::vector<int8_t> r2i_ = {
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
  };
  std::vector<int32_t> r2i_size_ = {n_cell_, n_output_};
  TfLiteTensor r2i_tensor_;

  // recurrent_to_forget_weights.
  std::vector<int8_t> r2f_ = {
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
  };
  std::vector<int32_t> r2f_size_ = {n_cell_, n_output_};
  TfLiteTensor r2f_tensor_;

  // recurrent_to_cell_weights.
  std::vector<int8_t> r2c_ = {
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
  };
  std::vector<int32_t> r2c_size_ = {n_cell_, n_output_};
  TfLiteTensor r2c_tensor_;

  // recurrent_to_output_weights.
  std::vector<int8_t> r2o_ = {
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
  };
  std::vector<int32_t> r2o_size_ = {n_cell_, n_output_};
  TfLiteTensor r2o_tensor_;

  std::vector<int32_t> layer_norm_input_size_ = {n_cell_};
  TfLiteTensor layer_norm_input_tensor_;

  TfLiteTensor layer_norm_forget_tensor_;
  std::vector<int32_t> layer_norm_forget_size_ = {n_cell_};

  std::vector<int32_t> layer_norm_cell_size_ = {n_cell_};
  TfLiteTensor layer_norm_cell_tensor_;

  std::vector<int32_t> layer_norm_output_size_ = {n_cell_};
  TfLiteTensor layer_norm_output_tensor_;

  std::vector<int32_t> input_gate_bias_size_ = {n_cell_};
  TfLiteTensor input_gate_bias_tensor_;

  std::vector<int32_t> forget_gate_bias_size_ = {n_cell_};
  TfLiteTensor forget_gate_bias_tensor_;

  std::vector<int32_t> cell_gate_bias_size_ = {n_cell_};
  TfLiteTensor cell_gate_bias_tensor_;

  std::vector<int32_t> output_gate_bias_size_ = {n_cell_};
  TfLiteTensor output_gate_bias_tensor_;

  // projection_weights.
  std::vector<int8_t> projection_ = {
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
      8, 2, 3, 4, 5, 6, 1, 2,  3,  4,  //
      6, 4, 5, 6, 1, 2, 3, 4,  -5, 6,  //
      1, 2, 3, 4, 7, 3, 4, -5, 6,  3,  //
  };
  std::vector<int32_t> projection_size_ = {n_cell_, n_output_};
  TfLiteTensor projection_tensor_;

  // projection_bias.
  std::vector<int32_t> projection_bias_ = {
      16, 4, 5, 6, 1, 1  //
  };

  std::vector<int32_t> projection_bias_size_ = {n_output_};
  TfLiteTensor projection_bias_tensor_;

  std::vector<int32_t> activation_size_ = {n_batch_, n_output_};
  TfLiteTensor activation_tensor_;

  std::vector<int32_t> cell_size_ = {n_batch_, n_cell_};
  TfLiteTensor cell_tensor_;

  std::vector<int32_t> output_size_ = {n_batch_, n_output_};
  TfLiteTensor output_tensor_;
};

class QuantizedLstmParam : public BaseLstmParam {
 public:
  // Getter methods.
  TfLiteTensor* GetInput() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_13(mht_13_v, 531, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetInput");

    PackWeightToTensor(&input_tensor_, input_, input_size_);
    input_tensor_.data.int8 = input_.data();
    return &input_tensor_;
  }
  TfLiteTensor* GetInputLayerNorm() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_14(mht_14_v, 539, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetInputLayerNorm");

    PackWeightToTensor(&layer_norm_input_tensor_, layer_norm_input_,
                       layer_norm_input_size_);
    layer_norm_input_tensor_.data.i16 = layer_norm_input_.data();
    return &layer_norm_input_tensor_;
  }
  TfLiteTensor* GetForgetLayerNorm() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_15(mht_15_v, 548, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetForgetLayerNorm");

    PackWeightToTensor(&layer_norm_forget_tensor_, layer_norm_forget_,
                       layer_norm_forget_size_);
    layer_norm_forget_tensor_.data.i16 = layer_norm_forget_.data();
    return &layer_norm_forget_tensor_;
  }
  TfLiteTensor* GetCellLayerNorm() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_16(mht_16_v, 557, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetCellLayerNorm");

    PackWeightToTensor(&layer_norm_cell_tensor_, layer_norm_cell_,
                       layer_norm_cell_size_);
    layer_norm_cell_tensor_.data.i16 = layer_norm_cell_.data();
    return &layer_norm_cell_tensor_;
  }
  TfLiteTensor* GetOutputLayerNorm() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_17(mht_17_v, 566, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetOutputLayerNorm");

    PackWeightToTensor(&layer_norm_output_tensor_, layer_norm_output_,
                       layer_norm_output_size_);
    layer_norm_output_tensor_.data.i16 = layer_norm_output_.data();
    return &layer_norm_output_tensor_;
  }
  TfLiteTensor* GetInputBias() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_18(mht_18_v, 575, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetInputBias");

    PackWeightToTensor(&input_gate_bias_tensor_, input_gate_bias_,
                       input_gate_bias_size_);
    input_gate_bias_tensor_.data.i32 = input_gate_bias_.data();
    return &input_gate_bias_tensor_;
  }
  TfLiteTensor* GetForgetBias() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_19(mht_19_v, 584, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetForgetBias");

    PackWeightToTensor(&forget_gate_bias_tensor_, forget_gate_bias_,
                       forget_gate_bias_size_);
    forget_gate_bias_tensor_.data.i32 = forget_gate_bias_.data();
    return &forget_gate_bias_tensor_;
  }
  TfLiteTensor* GetCellBias() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_20(mht_20_v, 593, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetCellBias");

    PackWeightToTensor(&cell_gate_bias_tensor_, cell_gate_bias_,
                       cell_gate_bias_size_);
    cell_gate_bias_tensor_.data.i32 = cell_gate_bias_.data();
    return &cell_gate_bias_tensor_;
  }
  TfLiteTensor* GetOutputBias() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_21(mht_21_v, 602, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetOutputBias");

    PackWeightToTensor(&output_gate_bias_tensor_, output_gate_bias_,
                       output_gate_bias_size_);
    output_gate_bias_tensor_.data.i32 = output_gate_bias_.data();
    return &output_gate_bias_tensor_;
  }
  TfLiteTensor* GetProjectionBias() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_22(mht_22_v, 611, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetProjectionBias");

    PackWeightToTensor(&projection_bias_tensor_, projection_bias_,
                       projection_bias_size_);
    projection_bias_tensor_.data.i32 = projection_bias_.data();
    return &projection_bias_tensor_;
  }

  // Set up quantization parameters.
  ops::builtin::lstm_eval::IntegerLstmParameter* GetQuantParam() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_23(mht_23_v, 622, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetQuantParam");

    integer_lstm_param_.effective_input_to_input_scale_a = 1808677632;
    integer_lstm_param_.effective_input_to_input_scale_b = -1;
    integer_lstm_param_.effective_recurrent_to_input_scale_a = 1078887680;
    integer_lstm_param_.effective_recurrent_to_input_scale_b = -1;
    integer_lstm_param_.effective_cell_to_input_scale_a = 1073741824;
    integer_lstm_param_.effective_cell_to_input_scale_b = 1;
    integer_lstm_param_.effective_input_to_forget_scale_a = 1845996800;
    integer_lstm_param_.effective_input_to_forget_scale_b = -3;
    integer_lstm_param_.effective_recurrent_to_forget_scale_a = 1477412736;
    integer_lstm_param_.effective_recurrent_to_forget_scale_b = -2;
    integer_lstm_param_.effective_cell_to_forget_scale_a = 1073741824;
    integer_lstm_param_.effective_cell_to_forget_scale_b = 1;
    integer_lstm_param_.effective_input_to_cell_scale_a = 1648385408;
    integer_lstm_param_.effective_input_to_cell_scale_b = -2;
    integer_lstm_param_.effective_recurrent_to_cell_scale_a = 1185544192,
    integer_lstm_param_.effective_recurrent_to_cell_scale_b = -1;
    integer_lstm_param_.effective_input_to_output_scale_a = 1328153600;
    integer_lstm_param_.effective_input_to_output_scale_b = -1;
    integer_lstm_param_.effective_recurrent_to_output_scale_a = 1479582592;
    integer_lstm_param_.effective_recurrent_to_output_scale_b = -1;
    integer_lstm_param_.effective_cell_to_output_scale_a = 1073741824,
    integer_lstm_param_.effective_cell_to_output_scale_b = 1;
    integer_lstm_param_.effective_proj_scale_a = 1105682560;
    integer_lstm_param_.effective_proj_scale_b = -8;
    integer_lstm_param_.effective_hidden_scale_a = 0;
    integer_lstm_param_.effective_hidden_scale_b = 0;
    integer_lstm_param_.layer_norm_input_scale_a = 2011617664;
    integer_lstm_param_.layer_norm_input_scale_b = -11;
    integer_lstm_param_.layer_norm_forget_scale_a = 1968024960;
    integer_lstm_param_.layer_norm_forget_scale_b = -13;
    integer_lstm_param_.layer_norm_cell_scale_a = 1097334528,
    integer_lstm_param_.layer_norm_cell_scale_b = -12;
    integer_lstm_param_.layer_norm_output_scale_a = 1837163008;
    integer_lstm_param_.layer_norm_output_scale_b = -12;
    integer_lstm_param_.quantized_cell_clip = 20480;
    integer_lstm_param_.quantized_proj_clip = 0;
    integer_lstm_param_.cell_scale = -11;
    integer_lstm_param_.input_variance_guard = 1;
    integer_lstm_param_.forget_variance_guard = 2;
    integer_lstm_param_.cell_variance_guard = 2;
    integer_lstm_param_.output_variance_guard = 1;
    integer_lstm_param_.hidden_zp = 0;
    integer_lstm_param_.input_to_forget_effective_bias.reset(
        new int32_t[n_cell_]);
    integer_lstm_param_.recurrent_to_forget_effective_bias.reset(
        new int32_t[n_cell_]);
    integer_lstm_param_.input_to_cell_effective_bias.reset(
        new int32_t[n_cell_]);
    integer_lstm_param_.recurrent_to_cell_effective_bias.reset(
        new int32_t[n_cell_]);
    integer_lstm_param_.input_to_output_effective_bias.reset(
        new int32_t[n_cell_]);
    integer_lstm_param_.recurrent_to_output_effective_bias.reset(
        new int32_t[n_cell_]);
    integer_lstm_param_.input_to_input_effective_bias.reset(
        new int32_t[n_cell_]);
    integer_lstm_param_.recurrent_to_input_effective_bias.reset(
        new int32_t[n_cell_]);
    integer_lstm_param_.projection_effective_bias.reset(new int32_t[n_output_]);
    std::fill_n(integer_lstm_param_.input_to_forget_effective_bias.get(),
                n_cell_, 152);
    std::fill_n(integer_lstm_param_.recurrent_to_forget_effective_bias.get(),
                n_cell_, 315);
    std::fill_n(integer_lstm_param_.input_to_cell_effective_bias.get(), n_cell_,
                165);
    std::fill_n(integer_lstm_param_.recurrent_to_cell_effective_bias.get(),
                n_cell_, 1165);
    std::fill_n(integer_lstm_param_.input_to_output_effective_bias.get(),
                n_cell_, 159);
    std::fill_n(integer_lstm_param_.recurrent_to_output_effective_bias.get(),
                n_cell_, 915);
    std::fill_n(integer_lstm_param_.input_to_input_effective_bias.get(),
                n_cell_, -15);
    std::fill_n(integer_lstm_param_.recurrent_to_input_effective_bias.get(),
                n_cell_, 315);
    std::fill_n(integer_lstm_param_.projection_effective_bias.get(), n_output_,
                115);
    return &integer_lstm_param_;
  }

  // Create scratch buffers.
  TfLiteTensor* GetScratch0() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_24(mht_24_v, 707, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetScratch0");

    PackWeightToTensor(&scratch0_tensor_, scratch0_, scratch0_size_);
    scratch0_tensor_.data.i16 = scratch0_.data();
    return &scratch0_tensor_;
  }
  TfLiteTensor* GetScratch1() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_25(mht_25_v, 715, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetScratch1");

    PackWeightToTensor(&scratch1_tensor_, scratch1_, scratch1_size_);
    scratch1_tensor_.data.i16 = scratch1_.data();
    return &scratch1_tensor_;
  }
  TfLiteTensor* GetScratch2() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_26(mht_26_v, 723, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetScratch2");

    PackWeightToTensor(&scratch2_tensor_, scratch2_, scratch2_size_);
    scratch2_tensor_.data.i16 = scratch2_.data();
    return &scratch2_tensor_;
  }
  TfLiteTensor* GetScratch3() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_27(mht_27_v, 731, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetScratch3");

    PackWeightToTensor(&scratch3_tensor_, scratch3_, scratch3_size_);
    scratch3_tensor_.data.i16 = scratch3_.data();
    return &scratch3_tensor_;
  }
  TfLiteTensor* GetScratch4() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_28(mht_28_v, 739, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetScratch4");

    PackWeightToTensor(&scratch4_tensor_, scratch4_, scratch4_size_);
    scratch4_tensor_.data.int8 = scratch4_.data();
    return &scratch4_tensor_;
  }
  TfLiteTensor* GetScratch5() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_29(mht_29_v, 747, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetScratch5");

    PackWeightToTensor(&scratch5_tensor_, scratch5_, scratch5_size_);
    scratch5_tensor_.data.i32 = scratch5_.data();
    return &scratch5_tensor_;
  }
  TfLiteTensor* GetActivation() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_30(mht_30_v, 755, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetActivation");

    PackWeightToTensor(&activation_tensor_, activation_, activation_size_);
    activation_tensor_.data.int8 = activation_.data();
    activation_tensor_.params.zero_point = 50;
    return &activation_tensor_;
  }
  TfLiteTensor* GetOutput() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_31(mht_31_v, 764, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetOutput");

    PackWeightToTensor(&output_tensor_, output_, output_size_);
    output_tensor_.data.int8 = output_.data();
    return &output_tensor_;
  }
  TfLiteTensor* GetCell() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_32(mht_32_v, 772, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetCell");

    PackWeightToTensor(&cell_tensor_, cell_, cell_size_);
    cell_tensor_.data.i16 = cell_.data();
    return &cell_tensor_;
  }
  ~QuantizedLstmParam() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_33(mht_33_v, 780, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "~QuantizedLstmParam");

    TfLiteIntArrayFree(scratch0_tensor_.dims);
    TfLiteIntArrayFree(scratch1_tensor_.dims);
    TfLiteIntArrayFree(scratch2_tensor_.dims);
    TfLiteIntArrayFree(scratch3_tensor_.dims);
    TfLiteIntArrayFree(scratch4_tensor_.dims);
    TfLiteIntArrayFree(scratch5_tensor_.dims);
  }

 private:
  // input.
  std::vector<int8_t> input_ = {
      8, 2, 3,  4, 5, 6, 1, -2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,  //
      1, 2, -3, 4, 5, 6, 1, 2,  3, 4, 5, 6, 1, 2, 3, 4, 5, 6,  //
  };

  std::vector<int16_t> layer_norm_input_ = {8, 2, 3, 4, 5, 6, 1, 2, 3, 4};

  // forget_layer_norm_coefficient.
  std::vector<int16_t> layer_norm_forget_ = {
      1, 2, 3, 4, 7, 3, 4, -5, 6, 3,  //
  };

  // cell_layer_norm_coefficients.
  std::vector<int16_t> layer_norm_cell_ = {
      6, 4, 5, 6, 1, 2, 3, 4, -5, 6,  //
  };

  // output_layer_norm_coefficients.
  std::vector<int16_t> layer_norm_output_ = {
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };

  // input_gate_bias.
  std::vector<int32_t> input_gate_bias_ = {
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };

  // forget_gate_bias.
  std::vector<int32_t> forget_gate_bias_ = {
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };

  // cell_gate_bias.
  std::vector<int32_t> cell_gate_bias_ = {
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };

  // output_gate_bias.
  std::vector<int32_t> output_gate_bias_ = {
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };

  // activation.
  std::vector<int8_t> activation_;

  // cell.
  std::vector<int16_t> cell_ = {
      16, 4,  5, 6, 1, 1, 3, 4, -5, 6,  //
      1,  14, 5, 6, 1, 1, 3, 4, -5, 6,  //
  };

  // output.
  std::vector<int8_t> output_ = {
      1, 1, 3, 4, -5, 6,  //
      1, 4, 3, 4, -5, 6,  //
  };

  // quantized_lstm_param
  ops::builtin::lstm_eval::IntegerLstmParameter integer_lstm_param_;

  // 5 scratch buffers.
  std::vector<int16_t> scratch0_;
  std::vector<int32_t> scratch0_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch0_tensor_;
  std::vector<int16_t> scratch1_;
  std::vector<int32_t> scratch1_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch1_tensor_;
  std::vector<int16_t> scratch2_;
  std::vector<int32_t> scratch2_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch2_tensor_;
  std::vector<int16_t> scratch3_;
  std::vector<int32_t> scratch3_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch3_tensor_;
  std::vector<int8_t> scratch4_;
  std::vector<int32_t> scratch4_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch4_tensor_;
  std::vector<int32_t> scratch5_;
  std::vector<int32_t> scratch5_size_ = {n_batch_, n_cell_};
  TfLiteTensor scratch5_tensor_;
};

void TestOneFullyQuantizedLSTM() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_34(mht_34_v, 875, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "TestOneFullyQuantizedLSTM");

  CpuBackendContext context;
  QuantizedLstmParam one_parameter;
  auto activation = one_parameter.GetActivation();
  auto output = one_parameter.GetOutput();
  auto cell = one_parameter.GetCell();
  auto param = one_parameter.GetQuantParam();
  ops::builtin::lstm_eval::EvalInteger8x8_16(
      one_parameter.GetInput(), one_parameter.Geti2i(), one_parameter.Geti2f(),
      one_parameter.Geti2c(), one_parameter.Geti2o(), one_parameter.Getr2i(),
      one_parameter.Getr2f(), one_parameter.Getr2c(), one_parameter.Getr2o(),
      nullptr, nullptr, nullptr, one_parameter.GetInputLayerNorm(),
      one_parameter.GetForgetLayerNorm(), one_parameter.GetCellLayerNorm(),
      one_parameter.GetOutputLayerNorm(), one_parameter.GetInputBias(),
      one_parameter.GetForgetBias(), one_parameter.GetCellBias(),
      one_parameter.GetOutputBias(), one_parameter.GetProjection(),
      one_parameter.GetProjectionBias(), nullptr, /*forward_sequence=*/true,
      /*time_major=*/true, param, activation, cell, output,
      one_parameter.GetScratch0(), one_parameter.GetScratch1(),
      one_parameter.GetScratch2(), one_parameter.GetScratch3(),
      one_parameter.GetScratch4(), one_parameter.GetScratch5(), &context);

  // Verify results.
  const std::vector<int16_t> expected_cell = {
      7, 1, 3, 2, 0, 1, 0, 2, -2, 4, 1, 6, 4, 3, 0, 1, 0, 2, -2, 4,
  };
  const std::vector<int8_t> expected_activation = {
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
  };
  EXPECT_TRUE(ArrayEq(cell->data.i16, expected_cell.data(), 20));
  EXPECT_TRUE(ArrayEq(activation->data.int8, expected_activation.data(), 12));
  EXPECT_TRUE(ArrayEq(output->data.int8, expected_activation.data(), 12));
}

TEST(TestOneFullyQuantizedLSTM, TestOneFullyQuantizedLSTM) {
  TestOneFullyQuantizedLSTM();
}

class HybridLstmParam : public BaseLstmParam {
 public:
  TfLiteTensor* GetFloatOutput() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_35(mht_35_v, 918, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetFloatOutput");

    PackWeightToTensor(&output_tensor_, output_float_, output_size_);
    output_tensor_.data.f = output_float_.data();
    return &output_tensor_;
  }
  const TfLiteLSTMParams GetLSTMParam() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_36(mht_36_v, 926, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetLSTMParam");

    return {kTfLiteActRelu, 0, 0, kTfLiteLSTMFullKernel, true};
  }
  TfLiteTensor* GetScratchBuffer() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_37(mht_37_v, 932, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetScratchBuffer");

    PackWeightToTensor(&scratch_buffer_tensor_, scratch_buffer_,
                       scratch_buffer_size_);
    scratch_buffer_tensor_.data.f = scratch_buffer_.data();
    return &scratch_buffer_tensor_;
  }
  TfLiteTensor* GetInputScalingFactors() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_38(mht_38_v, 941, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetInputScalingFactors");

    PackWeightToTensor(&input_sf_tensor_, input_sf_,
                       quantization_extra_scratch_buffer_sizes_);
    input_sf_tensor_.data.f = input_sf_.data();
    return &input_sf_tensor_;
  }
  TfLiteTensor* GetAuxInputScalingFactors() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_39(mht_39_v, 950, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetAuxInputScalingFactors");

    PackWeightToTensor(&aux_input_sf_tensor_, aux_input_sf_,
                       quantization_extra_scratch_buffer_sizes_);
    aux_input_sf_tensor_.data.f = aux_input_sf_.data();
    return &aux_input_sf_tensor_;
  }
  TfLiteTensor* GetOutputStateScalingFactors() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_40(mht_40_v, 959, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetOutputStateScalingFactors");

    PackWeightToTensor(&output_state_sf_tensor_, output_state_sf_,
                       quantization_extra_scratch_buffer_sizes_);
    output_state_sf_tensor_.data.f = output_state_sf_.data();
    return &output_state_sf_tensor_;
  }
  TfLiteTensor* GetProdScalingFactors() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_41(mht_41_v, 968, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetProdScalingFactors");

    PackWeightToTensor(&prod_scaling_factors_tensor_, prod_scaling_factors_,
                       quantization_extra_scratch_buffer_sizes_);
    prod_scaling_factors_tensor_.data.f = prod_scaling_factors_.data();
    return &prod_scaling_factors_tensor_;
  }
  TfLiteTensor* GetInputQuantized() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_42(mht_42_v, 977, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetInputQuantized");

    PackWeightToTensor(&input_quantized_tensor_, input_quantized_, input_size_);
    input_quantized_tensor_.data.int8 = input_quantized_.data();
    return &input_quantized_tensor_;
  }
  TfLiteTensor* GetActivationStateQuantized() {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_43(mht_43_v, 985, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetActivationStateQuantized");

    PackWeightToTensor(&activation_quantized_tensor_, activation_quantized_,
                       activation_size_);
    activation_quantized_tensor_.data.int8 = activation_quantized_.data();
    return &activation_quantized_tensor_;
  }
  TfLiteTensor* GetCellStateQuantized() {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_44(mht_44_v, 994, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetCellStateQuantized");

    PackWeightToTensor(&cell_quantized_tensor_, cell_quantized_, cell_size_);
    cell_quantized_tensor_.data.int8 = cell_quantized_.data();
    return &cell_quantized_tensor_;
  }
  TfLiteTensor* GetInputZeroPoints() {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_45(mht_45_v, 1002, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetInputZeroPoints");

    PackWeightToTensor(&input_zp_tensor_, input_zp_,
                       quantization_extra_scratch_buffer_sizes_);
    input_zp_tensor_.data.i32 = input_zp_.data();
    return &input_zp_tensor_;
  }
  TfLiteTensor* GetAuxInputZeroPoints() {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_46(mht_46_v, 1011, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetAuxInputZeroPoints");

    PackWeightToTensor(&aux_input_zp_tensor_, aux_input_zp_,
                       quantization_extra_scratch_buffer_sizes_);
    aux_input_zp_tensor_.data.i32 = aux_input_zp_.data();
    return &aux_input_zp_tensor_;
  }
  TfLiteTensor* GetOutputStateZeroPoints() {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_47(mht_47_v, 1020, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetOutputStateZeroPoints");

    PackWeightToTensor(&output_state_zp_tensor_, output_state_zp_,
                       quantization_extra_scratch_buffer_sizes_);
    output_state_zp_tensor_.data.i32 = output_state_zp_.data();
    return &output_state_zp_tensor_;
  }
  TfLiteTensor* GetRowSums() {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_48(mht_48_v, 1029, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetRowSums");

    PackWeightToTensor(&row_sums_tensor_, row_sums_, row_sums_size_);
    row_sums_tensor_.data.i32 = row_sums_.data();
    return &row_sums_tensor_;
  }
  TfLiteTensor* GetFloatInput() {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_49(mht_49_v, 1037, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetFloatInput");

    PackWeightToTensor(&input_tensor_, input_float_, input_size_);
    input_tensor_.data.f = input_float_.data();
    return &input_tensor_;
  }
  TfLiteTensor* GetActivation() {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_50(mht_50_v, 1045, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetActivation");

    PackWeightToTensor(&activation_tensor_, activation_state_,
                       activation_size_);
    activation_tensor_.data.f = activation_state_.data();
    return &activation_tensor_;
  }
  TfLiteTensor* GetCell() {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_51(mht_51_v, 1054, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetCell");

    PackWeightToTensor(&cell_tensor_, cell_state_, cell_size_);
    cell_tensor_.data.f = cell_state_.data();
    return &cell_tensor_;
  }
  TfLiteTensor* GetAccumScratchBuffer() {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_52(mht_52_v, 1062, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetAccumScratchBuffer");

    PackWeightToTensor(&accum_scratch_tensor_, accum_scratch_,
                       accum_scratch_size_);
    accum_scratch_tensor_.data.i32 = accum_scratch_.data();
    return &accum_scratch_tensor_;
  }
  TfLiteTensor* GetInputBias() {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_53(mht_53_v, 1071, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetInputBias");

    PackWeightToTensor(&input_gate_bias_tensor_, input_float_bias_,
                       input_gate_bias_size_);
    input_gate_bias_tensor_.data.f = input_float_bias_.data();
    return &input_gate_bias_tensor_;
  }
  TfLiteTensor* GetForgetBias() {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_54(mht_54_v, 1080, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetForgetBias");

    PackWeightToTensor(&forget_gate_bias_tensor_, forget_float_bias_,
                       forget_gate_bias_size_);
    forget_gate_bias_tensor_.data.f = forget_float_bias_.data();
    return &forget_gate_bias_tensor_;
  }
  TfLiteTensor* GetCellBias() {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_55(mht_55_v, 1089, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetCellBias");

    PackWeightToTensor(&cell_gate_bias_tensor_, cell_float_bias_,
                       cell_gate_bias_size_);
    cell_gate_bias_tensor_.data.f = cell_float_bias_.data();
    return &cell_gate_bias_tensor_;
  }
  TfLiteTensor* GetOutputBias() {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_56(mht_56_v, 1098, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetOutputBias");

    PackWeightToTensor(&output_gate_bias_tensor_, output_float_bias_,
                       output_gate_bias_size_);
    output_gate_bias_tensor_.data.f = output_float_bias_.data();
    return &output_gate_bias_tensor_;
  }
  TfLiteTensor* GetProjectionBias() {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_57(mht_57_v, 1107, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetProjectionBias");

    PackWeightToTensor(&projection_bias_tensor_, projection_float_bias_,
                       projection_bias_size_);
    projection_bias_tensor_.data.f = projection_float_bias_.data();
    return &projection_bias_tensor_;
  }
  int GetNumRowSums() {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_58(mht_58_v, 1116, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetNumRowSums");
 return n_row_sums_; }
  TfLiteTensor* GetInputLayerNorm() {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_59(mht_59_v, 1120, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetInputLayerNorm");

    PackWeightToTensor(&layer_norm_input_tensor_, layer_norm_float_input_,
                       layer_norm_input_size_);
    layer_norm_input_tensor_.data.f = layer_norm_float_input_.data();
    return &layer_norm_input_tensor_;
  }
  TfLiteTensor* GetForgetLayerNorm() {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_60(mht_60_v, 1129, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetForgetLayerNorm");

    PackWeightToTensor(&layer_norm_forget_tensor_, layer_norm_float_forget_,
                       layer_norm_forget_size_);
    layer_norm_forget_tensor_.data.f = layer_norm_float_forget_.data();
    return &layer_norm_forget_tensor_;
  }
  TfLiteTensor* GetCellLayerNorm() {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_61(mht_61_v, 1138, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetCellLayerNorm");

    PackWeightToTensor(&layer_norm_cell_tensor_, layer_norm_float_cell_,
                       layer_norm_cell_size_);
    layer_norm_cell_tensor_.data.f = layer_norm_float_cell_.data();
    return &layer_norm_cell_tensor_;
  }
  TfLiteTensor* GetOutputLayerNorm() {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_62(mht_62_v, 1147, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "GetOutputLayerNorm");

    PackWeightToTensor(&layer_norm_output_tensor_, layer_norm_float_output_,
                       layer_norm_output_size_);
    layer_norm_output_tensor_.data.f = layer_norm_float_output_.data();
    return &layer_norm_output_tensor_;
  }
  static TfLiteTensor* addScale(TfLiteTensor* t, float scale) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_63(mht_63_v, 1156, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "addScale");

    t->params.scale = scale;
    return t;
  }
  ~HybridLstmParam() {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_64(mht_64_v, 1163, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "~HybridLstmParam");

    TfLiteIntArrayFree(scratch_buffer_tensor_.dims);
    TfLiteIntArrayFree(accum_scratch_tensor_.dims);
    TfLiteIntArrayFree(input_sf_tensor_.dims);
    TfLiteIntArrayFree(aux_input_sf_tensor_.dims);
    TfLiteIntArrayFree(output_state_sf_tensor_.dims);
    TfLiteIntArrayFree(prod_scaling_factors_tensor_.dims);
    TfLiteIntArrayFree(input_quantized_tensor_.dims);
    TfLiteIntArrayFree(activation_quantized_tensor_.dims);
    TfLiteIntArrayFree(cell_quantized_tensor_.dims);
    TfLiteIntArrayFree(input_zp_tensor_.dims);
    TfLiteIntArrayFree(aux_input_zp_tensor_.dims);
    TfLiteIntArrayFree(output_state_zp_tensor_.dims);
    TfLiteIntArrayFree(row_sums_tensor_.dims);
  }

 private:
  const int n_row_sums_ = 9;  // Number of weights + 1 for projection weights.

  std::vector<float> scratch_buffer_;
  std::vector<int32_t> scratch_buffer_size_ = {n_batch_, n_cell_ * 4};
  TfLiteTensor scratch_buffer_tensor_;

  std::vector<int32_t> quantization_extra_scratch_buffer_sizes_ = {n_batch_};
  std::vector<float> input_sf_;
  TfLiteTensor input_sf_tensor_;
  std::vector<float> aux_input_sf_;
  TfLiteTensor aux_input_sf_tensor_;
  std::vector<float> output_state_sf_;
  TfLiteTensor output_state_sf_tensor_;

  std::vector<float> prod_scaling_factors_;
  TfLiteTensor prod_scaling_factors_tensor_;

  std::vector<int32_t> input_zp_;
  TfLiteTensor input_zp_tensor_;
  std::vector<int32_t> aux_input_zp_;
  TfLiteTensor aux_input_zp_tensor_;
  std::vector<int32_t> output_state_zp_;
  TfLiteTensor output_state_zp_tensor_;

  std::vector<int8_t> input_quantized_;
  TfLiteTensor input_quantized_tensor_;

  std::vector<int8_t> activation_quantized_;
  TfLiteTensor activation_quantized_tensor_;

  std::vector<int8_t> cell_quantized_;
  TfLiteTensor cell_quantized_tensor_;

  std::vector<float> cell_state_ = {
      16, 4, 5, 6, 1, 1, 3, 4, -5, 6, 1, 14, 5, 6, 1, 1, 3, 4, -5, 6,
  };

  std::vector<int32_t> row_sums_;
  std::vector<int32_t> row_sums_size_ = {n_row_sums_, n_cell_};
  TfLiteTensor row_sums_tensor_;

  std::vector<float> activation_state_;

  std::vector<int32_t> accum_scratch_;
  std::vector<int32_t> accum_scratch_size_ = {n_cell_, n_batch_};
  TfLiteTensor accum_scratch_tensor_;
  std::vector<float> output_float_ = {
      1, 1, 3, 4, -5, 6,  //
      1, 4, 3, 4, -5, 6,  //
  };
  std::vector<float> input_float_ = {
      6.06, 7.66, 7.10, 9.32, 3.85, 0.33, 7.15, 1.56, 9.54,
      5.30, 4.53, 0.19, 1.83, 4.60, 0.84, 5.08, 4.37, 9.92,  //
      4.08, 3.79, 1.17, 8.99, 0.14, 9.22, 3.18, 2.97, 7.53,
      0.59, 9.89, 9.13, 7.68, 0.63, 2.15, 4.31, 7.20, 4.09,  //
  };
  std::vector<float> input_float_bias_ = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  std::vector<float> forget_float_bias_ = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  std::vector<float> cell_float_bias_ = {
      -11, -7, -4, -5, -1, -1, -2, -3.5, -3, -4,
  };
  std::vector<float> output_float_bias_ = {0.16, 0.4, 0.5, 0.6,  0.1,
                                           0.1,  0.3, 0.4, -0.5, 0.6};
  std::vector<float> projection_float_bias_ = {0, 0, 0, 0, 0, 0};
  std::vector<float> layer_norm_float_input_ = {8, 2, 3, 4, 5, 6, 1, -2, 3, 4};
  std::vector<float> layer_norm_float_forget_ = {
      0.1, 0.2, 0.3, 0.4, 0.7, 0.3, 0.4, -0.5, 0.6, 0.3,  //
  };
  std::vector<float> layer_norm_float_cell_ = {
      0.6, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4, -0.5, 0.6,  //
  };
  std::vector<float> layer_norm_float_output_ = {
      0.6, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4, -0.5, 0.6,  //
  };
};

void TestOneHybridAsymmLSTM() {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_eval_testDTcc mht_65(mht_65_v, 1263, "", "./tensorflow/lite/kernels/lstm_eval_test.cc", "TestOneHybridAsymmLSTM");

  CpuBackendContext context;
  HybridLstmParam one_parameter;
  auto activation = one_parameter.GetActivation();
  auto output = one_parameter.GetFloatOutput();
  auto cell = one_parameter.GetCell();
  auto param = one_parameter.GetLSTMParam();
  bool compute_row_sums = true;
  constexpr float kDefaultScale = 18.0;
  ops::builtin::lstm_eval::EvalHybrid(
      one_parameter.GetFloatInput(),
      HybridLstmParam::addScale(one_parameter.Geti2i(), kDefaultScale), nullptr,
      HybridLstmParam::addScale(one_parameter.Geti2f(), kDefaultScale), nullptr,
      HybridLstmParam::addScale(one_parameter.Geti2c(), kDefaultScale), nullptr,
      HybridLstmParam::addScale(one_parameter.Geti2o(), kDefaultScale), nullptr,
      HybridLstmParam::addScale(one_parameter.Getr2i(), kDefaultScale), nullptr,
      HybridLstmParam::addScale(one_parameter.Getr2f(), kDefaultScale), nullptr,
      HybridLstmParam::addScale(one_parameter.Getr2c(), kDefaultScale), nullptr,
      HybridLstmParam::addScale(one_parameter.Getr2o(), kDefaultScale), nullptr,
      /*cell_to_input_weights=*/nullptr,
      /*cell_to_forget_weights=*/nullptr,
      /*cell_to_output_weights=*/nullptr, one_parameter.GetInputLayerNorm(),
      one_parameter.GetForgetLayerNorm(), one_parameter.GetCellLayerNorm(),
      one_parameter.GetOutputLayerNorm(),
      /*aux_input=*/nullptr,
      /*aux_input_to_input_weights=*/nullptr,
      /*aux_input_to_forget_weights=*/nullptr,
      /*aux_input_to_cell_weights=*/nullptr,
      /*aux_input_to_output_weights=*/nullptr, one_parameter.GetInputBias(),
      one_parameter.GetForgetBias(), one_parameter.GetCellBias(),
      one_parameter.GetOutputBias(),
      HybridLstmParam::addScale(one_parameter.GetProjection(), 1.0), nullptr,
      one_parameter.GetProjectionBias(), &param,
      /*forward_sequence=*/true,
      /*time_major=*/true,
      /*output_offset=*/0, one_parameter.GetScratchBuffer(),
      one_parameter.GetInputScalingFactors(),
      one_parameter.GetAuxInputScalingFactors(),
      one_parameter.GetOutputStateScalingFactors(),
      one_parameter.GetProdScalingFactors(),
      /*recovered_cell_weights=*/nullptr, one_parameter.GetInputQuantized(),
      /*aux_input_quantized=*/nullptr,
      one_parameter.GetActivationStateQuantized(),
      one_parameter.GetCellStateQuantized(), activation, cell,
      one_parameter.GetAccumScratchBuffer(), output,
      one_parameter.GetInputZeroPoints(), one_parameter.GetAuxInputZeroPoints(),
      one_parameter.GetOutputStateZeroPoints(), one_parameter.GetRowSums(),
      one_parameter.GetNumRowSums(), &compute_row_sums, &context);
  const std::vector<float> expected_cell = {
      7.83134,  1.96158, 2.18285, 3.28739,  0.483214,
      0.618206, 1.21539, 1.4052,  -3.17735, 2.24296,  //
      0.498944, 6.91104, 1.74126, 3.28993,  0.580477,
      0.489936, 1.2527,  1.50157, -3.71849, 2.76743,  //
  };
  const std::vector<float> expected_activation = {
      53.0403, 59.3623, 24.8493, 53.0403, 59.3623, 24.8493,  //
      36.7559, 57.5202, 29.7217, 36.7559, 57.5202, 29.7217,
  };
  EXPECT_TRUE(ArrayFloatNear(cell->data.f, expected_cell.data(), 20, 1e-2));
  EXPECT_TRUE(
      ArrayFloatNear(activation->data.f, expected_activation.data(), 12, 1e-4));
  EXPECT_TRUE(
      ArrayFloatNear(output->data.f, expected_activation.data(), 12, 1e-4));
}

TEST(TestOneHybridAsymmLSTM, TestOneHybridAsymmLSTM) {
  TestOneHybridAsymmLSTM();
}

}  // namespace
}  // namespace tflite
