/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_REDUCE_TEST_COMMON_H_
#define TENSORFLOW_LITE_KERNELS_REDUCE_TEST_COMMON_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSreduce_test_commonDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduce_test_commonDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSreduce_test_commonDTh() {
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


#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
class BaseOpModel : public SingleOpModel {
 public:
  void SetAxis(const std::vector<int>& data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduce_test_commonDTh mht_0(mht_0_v, 193, "", "./tensorflow/lite/kernels/reduce_test_common.h", "SetAxis");
 PopulateTensor(axis_, data); }

  template <class T>
  void SetInput(std::vector<T> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduce_test_commonDTh mht_1(mht_1_v, 199, "", "./tensorflow/lite/kernels/reduce_test_common.h", "SetInput");

    PopulateTensor(input_, data);
  }

  template <class T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  int Input() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduce_test_commonDTh mht_2(mht_2_v, 219, "", "./tensorflow/lite/kernels/reduce_test_common.h", "Input");
 return input_; }

 protected:
  void SymmetricInt16Scaling(TensorData& tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduce_test_commonDTh mht_3(mht_3_v, 225, "", "./tensorflow/lite/kernels/reduce_test_common.h", "SymmetricInt16Scaling");

    // Symmetric range and null zero-point is required for INT16 tensors. As
    // SingleOpModel::QuantizationParams calculates the scale on an asymmetric
    // base [int_type::min, int_type::max], manually calculate the scale on a
    // symmetric range [int_type::min+1, int_type::max] to ensure a null
    // zero-point.
    if (tensor.type == TensorType_INT16) {
      CHECK_EQ(std::abs(tensor.min), tensor.max);
      tensor.scale = tensor.max / std::numeric_limits<int16_t>::max();
      tensor.zero_point = 0;
      tensor.min = 0;
      tensor.max = 0;
    }
  }

 protected:
  int input_;
  int axis_;
  int output_;
};

// Model for the tests case where axis is a const tensor.
template <BuiltinOperator op_code, bool symmetric_int16_scaling = false>
class BaseConstOpModel : public BaseOpModel {
 public:
  BaseConstOpModel(TensorData input, TensorData output,
                   std::initializer_list<int> axis_shape,
                   std::initializer_list<int> axis, bool keep_dims) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduce_test_commonDTh mht_4(mht_4_v, 255, "", "./tensorflow/lite/kernels/reduce_test_common.h", "BaseConstOpModel");

    if (symmetric_int16_scaling) {
      SymmetricInt16Scaling(input);
      SymmetricInt16Scaling(output);
    }
    input_ = AddInput(input);
    axis_ = AddConstInput(TensorType_INT32, axis, axis_shape);
    output_ = AddOutput(output);
    SetBuiltinOp(op_code, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a dynamic tensor.
template <BuiltinOperator op_code, bool symmetric_int16_scaling = false>
class BaseDynamicOpModel : public BaseOpModel {
 public:
  BaseDynamicOpModel(TensorData input, TensorData output,
                     const TensorData& axis, bool keep_dims) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduce_test_commonDTh mht_5(mht_5_v, 277, "", "./tensorflow/lite/kernels/reduce_test_common.h", "BaseDynamicOpModel");

    if (symmetric_int16_scaling) {
      SymmetricInt16Scaling(input);
      SymmetricInt16Scaling(output);
    }
    input_ = AddInput(input);
    axis_ = AddInput(axis);
    output_ = AddOutput(output);
    SetBuiltinOp(op_code, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_REDUCE_TEST_COMMON_H_
