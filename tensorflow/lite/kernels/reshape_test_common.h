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
#ifndef TENSORFLOW_LITE_KERNELS_RESHAPE_TEST_COMMON_H_
#define TENSORFLOW_LITE_KERNELS_RESHAPE_TEST_COMMON_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSreshape_test_commonDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshape_test_commonDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSreshape_test_commonDTh() {
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


#include <stdint.h>

#include <initializer_list>
#include <string>
#include <vector>

#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
// There are three ways to specify the output shape of a Reshape
// op.
enum class ShapeSpecificationType {
  // The output shape is hardcoded in the ReshapeOptions object.
  kAsReshapeOption,
  // The output shape is specified as an input tensor, which is connected to a
  // Const node, which is guaranteed not to change once inference starts. The
  // shape is also hardcoded as in kAsReshapeOption.
  kAsConstantTensor,
  // The output shape is specified as an input tensor that can change based on
  // external input. That is, the shape is not know before the inference
  // starts. The shape is also hardcoded as in kAsReshapeOption.
  kAsTensor,
};

template <typename T, typename BASE = SingleOpModel>
class ReshapeOpModel : public BASE {
 public:
  ReshapeOpModel(std::initializer_list<int> input_shape,
                 std::initializer_list<int> shape_shape,
                 std::initializer_list<int> shape_data,
                 ShapeSpecificationType shape_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshape_test_commonDTh mht_0(mht_0_v, 219, "", "./tensorflow/lite/kernels/reshape_test_common.h", "ReshapeOpModel");

    switch (shape_type) {
      case ShapeSpecificationType::kAsTensor:
        this->BuildWithTensorShape(input_shape, shape_shape, shape_data);
        break;
      case ShapeSpecificationType::kAsConstantTensor:
        this->BuildWithConstantTensorShape(input_shape, shape_shape,
                                           shape_data);
        break;
      case ShapeSpecificationType::kAsReshapeOption:
        // In this case the shape of the new shape doesn't matter. It is
        // always hardcoded as a flat vector.
        this->BuildWithHardcodedShape(input_shape, shape_data);
        break;
    }
  }

  void SetInput(std::vector<T> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshape_test_commonDTh mht_1(mht_1_v, 239, "", "./tensorflow/lite/kernels/reshape_test_common.h", "SetInput");

    this->template PopulateTensor<T>(input_, data);
  }

  void SetStringInput(std::initializer_list<string> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshape_test_commonDTh mht_2(mht_2_v, 246, "", "./tensorflow/lite/kernels/reshape_test_common.h", "SetStringInput");

    this->PopulateStringTensor(input_, data);
  }

  std::vector<T> GetOutput() {
    return this->template ExtractVector<T>(output_);
  }
  std::vector<int> GetOutputShape() { return this->GetTensorShape(output_); }

 private:
  void BuildWithHardcodedShape(std::initializer_list<int> input_shape,
                               std::initializer_list<int> shape_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshape_test_commonDTh mht_3(mht_3_v, 260, "", "./tensorflow/lite/kernels/reshape_test_common.h", "BuildWithHardcodedShape");

    input_ = this->AddInput({GetTensorType<T>(), input_shape});
    output_ = this->AddOutput(GetTensorType<T>());
    this->SetBuiltinOp(
        BuiltinOperator_RESHAPE, BuiltinOptions_ReshapeOptions,
        CreateReshapeOptions(
            this->builder_,
            this->builder_.template CreateVector<int>(shape_data))
            .Union());
    this->BuildInterpreter({this->GetShape(input_)});
  }

  void BuildWithTensorShape(std::initializer_list<int> input_shape,
                            std::initializer_list<int> shape_shape,
                            std::initializer_list<int> shape_data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshape_test_commonDTh mht_4(mht_4_v, 277, "", "./tensorflow/lite/kernels/reshape_test_common.h", "BuildWithTensorShape");

    input_ = this->AddInput({GetTensorType<T>(), input_shape});
    output_ = this->AddOutput(GetTensorType<T>());
    int shape_input_tensor = this->AddInput({TensorType_INT32, shape_shape});
    // Note how shape also appears in ReshapeOptions
    this->SetBuiltinOp(
        BuiltinOperator_RESHAPE, BuiltinOptions_ReshapeOptions,
        CreateReshapeOptions(
            this->builder_,
            this->builder_.template CreateVector<int>(shape_data))
            .Union());
    this->BuildInterpreter(
        {this->GetShape(input_), this->GetShape(shape_input_tensor)});
    if (shape_data.size() != 0) {
      this->template PopulateTensor<int32_t>(shape_input_tensor, shape_data);
    }
  }

  void BuildWithConstantTensorShape(std::initializer_list<int> input_shape,
                                    std::initializer_list<int> shape_shape,
                                    std::initializer_list<int> shape_data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshape_test_commonDTh mht_5(mht_5_v, 300, "", "./tensorflow/lite/kernels/reshape_test_common.h", "BuildWithConstantTensorShape");

    input_ = this->AddInput({GetTensorType<T>(), input_shape});
    output_ = this->AddOutput(GetTensorType<T>());
    this->AddConstInput(TensorType_INT32, shape_data, shape_shape);
    // Note how the shape also appears in the ReshapeOptions.
    this->SetBuiltinOp(
        BuiltinOperator_RESHAPE, BuiltinOptions_ReshapeOptions,
        CreateReshapeOptions(
            this->builder_,
            this->builder_.template CreateVector<int>(shape_data))
            .Union());
    this->BuildInterpreter({this->GetShape(input_)});
  }

  int input_;
  int output_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_RESHAPE_TEST_COMMON_H_
