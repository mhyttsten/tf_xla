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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPStranspose_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPStranspose_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPStranspose_testDTcc() {
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
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

class TransposeOpModel : public SingleOpModelWithHexagon {
 public:
  TransposeOpModel(const TensorData& input,
                   std::initializer_list<int> perm_shape,
                   std::initializer_list<int> perm, bool const_perm,
                   const TensorData& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPStranspose_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/hexagon/builders/tests/transpose_test.cc", "TransposeOpModel");

    input_ = AddInput(input);
    if (const_perm) {
      perm_ = AddConstInput(TensorType_INT32, perm, perm_shape);
    } else {
      perm_ = AddInput({TensorType_INT32, perm_shape});
    }
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
    if (!const_perm) {
      PopulateTensor<int>(perm_, perm);
    }
  }

  template <typename integer_type>
  void SetInput(const std::vector<integer_type>& data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPStranspose_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/delegates/hexagon/builders/tests/transpose_test.cc", "SetInput");

    PopulateTensor<integer_type>(input_, data);
  }

  template <typename integer_type>
  std::vector<integer_type> GetOutput() {
    return ExtractVector<integer_type>(output_);
  }

 protected:
  int input_;
  int perm_;
  int output_;
};

template <typename integer_type>
void ComputeExpectedTransposeResult(
    const std::vector<int>& shape, const std::vector<int>& perms,
    std::vector<integer_type>* input,
    std::vector<integer_type>* input_transposed) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPStranspose_testDTcc mht_2(mht_2_v, 237, "", "./tensorflow/lite/delegates/hexagon/builders/tests/transpose_test.cc", "ComputeExpectedTransposeResult");

  // Count elements and allocate output.
  int count = 1;
  for (auto factor : shape) count *= factor;
  input_transposed->resize(count);

  // Create the dummy data
  (*input).resize(count);
  for (int i = 0; i < count; i++) {
    (*input)[i] = i;
  }

  // Make input and output shapes.
  const RuntimeShape input_shape = ::tflite::GetTensorShape(shape);
  RuntimeShape output_shape(perms.size());
  for (int i = 0; i < perms.size(); i++) {
    output_shape.SetDim(i, input_shape.Dims(perms[i]));
  }

  TransposeParams params;
  params.perm_count = perms.size();
  for (int i = 0; i < perms.size(); ++i) {
    params.perm[i] = perms[i];
  }

  reference_ops::Transpose<integer_type>(params, input_shape, input->data(),
                                         output_shape,
                                         input_transposed->data());
}

TEST(TransposeOpTest, Test1D_UInt8) {
  // Basic 1D identity.
  std::vector<uint8_t> expected_output, input;
  std::vector<int> input_shape = {3};
  ComputeExpectedTransposeResult(input_shape, {0}, &input, &expected_output);

  TransposeOpModel model({TensorType_UINT8, input_shape, -10, 10}, {1}, {0},
                         true, {TensorType_UINT8, {}, -10, 10});
  model.SetInput<uint8_t>(input);
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput<uint8_t>(), ElementsAreArray(expected_output));
}

TEST(TransposeOpTest, Test1D_Int8) {
  // Basic 1D identity.
  std::vector<int8_t> expected_output, input;
  std::vector<int> input_shape = {3};
  ComputeExpectedTransposeResult(input_shape, {0}, &input, &expected_output);

  TransposeOpModel model({TensorType_INT8, input_shape, -10, 10}, {1}, {0},
                         true, {TensorType_INT8, {}, -10, 10});
  model.SetInput<int8_t>(input);
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput<int8_t>(), ElementsAreArray(expected_output));
}

TEST(TransposeOpTest, Test2D_UInt8) {
  std::vector<uint8_t> expected_output, input;
  std::vector<int> input_shape = {3, 2};
  std::vector<int> perm = {1, 0};
  ComputeExpectedTransposeResult(input_shape, perm, &input, &expected_output);

  TransposeOpModel model({TensorType_UINT8, input_shape, -10, 10}, {2}, {1, 0},
                         true, {TensorType_UINT8, {}, -10, 10});
  model.SetInput<uint8_t>(input);
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput<uint8_t>(), ElementsAreArray(expected_output));
}

TEST(TransposeOpTest, Test2D_Int8) {
  std::vector<int8_t> expected_output, input;
  std::vector<int> input_shape = {3, 2};
  std::vector<int> perm = {1, 0};
  ComputeExpectedTransposeResult(input_shape, perm, &input, &expected_output);

  TransposeOpModel model({TensorType_INT8, input_shape, -10, 10}, {2}, {1, 0},
                         true, {TensorType_INT8, {}, -10, 10});
  model.SetInput<int8_t>(input);
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput<int8_t>(), ElementsAreArray(expected_output));
}

TEST(TransposeOpTest, Test4D_UInt8) {
  std::vector<uint8_t> expected_output, input;
  std::vector<int> input_shape = {2, 2, 3, 1};
  std::vector<int> perm = {3, 0, 1, 2};
  ComputeExpectedTransposeResult(input_shape, perm, &input, &expected_output);

  TransposeOpModel model({TensorType_UINT8, input_shape, -10, 10}, {4},
                         {3, 0, 1, 2}, true, {TensorType_UINT8, {}, -10, 10});
  model.SetInput<uint8_t>(input);
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput<uint8_t>(), ElementsAreArray(expected_output));
}

TEST(TransposeOpTest, Test4D_Int8) {
  std::vector<int8_t> expected_output, input;
  std::vector<int> input_shape = {2, 2, 3, 1};
  std::vector<int> perm = {3, 0, 1, 2};
  ComputeExpectedTransposeResult(input_shape, perm, &input, &expected_output);

  TransposeOpModel model({TensorType_INT8, input_shape, -10, 10}, {4},
                         {3, 0, 1, 2}, true, {TensorType_INT8, {}, -10, 10});
  model.SetInput<int8_t>(input);
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput<int8_t>(), ElementsAreArray(expected_output));
}
}  // namespace tflite
