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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSsplit_testerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSsplit_testerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSsplit_testerDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/xnnpack/split_tester.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

template <class T>
void SplitTester::Test(Interpreter *delegate_interpreter,
                       Interpreter *default_interpreter) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSsplit_testerDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/delegates/xnnpack/split_tester.cc", "SplitTester::Test");

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_int_distribution<int32_t> input_distribution(
      std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  auto input_rng = std::bind(input_distribution, std::ref(rng));

  T *default_input_data = default_interpreter->typed_input_tensor<T>(1);
  std::generate(default_input_data,
                default_input_data + ComputeSize(InputShape()),
                std::ref(input_rng));

  T *xnnpack_input_data = delegate_interpreter->typed_input_tensor<T>(1);
  std::copy(default_input_data, default_input_data + ComputeSize(InputShape()),
            xnnpack_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  T *default_output1_data = default_interpreter->typed_output_tensor<T>(0);
  T *xnnpack_output1_data = delegate_interpreter->typed_output_tensor<T>(0);
  T *default_output2_data = default_interpreter->typed_output_tensor<T>(1);
  T *xnnpack_output2_data = delegate_interpreter->typed_output_tensor<T>(1);

  for (size_t i = 0; i < ComputeSize(OutputShape()); i++) {
    ASSERT_EQ(static_cast<int32_t>(default_output1_data[i]),
              static_cast<int32_t>(xnnpack_output1_data[i]));
    ASSERT_EQ(static_cast<int32_t>(default_output2_data[i]),
              static_cast<int32_t>(xnnpack_output2_data[i]));
  }
}

template <>
void SplitTester::Test<float>(Interpreter *delegate_interpreter,
                              Interpreter *default_interpreter) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSsplit_testerDTcc mht_1(mht_1_v, 246, "", "./tensorflow/lite/delegates/xnnpack/split_tester.cc", "SplitTester::Test<float>");

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> input_distribution(-25.0f, 25.0f);
  auto input_rng = std::bind(input_distribution, std::ref(rng));

  float *default_input_data = default_interpreter->typed_input_tensor<float>(1);
  std::generate(default_input_data,
                default_input_data + ComputeSize(InputShape()),
                std::ref(input_rng));

  float *xnnpack_input_data =
      delegate_interpreter->typed_input_tensor<float>(1);
  std::copy(default_input_data, default_input_data + ComputeSize(InputShape()),
            xnnpack_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float *default_output1_data =
      default_interpreter->typed_output_tensor<float>(0);
  float *xnnpack_output1_data =
      delegate_interpreter->typed_output_tensor<float>(0);
  float *default_output2_data =
      default_interpreter->typed_output_tensor<float>(0);
  float *xnnpack_output2_data =
      delegate_interpreter->typed_output_tensor<float>(0);

  for (size_t i = 0; i < ComputeSize(OutputShape()); i++) {
    ASSERT_EQ(default_output1_data[i], xnnpack_output1_data[i]);
    ASSERT_EQ(default_output2_data[i], xnnpack_output2_data[i]);
  }
}

void SplitTester::Test(TensorType tensor_type, TfLiteDelegate *delegate) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSsplit_testerDTcc mht_2(mht_2_v, 283, "", "./tensorflow/lite/delegates/xnnpack/split_tester.cc", "SplitTester::Test");

  std::vector<char> buffer = CreateTfLiteModel(tensor_type);
  const Model *model = GetModel(buffer.data());

  int32_t axis = SplitDimension();
  axis += axis < 0 ? InputShape().size() : 0;
  ASSERT_EQ(0, InputShape()[axis] % NumSplits());

  std::unique_ptr<Interpreter> delegate_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &delegate_interpreter),
      kTfLiteOk);
  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &default_interpreter),
      kTfLiteOk);

  ASSERT_TRUE(delegate_interpreter);
  ASSERT_TRUE(default_interpreter);
  ASSERT_EQ(delegate_interpreter->inputs().size(), 2);
  ASSERT_EQ(default_interpreter->inputs().size(), 2);
  ASSERT_EQ(delegate_interpreter->outputs().size(), NumSplits());
  ASSERT_EQ(default_interpreter->outputs().size(), NumSplits());

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  switch (tensor_type) {
    case TensorType_FLOAT32:
      Test<float>(delegate_interpreter.get(), default_interpreter.get());
      break;
    case TensorType_INT8:
      Test<int8_t>(delegate_interpreter.get(), default_interpreter.get());
      break;
    case TensorType_UINT8:
      Test<uint8_t>(delegate_interpreter.get(), default_interpreter.get());
      break;
    default:
      GTEST_FAIL();
  }
}

std::vector<char> SplitTester::CreateTfLiteModel(TensorType tensor_type) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSsplit_testerDTcc mht_3(mht_3_v, 336, "", "./tensorflow/lite/delegates/xnnpack/split_tester.cc", "SplitTester::CreateTfLiteModel");

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, BuiltinOperator_SPLIT, 0);

  std::array<int32_t, 1> split_dim = {SplitDimension()};
  std::vector<flatbuffers::Offset<Buffer>> buffers{
      {CreateBuffer(builder, builder.CreateVector({})),
       CreateBuffer(builder,
                    builder.CreateVector(
                        reinterpret_cast<const uint8_t *>(split_dim.data()),
                        split_dim.size() * sizeof(int32_t)))}};
  std::array<int32_t, 0> split_dim_shape = {};

  flatbuffers::Offset<QuantizationParameters> quantization_params =
      CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0,
          builder.CreateVector<float>({/*scale=*/1.0f}),
          builder.CreateVector<int64_t>({/*zero_point=*/0}));

  std::vector<flatbuffers::Offset<Tensor>> tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(split_dim_shape.data(),
                                                 split_dim_shape.size()),
                   TensorType_INT32, /*buffer=*/1, /*name=*/0,
                   quantization_params),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(InputShape().data(),
                                                 InputShape().size()),
                   tensor_type,
                   /*buffer=*/0, /*name=*/0, quantization_params),
  }};

  for (int i = 0; i < NumSplits(); i++) {
    tensors.push_back(
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(OutputShape().data(),
                                                   OutputShape().size()),
                     tensor_type,
                     /*buffer=*/0, /*name=*/0, quantization_params));
  }

  const std::array<int32_t, 2> op_inputs{0, 1};
  std::vector<int32_t> op_outputs;
  op_outputs.reserve(NumSplits());
  for (int i = 0; i < NumSplits(); i++) {
    op_outputs.push_back(op_inputs.size() + i);
  }
  EXPECT_EQ(op_outputs.size(), NumSplits());

  const flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      tflite::BuiltinOptions_SplitOptions,
      CreateSplitOptions(builder, NumSplits()).Union());

  const std::array<int32_t, 2> subgraph_inputs = op_inputs;
  const std::vector<int32_t> subgraph_outputs = op_outputs;
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(&op, 1));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("Split model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t SplitTester::ComputeSize(const std::vector<int32_t> &shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSsplit_testerDTcc mht_4(mht_4_v, 417, "", "./tensorflow/lite/delegates/xnnpack/split_tester.cc", "SplitTester::ComputeSize");

  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
