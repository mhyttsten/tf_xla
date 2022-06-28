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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTcc() {
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

#include "tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h"

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
void QuantizedResizeBilinearTester::Test(
    Interpreter* delegate_interpreter, Interpreter* default_interpreter) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.cc", "QuantizedResizeBilinearTester::Test");

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng = std::bind(
      std::uniform_int_distribution<int32_t>(std::numeric_limits<T>::min(),
                                             std::numeric_limits<T>::max()),
      std::ref(rng));

  T* default_input_data = default_interpreter->typed_input_tensor<T>(0);
  std::generate(default_input_data,
                default_input_data +
                    BatchSize() * InputHeight() * InputWidth() * Channels(),
                std::ref(input_rng));

  T* delegate_input_data = delegate_interpreter->typed_input_tensor<T>(0);
  std::copy(default_input_data,
            default_input_data +
                BatchSize() * InputHeight() * InputWidth() * Channels(),
            delegate_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  T* default_output_data = default_interpreter->typed_output_tensor<T>(0);
  T* delegate_output_data = delegate_interpreter->typed_output_tensor<T>(0);

  for (int i = 0; i < BatchSize(); i++) {
    for (int y = 0; y < OutputHeight(); y++) {
      for (int x = 0; x < OutputWidth(); x++) {
        for (int c = 0; c < Channels(); c++) {
          const int index =
              ((i * OutputHeight() + y) * OutputWidth() + x) * Channels() + c;
          ASSERT_LE(std::abs(static_cast<int32_t>(default_output_data[index]) -
                             static_cast<int32_t>(delegate_output_data[index])),
                    1)
              << "batch " << i << " / " << BatchSize() << ", y position " << y
              << " / " << OutputHeight() << ", x position " << x << " / "
              << OutputWidth() << ", channel " << c << " / " << Channels();
        }
      }
    }
  }
}

void QuantizedResizeBilinearTester::Test(TfLiteDelegate* delegate) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTcc mht_1(mht_1_v, 255, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.cc", "QuantizedResizeBilinearTester::Test");

  std::vector<char> buffer = CreateTfLiteModel();
  const Model* model = GetModel(buffer.data());

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

  ASSERT_EQ(delegate_interpreter->inputs().size(), 1);
  ASSERT_EQ(default_interpreter->inputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
  ASSERT_EQ(default_interpreter->outputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  if (Unsigned()) {
    Test<uint8_t>(delegate_interpreter.get(), default_interpreter.get());
  } else {
    Test<int8_t>(delegate_interpreter.get(), default_interpreter.get());
  }
}

std::vector<char> QuantizedResizeBilinearTester::CreateTfLiteModel() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_resize_bilinear_testerDTcc mht_2(mht_2_v, 298, "", "./tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.cc", "QuantizedResizeBilinearTester::CreateTfLiteModel");

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, BuiltinOperator_RESIZE_BILINEAR);

  flatbuffers::Offset<tflite::ResizeBilinearOptions> resize_bilinear_options =
      CreateResizeBilinearOptions(builder, AlignCorners(), HalfPixelCenters());

  const std::array<int32_t, 2> size_data{{OutputHeight(), OutputWidth()}};

  const std::array<flatbuffers::Offset<Buffer>, 2> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(size_data.data()),
                       size_data.size() * sizeof(int32_t))),
  }};

  const std::array<int32_t, 4> input_shape{
      {BatchSize(), InputHeight(), InputWidth(), Channels()}};
  const std::array<int32_t, 4> output_shape{
      {BatchSize(), OutputHeight(), OutputWidth(), Channels()}};
  const std::array<int32_t, 1> size_shape{
      {static_cast<int32_t>(size_data.size())}};

  const std::array<flatbuffers::Offset<Tensor>, 3> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name=*/0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(size_shape.data(), size_shape.size()),
          TensorType_INT32, /*buffer=*/1),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(output_shape.data(),
                                                 output_shape.size()),
                   Unsigned() ? TensorType_UINT8 : TensorType_INT8,
                   /*buffer=*/0, /*name=*/0,
                   CreateQuantizationParameters(
                       builder, /*min=*/0, /*max=*/0,
                       builder.CreateVector<float>({Scale()}),
                       builder.CreateVector<int64_t>({ZeroPoint()}))),
  }};

  const std::array<int32_t, 2> op_inputs{{0, 1}};
  const std::array<int32_t, 1> op_outputs{{2}};
  flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      BuiltinOptions_ResizeBilinearOptions, resize_bilinear_options.Union());

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 1> subgraph_outputs{{2}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Quantized Resize Bilinear model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

}  // namespace xnnpack
}  // namespace tflite
