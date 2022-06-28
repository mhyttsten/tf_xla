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
class MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_lib_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_lib_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_lib_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/serialization/writer_lib.h"

#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {

using subgraph_test_util::CheckIntTensor;
using subgraph_test_util::FillIntTensor;

std::string CreateFilePath(const std::string& file_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_lib_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/lite/tools/serialization/writer_lib_test.cc", "CreateFilePath");

  return std::string(getenv("TEST_TMPDIR")) + file_name;
}

// The bool param indicates whether we use SubgraphWriter(true) or
// ModelWriter(false) for the test
class SingleSubgraphTest : public ::testing::TestWithParam<bool> {
 protected:
  void WriteToFile(Interpreter* interpreter, const std::string& filename,
                   bool use_subgraph_writer) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_lib_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/lite/tools/serialization/writer_lib_test.cc", "WriteToFile");

    if (use_subgraph_writer) {
      SubgraphWriter writer(&interpreter->primary_subgraph());
      CHECK_EQ(writer.Write(filename), kTfLiteOk);
    } else {
      ModelWriter writer(interpreter);
      CHECK_EQ(writer.Write(filename), kTfLiteOk);
    }
  }
};

TEST_P(SingleSubgraphTest, InvalidDestinations) {
  Interpreter interpreter;
  interpreter.AddTensors(3);
  float foo[] = {1, 2, 3};
  interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "a", {3},
                                           TfLiteQuantization());
  interpreter.SetTensorParametersReadOnly(
      1, kTfLiteFloat32, "b", {3}, TfLiteQuantization(),
      reinterpret_cast<char*>(foo), sizeof(foo));
  interpreter.SetTensorParametersReadWrite(2, kTfLiteFloat32, "c", {3},
                                           TfLiteQuantization());
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({2});
  const char* initial_data = "";
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  TfLiteAddParams* builtin_data =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data->activation = kTfLiteActNone;
  builtin_data->pot_scale_int16 = false;
  const TfLiteRegistration* reg = resolver.FindOp(BuiltinOperator_ADD, 1);
  interpreter.AddNodeWithParameters({0, 1}, {2}, initial_data, 0,
                                    reinterpret_cast<void*>(builtin_data), reg);

  // Check if invalid filename is handled gracefully.
  if (GetParam()) {
    SubgraphWriter writer(&interpreter.primary_subgraph());
    CHECK_EQ(writer.Write(""), kTfLiteError);
  } else {
    ModelWriter writer(&interpreter);
    CHECK_EQ(writer.Write(""), kTfLiteError);
  }

  // Check if invalid buffer is handled gracefully.
  size_t size;
  if (GetParam()) {
    SubgraphWriter writer(&interpreter.primary_subgraph());
    CHECK_EQ(writer.GetBuffer(nullptr, &size), kTfLiteError);
  } else {
    ModelWriter writer(&interpreter);
    CHECK_EQ(writer.GetBuffer(nullptr, &size), kTfLiteError);
  }
}

TEST_P(SingleSubgraphTest, FloatModelTest) {
  Interpreter interpreter;
  interpreter.AddTensors(3);
  float foo[] = {1, 2, 3};
  interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "a", {3},
                                           TfLiteQuantization());
  interpreter.SetTensorParametersReadOnly(
      1, kTfLiteFloat32, "b", {3}, TfLiteQuantization(),
      reinterpret_cast<char*>(foo), sizeof(foo));
  interpreter.SetTensorParametersReadWrite(2, kTfLiteFloat32, "c", {3},
                                           TfLiteQuantization());
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({2});
  const char* initial_data = "";
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  TfLiteAddParams* builtin_data =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data->activation = kTfLiteActNone;
  builtin_data->pot_scale_int16 = false;
  const TfLiteRegistration* reg = resolver.FindOp(BuiltinOperator_ADD, 1);
  interpreter.AddNodeWithParameters({0, 1}, {2}, initial_data, 0,
                                    reinterpret_cast<void*>(builtin_data), reg);

  const std::string test_file = CreateFilePath("test_float.tflite");
  WriteToFile(&interpreter, test_file, GetParam());
  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(test_file.c_str());
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> new_interpreter;
  builder(&new_interpreter);
  CHECK_EQ(new_interpreter->AllocateTensors(), kTfLiteOk);
}

// Tests writing only a portion of the subgraph.
TEST_P(SingleSubgraphTest, CustomInputOutputTest) {
  Interpreter interpreter;
  interpreter.AddTensors(4);
  constexpr float kFoo[] = {1, 2, 3};
  interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "a", {3},
                                           TfLiteQuantization());
  interpreter.SetTensorParametersReadOnly(
      1, kTfLiteFloat32, "b", {3}, TfLiteQuantization(),
      reinterpret_cast<const char*>(kFoo), sizeof(kFoo));
  interpreter.SetTensorParametersReadWrite(2, kTfLiteFloat32, "c", {3},
                                           TfLiteQuantization());
  interpreter.SetTensorParametersReadWrite(3, kTfLiteFloat32, "d", {3},
                                           TfLiteQuantization());
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({3});

  // Add two ops: Add and Relu
  const char* initial_data = "";
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  TfLiteAddParams* builtin_data =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data->activation = kTfLiteActNone;
  builtin_data->pot_scale_int16 = false;
  const TfLiteRegistration* reg = resolver.FindOp(BuiltinOperator_ADD, 1);
  interpreter.AddNodeWithParameters({0, 1}, {2}, initial_data, 0,
                                    reinterpret_cast<void*>(builtin_data), reg);

  const TfLiteRegistration* reg2 = resolver.FindOp(BuiltinOperator_RELU, 1);
  interpreter.AddNodeWithParameters({2}, {3}, nullptr, 0, nullptr, reg2);

  // Only write the second op.
  const std::string test_file = CreateFilePath("test_custom.tflite");
  SubgraphWriter writer(&interpreter.primary_subgraph());
  EXPECT_EQ(writer.SetCustomInputOutput(/*inputs=*/{2}, /*outputs=*/{3},
                                        /*execution_plan=*/{1}),
            kTfLiteOk);
  writer.SetUnusedTensors({0, 1});
  writer.Write(test_file);

  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(test_file.c_str());
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> new_interpreter;
  builder(&new_interpreter);
  ASSERT_EQ(new_interpreter->AllocateTensors(), kTfLiteOk);
}

TEST_P(SingleSubgraphTest, CustomInputOutputErrorCasesTest) {
  Interpreter interpreter;
  interpreter.AddTensors(5);
  constexpr float kFoo[] = {1, 2, 3};
  interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "a", {3},
                                           TfLiteQuantization());
  interpreter.SetTensorParametersReadOnly(
      1, kTfLiteFloat32, "b", {3}, TfLiteQuantization(),
      reinterpret_cast<const char*>(kFoo), sizeof(kFoo));
  interpreter.SetTensorParametersReadWrite(2, kTfLiteFloat32, "c", {3},
                                           TfLiteQuantization());
  interpreter.SetTensorParametersReadWrite(3, kTfLiteFloat32, "d", {3},
                                           TfLiteQuantization());
  interpreter.SetTensorParametersReadWrite(4, kTfLiteFloat32, "e", {3},
                                           TfLiteQuantization());
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({4});

  // Add three ops.
  const char* initial_data = "";
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  TfLiteAddParams* builtin_data =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data->activation = kTfLiteActNone;
  builtin_data->pot_scale_int16 = false;
  const TfLiteRegistration* reg = resolver.FindOp(BuiltinOperator_ADD, 1);
  interpreter.AddNodeWithParameters({0, 1}, {2}, initial_data, 0,
                                    reinterpret_cast<void*>(builtin_data), reg);

  const TfLiteRegistration* reg2 = resolver.FindOp(BuiltinOperator_RELU, 1);
  interpreter.AddNodeWithParameters({2}, {3}, nullptr, 0, nullptr, reg2);

  const TfLiteRegistration* reg3 = resolver.FindOp(BuiltinOperator_RELU6, 1);
  interpreter.AddNodeWithParameters({3}, {4}, nullptr, 0, nullptr, reg3);

  SubgraphWriter writer(&interpreter.primary_subgraph());

  // Test wrong input.
  EXPECT_EQ(writer.SetCustomInputOutput(/*inputs=*/{2}, /*outputs=*/{3},
                                        /*execution_plan=*/{0, 1}),
            kTfLiteError);
  // Test wrong output.
  EXPECT_EQ(writer.SetCustomInputOutput(/*inputs=*/{0, 1}, /*outputs=*/{4},
                                        /*execution_plan=*/{0, 1}),
            kTfLiteError);
  // Test a valid case.
  EXPECT_EQ(writer.SetCustomInputOutput(/*inputs=*/{0, 1}, /*outputs=*/{3},
                                        /*execution_plan=*/{0, 1}),
            kTfLiteOk);
}

// Tests if SetCustomInputOutput handles variable tensors correctly.
TEST_P(SingleSubgraphTest, CustomInputOutputVariableTensorTest) {
  Interpreter interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  // Create tensors.
  interpreter.AddTensors(3);
  interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "a", {3},
                                           TfLiteQuantization());
  interpreter.SetTensorParametersReadWrite(1, kTfLiteFloat32, "b", {3},
                                           TfLiteQuantization(),
                                           /*is_variable=*/true);
  interpreter.SetTensorParametersReadWrite(2, kTfLiteFloat32, "c", {3},
                                           TfLiteQuantization());
  interpreter.SetInputs({0});
  interpreter.SetOutputs({2});
  interpreter.SetVariables({1});

  // Create an Add node.
  TfLiteAddParams* builtin_data =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data->activation = kTfLiteActNone;
  builtin_data->pot_scale_int16 = false;
  interpreter.AddNodeWithParameters({0, 1}, {2}, nullptr, 0,
                                    reinterpret_cast<void*>(builtin_data),
                                    resolver.FindOp(BuiltinOperator_ADD, 1));

  // Write model to file.
  const std::string test_file = CreateFilePath("test_variables.tflite");
  SubgraphWriter writer(&interpreter.primary_subgraph());
  EXPECT_EQ(writer.SetCustomInputOutput(/*inputs=*/{0}, /*outputs=*/{2},
                                        /*execution_plan=*/{0}),
            kTfLiteOk);
  writer.Write(test_file);

  // Read model and test.
  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(test_file.c_str());
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> new_interpreter;
  builder(&new_interpreter);
  CHECK_EQ(new_interpreter->AllocateTensors(), kTfLiteOk);
}

TEST_P(SingleSubgraphTest, PerTensorQuantizedModelTest) {
  Interpreter interpreter;
  interpreter.AddTensors(3);
  interpreter.SetTensorParametersReadWrite(
      0, kTfLiteUInt8, "a", {3}, TfLiteQuantizationParams({1 / 256., 128}));
  interpreter.SetTensorParametersReadWrite(
      1, kTfLiteUInt8, "b", {3}, TfLiteQuantizationParams({1 / 256., 128}));
  interpreter.SetTensorParametersReadWrite(
      2, kTfLiteUInt8, "c", {3}, TfLiteQuantizationParams({1 / 256., 128}));
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({2});
  const char* initial_data = "";
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  TfLiteAddParams* builtin_data =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data->activation = kTfLiteActNone;
  builtin_data->pot_scale_int16 = false;
  const TfLiteRegistration* reg = resolver.FindOp(BuiltinOperator_ADD, 1);
  interpreter.AddNodeWithParameters({0, 1}, {2}, initial_data, 0,
                                    reinterpret_cast<void*>(builtin_data), reg);

  const std::string test_file = CreateFilePath("test_uint8.tflite");
  WriteToFile(&interpreter, test_file, GetParam());
  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(test_file.c_str());
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> new_interpreter;
  builder(&new_interpreter);
  CHECK_EQ(new_interpreter->AllocateTensors(), kTfLiteOk);
}

INSTANTIATE_TEST_SUITE_P(Writer, SingleSubgraphTest, ::testing::Bool());

struct ReshapeTestPattern {
  int num_inputs;
  bool is_param_valid;
  bool has_buggy_non_flatten_shape;
};

class ReshapeLayerTest : public ::testing::TestWithParam<ReshapeTestPattern> {};

TEST_P(ReshapeLayerTest, ReshapeLayerTest) {
  const auto param = GetParam();
  Interpreter interpreter;
  const int total_tensors = param.num_inputs + 1;
  interpreter.AddTensors(total_tensors);
  int output_shape[] = {1, 2, 3};
  interpreter.SetTensorParametersReadWrite(/*tensor_index=*/0, kTfLiteFloat32,
                                           /*name=*/"a", /*dims=*/{6},
                                           TfLiteQuantization());
  ASSERT_LE(param.num_inputs, 2);
  if (param.num_inputs == 2) {
    // Some TOCO generated models have buggy shape arguments, which are required
    // to be flatten, for example, dims={3, 1} instead of dims={3}.
    if (param.has_buggy_non_flatten_shape) {
      interpreter.SetTensorParametersReadOnly(
          /*tensor_index=*/1, kTfLiteInt32, /*name=*/"b", /*dims=*/{3, 1},
          TfLiteQuantization(), reinterpret_cast<char*>(output_shape),
          sizeof(output_shape));
    } else {
      interpreter.SetTensorParametersReadOnly(
          /*tensor_index=*/1, kTfLiteInt32, /*name=*/"b", /*dims=*/{3},
          TfLiteQuantization(), reinterpret_cast<char*>(output_shape),
          sizeof(output_shape));
    }
  }
  interpreter.SetTensorParametersReadWrite(/*tensor_index=*/total_tensors - 1,
                                           kTfLiteFloat32, /*name=*/"c",
                                           /*dims=*/{3}, TfLiteQuantization());

  std::vector<int> input_tensors(param.num_inputs);
  std::iota(input_tensors.begin(), input_tensors.end(), 0);

  interpreter.SetInputs(input_tensors);
  interpreter.SetOutputs({total_tensors - 1});
  const char* initial_data = "";
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  TfLiteReshapeParams* builtin_data = reinterpret_cast<TfLiteReshapeParams*>(
      malloc(sizeof(TfLiteReshapeParams)));
  memset(builtin_data, 0, sizeof(TfLiteReshapeParams));
  if (param.is_param_valid) {
    builtin_data->num_dimensions = 3;
    for (int dim = 0; dim < builtin_data->num_dimensions; ++dim) {
      builtin_data->shape[dim] = output_shape[dim];
    }
  }
  const TfLiteRegistration* reg = resolver.FindOp(BuiltinOperator_RESHAPE, 1);
  interpreter.AddNodeWithParameters(input_tensors,
                                    /*outputs=*/{total_tensors - 1},
                                    initial_data, /*init_data_size=*/0,
                                    reinterpret_cast<void*>(builtin_data), reg);

  SubgraphWriter writer(&interpreter.primary_subgraph());
  std::stringstream ss;
  ss << CreateFilePath("test_reshape_") << param.num_inputs
     << param.is_param_valid << ".tflite";
  std::string filename = ss.str();
  writer.Write(filename);
  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(filename.c_str());
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> new_interpreter;
  builder(&new_interpreter);
  ASSERT_EQ(new_interpreter->AllocateTensors(), kTfLiteOk);
}

INSTANTIATE_TEST_SUITE_P(
    Writer, ReshapeLayerTest,
    ::testing::Values(ReshapeTestPattern{/*num_inputs=*/2,
                                         /*is_param_valid=*/true,
                                         /*has_buggy_non_flatten_shape=*/false},
                      ReshapeTestPattern{/*num_inputs=*/2,
                                         /*is_param_valid=*/false,
                                         /*has_buggy_non_flatten_shape=*/false},
                      ReshapeTestPattern{/*num_inputs=*/1,
                                         /*is_param_valid=*/true,
                                         /*has_buggy_non_flatten_shape=*/false},
                      ReshapeTestPattern{/*num_inputs=*/2,
                                         /*is_param_valid=*/true,
                                         /*has_buggy_non_flatten_shape=*/true}),
    [](const ::testing::TestParamInfo<ReshapeLayerTest::ParamType>& info) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_lib_testDTcc mht_2(mht_2_v, 577, "", "./tensorflow/lite/tools/serialization/writer_lib_test.cc", "lambda");

      std::stringstream ss;
      ss << "num_inputs_" << info.param.num_inputs << "_valid_param_"
         << info.param.is_param_valid << "_buggy_shape_"
         << info.param.has_buggy_non_flatten_shape;
      std::string name = ss.str();
      return name;
    });

class WhileTest : public subgraph_test_util::ControlFlowOpTest {
 protected:
  TfLiteCustomAllocation NewCustomAlloc(size_t num_bytes,
                                        int required_alignment) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_lib_testDTcc mht_3(mht_3_v, 592, "", "./tensorflow/lite/tools/serialization/writer_lib_test.cc", "NewCustomAlloc");

    // Extra memory to ensure alignment.
    char* new_alloc = new char[num_bytes + required_alignment];
    char* new_underlying_buffer_aligned_ptr = reinterpret_cast<char*>(
        AlignTo(required_alignment, reinterpret_cast<intptr_t>(new_alloc)));
    custom_alloc_buffers_.emplace_back(new_alloc);

    return TfLiteCustomAllocation(
        {new_underlying_buffer_aligned_ptr, num_bytes});
  }

  intptr_t AlignTo(size_t alignment, intptr_t offset) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_lib_testDTcc mht_4(mht_4_v, 606, "", "./tensorflow/lite/tools/serialization/writer_lib_test.cc", "AlignTo");

    return offset % alignment == 0 ? offset
                                   : offset + (alignment - offset % alignment);
  }

  std::vector<std::unique_ptr<char[]>> custom_alloc_buffers_;
};

// The test builds a model that produces the i-th number of
// triangular number sequence: 1, 3, 6, 10, 15, 21, 28.
TEST_F(WhileTest, TestTriangularNumberSequence) {
  const int kSeqNumber = 4;
  const int kExpectedValue = 15;

  interpreter_.reset(new Interpreter);
  AddSubgraphs(2);
  builder_->BuildLessEqualCondSubgraph(interpreter_->subgraph(1), kSeqNumber);
  builder_->BuildAccumulateLoopBodySubgraph(interpreter_->subgraph(2));
  builder_->BuildWhileSubgraph(&interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});

  // Use custom allocation for second input, to ensure things work well for
  // non-traditional allocation types.
  auto alloc =
      NewCustomAlloc(interpreter_->tensor(interpreter_->inputs()[1])->bytes,
                     kDefaultTensorAlignment);
  auto* input_data = reinterpret_cast<int*>(alloc.data);
  input_data[0] = 1;
  interpreter_->SetCustomAllocationForTensor(interpreter_->inputs()[1], alloc);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {kSeqNumber + 1});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output2, {1}, {kExpectedValue});

  // Now serialize & deserialize model into a new Interpreter.
  ModelWriter writer(interpreter_.get());
  const std::string test_file = CreateFilePath("test_while.tflite");
  writer.Write(test_file);
  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(test_file.c_str());
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> new_interpreter;
  builder(&new_interpreter);

  // Check deserialized model.
  new_interpreter->ResizeInputTensor(interpreter_->inputs()[0], {1});
  new_interpreter->ResizeInputTensor(interpreter_->inputs()[1], {1});
  ASSERT_EQ(new_interpreter->AllocateTensors(), kTfLiteOk);
  FillIntTensor(new_interpreter->tensor(interpreter_->inputs()[0]), {1});
  FillIntTensor(new_interpreter->tensor(interpreter_->inputs()[1]), {1});
  ASSERT_EQ(new_interpreter->Invoke(), kTfLiteOk);
  output1 = new_interpreter->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {kSeqNumber + 1});
  output2 = new_interpreter->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output2, {1}, {kExpectedValue});
}

// Verifies the ModelWriters constructing from  an interpreter or subgraphs
// produce the same results.
TEST_F(WhileTest, TestModelWriterFromSubgraphs) {
  const int kSeqNumber = 4;
  const int kExpectedValue = 15;

  interpreter_.reset(new Interpreter);
  AddSubgraphs(2);
  builder_->BuildLessEqualCondSubgraph(interpreter_->subgraph(1), kSeqNumber);
  builder_->BuildAccumulateLoopBodySubgraph(interpreter_->subgraph(2));
  builder_->BuildWhileSubgraph(&interpreter_->primary_subgraph());

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[0]), {1});

  // Use custom allocation for second input, to ensure things work well for
  // non-traditional allocation types.
  auto alloc =
      NewCustomAlloc(interpreter_->tensor(interpreter_->inputs()[1])->bytes,
                     kDefaultTensorAlignment);
  auto* input_data = reinterpret_cast<int*>(alloc.data);
  input_data[0] = 1;
  interpreter_->SetCustomAllocationForTensor(interpreter_->inputs()[1], alloc);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output1, {1}, {kSeqNumber + 1});
  TfLiteTensor* output2 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output2, {1}, {kExpectedValue});

  // Serializes the model using the interpreter.
  ModelWriter writer_1(interpreter_.get());
  const std::string test_file_1 = CreateFilePath("test_while_1.tflite");
  writer_1.Write(test_file_1);

  // Serializes the model using subgraphs.
  std::vector<Subgraph*> subgraphs;
  for (int i = 0; i < interpreter_->subgraphs_size(); ++i) {
    subgraphs.push_back(interpreter_->subgraph(i));
  }
  ModelWriter writer_2(subgraphs);
  const std::string test_file_2 = CreateFilePath("test_while_2.tflite");
  writer_2.Write(test_file_2);

  // The results from both ModelWriters should be the same.
  std::ifstream file_ifs_1(test_file_1, std::ios::in);
  std::ostringstream model_content_1;
  model_content_1 << file_ifs_1.rdbuf();

  std::ifstream file_ifs_2(test_file_2, std::ios::in);
  std::ostringstream model_content_2;
  model_content_2 << file_ifs_2.rdbuf();

  EXPECT_FALSE(model_content_1.str().empty());
  EXPECT_EQ(model_content_1.str(), model_content_2.str());
}

}  // namespace tflite
