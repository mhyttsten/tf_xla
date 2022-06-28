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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc() {
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

#include "tensorflow/lite/kernels/subgraph_test_util.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {

// Forward declaration for op kernels.
namespace ops {
namespace custom {
namespace random_int {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 0);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* output = GetOutput(context, node, 0);
  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(1);
  outputSize->data[0] = 1;
  return context->ResizeTensor(context, output, outputSize);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_1(mht_1_v, 223, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "Eval");

  TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  std::random_device rd;
  std::uniform_int_distribution<int> dist(1, 32768);
  output.data.i32[0] = dist(rd);
  return kTfLiteOk;
}

}  // namespace random_int

TfLiteRegistration* Register_RANDOM_INT() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_2(mht_2_v, 237, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "Register_RANDOM_INT");

  static TfLiteRegistration r = {nullptr, nullptr, random_int::Prepare,
                                 random_int::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops

namespace subgraph_test_util {

namespace {

void SetupTensor(Subgraph* subgraph, int tensor_index, TfLiteType type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_3(mht_3_v, 253, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SetupTensor");

  ASSERT_EQ(subgraph->SetTensorParametersReadWrite(tensor_index, type, "", 0,
                                                   nullptr, {}, false),
            kTfLiteOk);
}

}  // namespace

SubgraphBuilder::~SubgraphBuilder() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_4(mht_4_v, 264, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::~SubgraphBuilder");

  for (auto buffer : buffers_) {
    free(buffer);
  }
}

void SubgraphBuilder::BuildAddSubgraph(Subgraph* subgraph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_5(mht_5_v, 273, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildAddSubgraph");

  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput = 2;
  const int kTensorCount = 3;
  // kInput1(0) --> +---+
  //                |ADD| --> kOutput(2)
  // kInput2(1) --> +---+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);

  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  auto* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kInput2}, {kOutput}, {}, nullptr, 0,
                                  params, add_reg, &node_index);
}

// Build a subgraph with an mul op. Helper function for testing.
void SubgraphBuilder::BuildMulSubgraph(Subgraph* subgraph) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_6(mht_6_v, 307, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildMulSubgraph");

  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput = 2;
  const int kTensorCount = 3;
  // kInput1(0) --> +---+
  //                |MUL| --> kOutput(2)
  // kInput2(1) --> +---+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);

  TfLiteMulParams* params =
      reinterpret_cast<TfLiteMulParams*>(malloc(sizeof(TfLiteMulParams)));
  params->activation = kTfLiteActNone;
  auto* mul_reg = ops::builtin::Register_MUL();
  mul_reg->builtin_code = kTfLiteBuiltinMul;
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kInput2}, {kOutput}, {}, nullptr, 0,
                                  params, mul_reg, &node_index);
}

// Build a subgraph with a pad op. Helper function for testing.
void SubgraphBuilder::BuildPadSubgraph(Subgraph* subgraph) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_7(mht_7_v, 341, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildPadSubgraph");

  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput = 2;
  const int kTensorCount = 3;
  // kInput1(0) --> +---+
  //                |PAD| --> kOutput(2)
  // kInput2(1) --> +---+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);

  TfLitePadParams* params =
      reinterpret_cast<TfLitePadParams*>(malloc(sizeof(TfLitePadParams)));
  auto* pad_reg = ops::builtin::Register_PAD();
  pad_reg->builtin_code = kTfLiteBuiltinPad;
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kInput2}, {kOutput}, {}, nullptr, 0,
                                  params, pad_reg, &node_index);
}

void SubgraphBuilder::BuildIfSubgraph(Subgraph* subgraph) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_8(mht_8_v, 373, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildIfSubgraph");

  const int kCondInput = 0;
  const int kInput1 = 1;
  const int kInput2 = 2;
  const int kOutput = 3;
  const int kTensorCount = 4;

  // kCondInput(0) --> +----+
  // kInput1(1)  ----> | IF | --> kOutput(3)
  // kInput2(2)  ----> +----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kCondInput, kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kCondInput, kTfLiteBool);
  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);

  TfLiteIfParams* params =
      reinterpret_cast<TfLiteIfParams*>(malloc(sizeof(TfLiteIfParams)));
  params->then_subgraph_index = 1;
  params->else_subgraph_index = 2;
  auto* if_reg = ops::builtin::Register_IF();
  if_reg->builtin_code = kTfLiteBuiltinIf;

  int node_index;
  subgraph->AddNodeWithParameters({kCondInput, kInput1, kInput2}, {kOutput}, {},
                                  nullptr, 0, params, if_reg, &node_index);
}

void SubgraphBuilder::BuildLessEqualCondSubgraph(Subgraph* subgraph, int rhs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_9(mht_9_v, 411, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildLessEqualCondSubgraph");

  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput = 2;
  const int kConstRhs = 3;
  const int kTensorCount = 4;

  // kInput1(0) ----> +------------+
  //                  | LESS_EQUAL | --> kOutput(2)
  // kConstRhs(3) --> +------------+
  //
  // kInput2(1) --> (unused)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteBool);

  auto* le_reg = ops::builtin::Register_LESS_EQUAL();
  le_reg->builtin_code = kTfLiteBuiltinLessEqual;

  CreateConstantInt32Tensor(subgraph, kConstRhs, {1}, {rhs});
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kConstRhs}, {kOutput}, {}, nullptr,
                                  0, nullptr, le_reg, &node_index);
}

void SubgraphBuilder::BuildAccumulateLoopBodySubgraph(Subgraph* subgraph) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_10(mht_10_v, 447, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildAccumulateLoopBodySubgraph");

  const int kInputCounter = 0;
  const int kInputValue = 1;
  const int kOutputCounter = 2;
  const int kOutputValue = 3;
  const int kConstStep = 4;
  const int kTensorCount = 5;

  // kInputCounter(0) --> +-----+
  //                      | ADD | --> kOutputCounter(2)
  // kConstStep(4) -----> +-----+            |
  //                                         |
  //                                         v
  //                                      +-----+
  //                                      | ADD | --> kOutputValue(3)
  // kInputValue(1) ----------------------+-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue}), kTfLiteOk);

  SetupTensor(subgraph, kInputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kInputValue, kTfLiteInt32);
  SetupTensor(subgraph, kOutputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kOutputValue, kTfLiteInt32);
  CreateConstantInt32Tensor(subgraph, kConstStep, {1}, {1});

  int node_index;
  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  params->pot_scale_int16 = false;
  auto* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  subgraph->AddNodeWithParameters({0, 4}, {2}, {}, nullptr, 0, params, add_reg,
                                  &node_index);
  params = reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  params->pot_scale_int16 = false;
  subgraph->AddNodeWithParameters({2, 1}, {3}, {}, nullptr, 0, params, add_reg,
                                  &node_index);
}

void SubgraphBuilder::BuildPadLoopBodySubgraph(Subgraph* subgraph,
                                               const std::vector<int> padding) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_11(mht_11_v, 497, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildPadLoopBodySubgraph");

  const int kInputCounter = 0;
  const int kInputValue = 1;
  const int kOutputCounter = 2;
  const int kOutputValue = 3;
  const int kConstStep = 4;
  const int kConstPadding = 5;
  const int kTensorCount = 6;

  // kInputCounter(0) --> +-----+
  //                      | ADD | --> kOutputCounter(2)
  // kConstStep(4) -----> +-----+
  //
  // kInputValue(1) ----> +-----+
  //                      | PAD | --> kOutputValue(3)
  // kConstPadding(5) --> +-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue}), kTfLiteOk);

  SetupTensor(subgraph, kInputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kInputValue, kTfLiteInt32);
  SetupTensor(subgraph, kOutputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kOutputValue, kTfLiteInt32);

  CreateConstantInt32Tensor(subgraph, kConstStep, {1}, {1});
  ASSERT_EQ(padding.size() % 2, 0);
  int padding_dims = padding.size();
  CreateConstantInt32Tensor(subgraph, kConstPadding, {1, padding_dims},
                            padding);

  int node_index;
  TfLiteAddParams* add_params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  add_params->activation = kTfLiteActNone;
  auto* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  subgraph->AddNodeWithParameters({kInputCounter, kConstStep}, {kOutputCounter},
                                  {}, nullptr, 0, add_params, add_reg,
                                  &node_index);
  TfLitePadParams* pad_params =
      reinterpret_cast<TfLitePadParams*>(malloc(sizeof(TfLiteAddParams)));
  auto* pad_reg = ops::builtin::Register_PAD();
  pad_reg->builtin_code = kTfLiteBuiltinPad;
  subgraph->AddNodeWithParameters({kInputValue, kConstPadding}, {kOutputValue},
                                  {}, nullptr, 0, pad_params, pad_reg,
                                  &node_index);
}

void SubgraphBuilder::BuildWhileSubgraph(Subgraph* subgraph) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_12(mht_12_v, 553, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildWhileSubgraph");

  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput1 = 2;
  const int kOutput2 = 3;
  const int kTensorCount = 4;

  // kInput1(0) --> +-------+ --> kOutput1(2)
  //                | WHILE |
  // kInput2(1) --> +-------+ --> kOutput2(3)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput1, kOutput2}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput1, kTfLiteInt32);
  SetupTensor(subgraph, kOutput2, kTfLiteInt32);

  TfLiteWhileParams* params =
      reinterpret_cast<TfLiteWhileParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->cond_subgraph_index = 1;
  params->body_subgraph_index = 2;
  auto* while_reg = ops::builtin::Register_WHILE();
  while_reg->builtin_code = kTfLiteBuiltinWhile;

  int node_index;
  subgraph->AddNodeWithParameters({0, 1}, {2, 3}, {}, nullptr, 0, params,
                                  while_reg, &node_index);
}

void SubgraphBuilder::BuildAssignRandomValueToVariableSubgraph(
    Subgraph* subgraph) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_13(mht_13_v, 592, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildAssignRandomValueToVariableSubgraph");

  const int kConstResourceId = 0;
  const int kRandomValue = 1;
  const int kTensorCount = 3;

  // Construct a graph like ths:
  //   %1 = random_int()
  //   variable_assign(%0, %1)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetInputs({}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({}), kTfLiteOk);

  SetupTensor(subgraph, kRandomValue, kTfLiteInt32);
  CreateConstantInt32Tensor(subgraph, kConstResourceId, {1}, {1024});

  int node_index;
  subgraph->AddNodeWithParameters({}, {kRandomValue}, {}, nullptr, 0, nullptr,
                                  ::tflite::ops::custom::Register_RANDOM_INT(),
                                  &node_index);
  subgraph->AddNodeWithParameters(
      {kConstResourceId, kRandomValue}, {}, {}, nullptr, 0, nullptr,
      ::tflite::ops::builtin::Register_ASSIGN_VARIABLE(), &node_index);
}

void SubgraphBuilder::BuildCallOnceAndReadVariableSubgraph(Subgraph* subgraph) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_14(mht_14_v, 622, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildCallOnceAndReadVariableSubgraph");

  const int kConstResourceId = 0;
  const int kOutput = 1;
  const int kTensorCount = 2;

  // Construct a graph like ths:
  //   Output: %1
  //   %1 = read_variable(%0)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetInputs({}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kOutput, kTfLiteInt32);
  CreateConstantInt32Tensor(subgraph, kConstResourceId, {1}, {1024});

  TfLiteCallOnceParams* params = reinterpret_cast<TfLiteCallOnceParams*>(
      malloc(sizeof(TfLiteCallOnceParams)));
  params->init_subgraph_index = 1;

  int node_index;
  subgraph->AddNodeWithParameters({}, {}, {}, nullptr, 0, params,
                                  ::tflite::ops::builtin::Register_CALL_ONCE(),
                                  &node_index);
  subgraph->AddNodeWithParameters(
      {kConstResourceId}, {kOutput}, {}, nullptr, 0, nullptr,
      ::tflite::ops::builtin::Register_READ_VARIABLE(), &node_index);
}

void SubgraphBuilder::BuildCallOnceAndReadVariablePlusOneSubgraph(
    Subgraph* subgraph) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_15(mht_15_v, 657, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildCallOnceAndReadVariablePlusOneSubgraph");

  const int kConstResourceId = 0;
  const int kConstOne = 1;
  const int kReadVariableResult = 2;
  const int kOutput = 3;
  const int kTensorCount = 4;

  // Construct a graph like ths:
  //   Output: %3
  //   %2 = read_variable(%0)
  //   %3 = add(%2, %1)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetInputs({}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kReadVariableResult, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);
  CreateConstantInt32Tensor(subgraph, kConstResourceId, {1}, {1024});
  CreateConstantInt32Tensor(subgraph, kConstOne, {1}, {1});

  TfLiteCallOnceParams* params = reinterpret_cast<TfLiteCallOnceParams*>(
      malloc(sizeof(TfLiteCallOnceParams)));
  params->init_subgraph_index = 1;

  int node_index;
  subgraph->AddNodeWithParameters({}, {}, {}, nullptr, 0, params,
                                  ::tflite::ops::builtin::Register_CALL_ONCE(),
                                  &node_index);
  subgraph->AddNodeWithParameters(
      {kConstResourceId}, {kReadVariableResult}, {}, nullptr, 0, nullptr,
      ::tflite::ops::builtin::Register_READ_VARIABLE(), &node_index);

  TfLiteAddParams* add_params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  add_params->activation = kTfLiteActNone;
  subgraph->AddNodeWithParameters(
      {kReadVariableResult, kConstOne}, {kOutput}, {}, nullptr, 0, add_params,
      ::tflite::ops::builtin::Register_ADD(), &node_index);
}

void SubgraphBuilder::BuildLessEqualCondSubgraphWithDynamicTensor(
    Subgraph* subgraph, int rhs) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_16(mht_16_v, 704, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildLessEqualCondSubgraphWithDynamicTensor");

  const int kStringInput1 = 0;
  const int kStringInput2 = 1;
  const int kIntegerInput = 2;
  const int kOutput = 3;
  const int kConstRhs = 4;
  const int kTensorCount = 5;

  // kIntegerInput(2) --> +------------+
  //                      | LESS_EQUAL | --> kOutput(3)
  //     kConstRhs(4) --> +------------+
  //
  // kStringInput1(0) --> (unused)
  // kStringInput2(1) --> (unused)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kStringInput1, kStringInput2, kIntegerInput}),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kStringInput1, kTfLiteString);
  SetupTensor(subgraph, kStringInput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerInput, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteBool);

  auto* le_reg = ops::builtin::Register_LESS_EQUAL();
  le_reg->builtin_code = kTfLiteBuiltinLessEqual;

  CreateConstantInt32Tensor(subgraph, kConstRhs, {1}, {rhs});
  int node_index;
  subgraph->AddNodeWithParameters({kIntegerInput, kConstRhs}, {kOutput}, {},
                                  nullptr, 0, nullptr, le_reg, &node_index);
}

void SubgraphBuilder::BuildBodySubgraphWithDynamicTensor(Subgraph* subgraph) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_17(mht_17_v, 744, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildBodySubgraphWithDynamicTensor");

  const int kStringInput1 = 0;
  const int kStringInput2 = 1;
  const int kIntegerInput = 2;
  const int kStringOutput1 = 0;  // Forwarded of the `kStringInput1` tensor.
  const int kStringOutput2 = 4;
  const int kIntegerOutput = 5;
  const int kConst = 6;
  const int kTensorCount = 7;

  // Construct a graph like this:
  //   %5 = tf.Add(%2, 1)
  //   %4 = tf.Fill(%0, %5)
  //   yield(%0, %4, %5)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kStringInput1, kStringInput2, kIntegerInput}),
            kTfLiteOk);
  ASSERT_EQ(
      subgraph->SetOutputs({kStringOutput1, kStringOutput2, kIntegerOutput}),
      kTfLiteOk);

  SetupTensor(subgraph, kStringInput1, kTfLiteString);
  SetupTensor(subgraph, kStringInput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerInput, kTfLiteInt32);
  SetupTensor(subgraph, kStringOutput1, kTfLiteString);
  SetupTensor(subgraph, kStringOutput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerOutput, kTfLiteInt32);
  SetupTensor(subgraph, kConst, kTfLiteInt32);

  TfLiteAddParams* add_params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  add_params->activation = kTfLiteActNone;

  auto* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;

  CreateConstantInt32Tensor(subgraph, kConst, {1}, {1});
  int node_index;
  subgraph->AddNodeWithParameters({kIntegerInput, kConst}, {kIntegerOutput}, {},
                                  nullptr, 0, add_params, add_reg, &node_index);

  auto* fill_reg = ops::builtin::Register_FILL();
  fill_reg->builtin_code = kTfLiteBuiltinFill;
  subgraph->AddNodeWithParameters({kIntegerOutput, kStringInput1},
                                  {kStringOutput2}, {}, nullptr, 0, nullptr,
                                  fill_reg, &node_index);
}

void SubgraphBuilder::BuildWhileSubgraphWithDynamicTensor(Subgraph* subgraph) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_18(mht_18_v, 799, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::BuildWhileSubgraphWithDynamicTensor");

  const int kStringInput1 = 0;
  const int kStringInput2 = 1;
  const int kIntegerInput = 2;
  const int kStringOutput1 = 3;
  const int kStringOutput2 = 4;
  const int kIntegerOutput = 5;
  const int kTensorCount = 6;

  // Create a while op with 2 string tensor and 1 integer tensor.
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kStringInput1, kStringInput2, kIntegerInput}),
            kTfLiteOk);
  ASSERT_EQ(
      subgraph->SetOutputs({kStringOutput1, kStringOutput2, kIntegerOutput}),
      kTfLiteOk);

  SetupTensor(subgraph, kStringInput1, kTfLiteString);
  SetupTensor(subgraph, kStringInput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerInput, kTfLiteInt32);
  SetupTensor(subgraph, kStringOutput1, kTfLiteString);
  SetupTensor(subgraph, kStringOutput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerOutput, kTfLiteInt32);

  TfLiteWhileParams* params =
      reinterpret_cast<TfLiteWhileParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->cond_subgraph_index = 1;
  params->body_subgraph_index = 2;
  auto* while_reg = ops::builtin::Register_WHILE();
  while_reg->builtin_code = kTfLiteBuiltinWhile;

  int node_index;
  subgraph->AddNodeWithParameters(
      {kStringInput1, kStringInput2, kIntegerInput},
      {kStringOutput1, kStringOutput2, kIntegerOutput}, {}, nullptr, 0, params,
      while_reg, &node_index);
}

void SubgraphBuilder::CreateConstantInt32Tensor(Subgraph* subgraph,
                                                int tensor_index,
                                                const std::vector<int>& shape,
                                                const std::vector<int>& data) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_19(mht_19_v, 846, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "SubgraphBuilder::CreateConstantInt32Tensor");

  ASSERT_GT(shape.size(), 0);
  int num_elements = 1;
  for (int dim : shape) {
    num_elements *= dim;
  }
  ASSERT_EQ(data.size(), num_elements);
  size_t size_in_bytes = sizeof(int32_t) * num_elements;
  // Maybe aligned.
  int32_t* buffer = reinterpret_cast<int32_t*>(malloc(size_in_bytes));
  for (int i = 0; i < num_elements; ++i) {
    buffer[i] = data[i];
  }
  buffers_.push_back(buffer);
  ASSERT_EQ(subgraph->SetTensorParametersReadOnly(
                tensor_index, kTfLiteInt32, "", shape, {},
                reinterpret_cast<const char*>(buffer), size_in_bytes),
            kTfLiteOk);
}

void FillIntTensor(TfLiteTensor* tensor, const std::vector<int32_t>& data) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_20(mht_20_v, 869, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "FillIntTensor");

  int count = NumElements(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    tensor->data.i32[i] = data[i];
  }
}

void FillScalarStringTensor(TfLiteTensor* tensor, const std::string& data) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("data: \"" + data + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_21(mht_21_v, 881, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "FillScalarStringTensor");

  StringRef str_ref;
  str_ref.str = data.c_str();
  str_ref.len = data.size();
  DynamicBuffer buf;
  buf.AddString(str_ref);
  buf.WriteToTensor(tensor, /*new_shape=*/TfLiteIntArrayCreate(0));
}

void CheckScalarStringTensor(const TfLiteTensor* tensor,
                             const std::string& data) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("data: \"" + data + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_22(mht_22_v, 895, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "CheckScalarStringTensor");

  ASSERT_EQ(tensor->dims->size, 0);
  ASSERT_EQ(tensor->type, kTfLiteString);
  StringRef str_ref = GetString(tensor, 0);
  EXPECT_EQ(std::string(str_ref.str, str_ref.len), data);
}

void CheckStringTensor(const TfLiteTensor* tensor,
                       const std::vector<int>& shape,
                       const std::vector<std::string>& data) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_23(mht_23_v, 907, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "CheckStringTensor");

  ASSERT_EQ(tensor->dims->size, shape.size());
  for (int i = 0; i < tensor->dims->size; ++i) {
    ASSERT_EQ(tensor->dims->data[i], shape[i]);
  }
  ASSERT_EQ(tensor->type, kTfLiteString);
  int count = GetStringCount(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    StringRef str_ref = GetString(tensor, i);
    EXPECT_EQ(std::string(str_ref.str, str_ref.len), data[i]);
  }
}
void CheckIntTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                    const std::vector<int32_t>& data) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_24(mht_24_v, 924, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "CheckIntTensor");

  ASSERT_EQ(tensor->dims->size, shape.size());
  for (int i = 0; i < tensor->dims->size; ++i) {
    ASSERT_EQ(tensor->dims->data[i], shape[i]);
  }
  ASSERT_EQ(tensor->type, kTfLiteInt32);
  int count = NumElements(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(tensor->data.i32[i], data[i]);
  }
}

void CheckBoolTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                     const std::vector<bool>& data) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTcc mht_25(mht_25_v, 941, "", "./tensorflow/lite/kernels/subgraph_test_util.cc", "CheckBoolTensor");

  ASSERT_EQ(tensor->dims->size, shape.size());
  for (int i = 0; i < tensor->dims->size; ++i) {
    ASSERT_EQ(tensor->dims->data[i], shape[i]);
  }
  ASSERT_EQ(tensor->type, kTfLiteBool);
  int count = NumElements(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(tensor->data.b[i], data[i]);
  }
}

}  // namespace subgraph_test_util
}  // namespace tflite
