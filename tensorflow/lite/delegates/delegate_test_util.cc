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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc() {
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

#include "tensorflow/lite/delegates/delegate_test_util.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace delegates {
namespace test_utils {

TfLiteRegistration AddOpRegistration() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_0(mht_0_v, 212, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "AddOpRegistration");

  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

  reg.custom_name = "my_add";
  reg.builtin_code = tflite::BuiltinOperator_CUSTOM;

  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "lambda");

    const TfLiteTensor* input1;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input1));
    const TfLiteTensor* input2;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &input2));
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

    // Verify that the two inputs have the same shape.
    TF_LITE_ENSURE_EQ(context, input1->dims->size, input2->dims->size);
    for (int i = 0; i < input1->dims->size; ++i) {
      TF_LITE_ENSURE_EQ(context, input1->dims->data[i], input2->dims->data[i]);
    }

    // Set output shape to match input shape.
    TF_LITE_ENSURE_STATUS(context->ResizeTensor(
        context, output, TfLiteIntArrayCopy(input1->dims)));
    return kTfLiteOk;
  };

  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_2(mht_2_v, 244, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "lambda");

    const TfLiteTensor* a0;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &a0));
    TF_LITE_ENSURE(context, a0);
    TF_LITE_ENSURE(context, a0->data.f);
    const TfLiteTensor* a1;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &a1));
    TF_LITE_ENSURE(context, a1);
    TF_LITE_ENSURE(context, a1->data.f);
    TfLiteTensor* out;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &out));
    TF_LITE_ENSURE(context, out);
    TF_LITE_ENSURE(context, out->data.f);
    // Set output data to element-wise sum of input data.
    int num = a0->dims->data[0];
    for (int i = 0; i < num; i++) {
      out->data.f[i] = a0->data.f[i] + a1->data.f[i];
    }
    return kTfLiteOk;
  };
  return reg;
}

void TestDelegation::SetUpSubgraph(Subgraph* subgraph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_3(mht_3_v, 270, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "TestDelegation::SetUpSubgraph");

  subgraph->AddTensors(5);
  subgraph->SetInputs({0, 1});
  subgraph->SetOutputs({3, 4});
  std::vector<int> dims({3});
  TfLiteQuantization quant{kTfLiteNoQuantization, nullptr};
  subgraph->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(3, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(4, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  TfLiteRegistration reg = AddOpRegistration();
  int node_index_ignored;
  subgraph->AddNodeWithParameters({0, 0}, {2}, {}, nullptr, 0, nullptr, &reg,
                                  &node_index_ignored);
  subgraph->AddNodeWithParameters({1, 1}, {3}, {}, nullptr, 0, nullptr, &reg,
                                  &node_index_ignored);
  subgraph->AddNodeWithParameters({2, 1}, {4}, {}, nullptr, 0, nullptr, &reg,
                                  &node_index_ignored);
}

void TestDelegation::AddSubgraphs(int subgraphs_to_add,
                                  int* first_new_subgraph_index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_4(mht_4_v, 300, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "TestDelegation::AddSubgraphs");

  interpreter_->AddSubgraphs(subgraphs_to_add, first_new_subgraph_index);
}

void TestDelegate::SetUp() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_5(mht_5_v, 307, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "TestDelegate::SetUp");

  interpreter_.reset(new Interpreter);
  SetUpSubgraph(&interpreter_->primary_subgraph());
}

void TestDelegate::TearDown() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_6(mht_6_v, 315, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "TestDelegate::TearDown");

  // Interpreter relies on delegate to free the resources properly. Thus
  // the life cycle of delegate must be longer than interpreter.
  interpreter_.reset();
  delegate_.reset();
  delegate2_.reset();
}

void TestTwoDelegates::SetUp() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_7(mht_7_v, 326, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "TestTwoDelegates::SetUp");

  interpreter_.reset(new Interpreter);
  SetUpSubgraph(&interpreter_->primary_subgraph());
}

void TestTwoDelegates::TearDown() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_8(mht_8_v, 334, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "TestTwoDelegates::TearDown");

  // Interpreter relies on delegate to free the resources properly. Thus
  // the life cycle of delegate must be longer than interpreter.
  interpreter_.reset();
  delegate_.reset();
  delegate2_.reset();
}

SimpleDelegate::SimpleDelegate(const std::vector<int>& nodes,
                               int64_t delegate_flags, bool fail_node_prepare,
                               int min_ops_per_subset, bool fail_node_invoke,
                               bool automatic_shape_propagation, bool custom_op,
                               bool set_output_tensor_dynamic)
    : nodes_(nodes),
      fail_delegate_node_prepare_(fail_node_prepare),
      min_ops_per_subset_(min_ops_per_subset),
      fail_delegate_node_invoke_(fail_node_invoke),
      automatic_shape_propagation_(automatic_shape_propagation),
      custom_op_(custom_op),
      set_output_tensor_dynamic_(set_output_tensor_dynamic) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_9(mht_9_v, 356, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "SimpleDelegate::SimpleDelegate");

  delegate_.Prepare = [](TfLiteContext* context,
                         TfLiteDelegate* delegate) -> TfLiteStatus {
    auto* simple = static_cast<SimpleDelegate*>(delegate->data_);
    TfLiteIntArray* nodes_to_separate =
        TfLiteIntArrayCreate(simple->nodes_.size());
    // Mark nodes that we want in TfLiteIntArray* structure.
    int index = 0;
    for (auto node_index : simple->nodes_) {
      nodes_to_separate->data[index++] = node_index;
      // make sure node is added
      TfLiteNode* node;
      TfLiteRegistration* reg;
      context->GetNodeAndRegistration(context, node_index, &node, &reg);
      if (simple->custom_op_) {
        TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_CUSTOM);
        TFLITE_CHECK_EQ(strcmp(reg->custom_name, "my_add"), 0);
      } else {
        TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_ADD);
      }
    }
    // Check that all nodes are available
    TfLiteIntArray* execution_plan;
    TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));
    for (int exec_index = 0; exec_index < execution_plan->size; exec_index++) {
      int node_index = execution_plan->data[exec_index];
      TfLiteNode* node;
      TfLiteRegistration* reg;
      context->GetNodeAndRegistration(context, node_index, &node, &reg);
      if (exec_index == node_index) {
        // Check op details only if it wasn't delegated already.
        if (simple->custom_op_) {
          TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_CUSTOM);
          TFLITE_CHECK_EQ(strcmp(reg->custom_name, "my_add"), 0);
        } else {
          TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_ADD);
        }
      }
    }

    // Get preview of delegate partitioning from the context.
    TfLiteDelegateParams* params_array;
    int num_partitions;
    TFLITE_CHECK_EQ(
        context->PreviewDelegatePartitioning(context, nodes_to_separate,
                                             &params_array, &num_partitions),
        kTfLiteOk);

    if (simple->min_ops_per_subset() > 0) {
      // Build a new vector of ops from subsets with at least the minimum
      // size.
      std::vector<int> allowed_ops;
      for (int idx = 0; idx < num_partitions; ++idx) {
        const auto* nodes_in_subset = params_array[idx].nodes_to_replace;
        if (nodes_in_subset->size < simple->min_ops_per_subset()) continue;
        allowed_ops.insert(allowed_ops.end(), nodes_in_subset->data,
                           nodes_in_subset->data + nodes_in_subset->size);
      }

      // Free existing nodes_to_separate & initialize a new array with
      // allowed_ops.
      TfLiteIntArrayFree(nodes_to_separate);
      nodes_to_separate = TfLiteIntArrayCreate(allowed_ops.size());
      memcpy(nodes_to_separate->data, allowed_ops.data(),
             sizeof(int) * nodes_to_separate->size);
    }

    // Another call to PreviewDelegatePartitioning should be okay, since
    // partitioning memory is managed by context.
    TFLITE_CHECK_EQ(
        context->PreviewDelegatePartitioning(context, nodes_to_separate,
                                             &params_array, &num_partitions),
        kTfLiteOk);

    context->ReplaceNodeSubsetsWithDelegateKernels(
        context, simple->FakeFusedRegistration(), nodes_to_separate, delegate);
    TfLiteIntArrayFree(nodes_to_separate);
    return kTfLiteOk;
  };
  delegate_.CopyToBufferHandle = [](TfLiteContext* context,
                                    TfLiteDelegate* delegate,
                                    TfLiteBufferHandle buffer_handle,
                                    TfLiteTensor* tensor) -> TfLiteStatus {
    // TODO(b/156586986): Implement tests to test buffer copying logic.
    return kTfLiteOk;
  };
  delegate_.CopyFromBufferHandle = [](TfLiteContext* context,
                                      TfLiteDelegate* delegate,
                                      TfLiteBufferHandle buffer_handle,
                                      TfLiteTensor* output) -> TfLiteStatus {
    TFLITE_CHECK_GE(buffer_handle, -1);
    TFLITE_CHECK_EQ(output->buffer_handle, buffer_handle);
    const float floats[] = {6., 6., 6.};
    int num = output->dims->data[0];
    for (int i = 0; i < num; i++) {
      output->data.f[i] = floats[i];
    }
    return kTfLiteOk;
  };

  delegate_.FreeBufferHandle =
      [](TfLiteContext* context, TfLiteDelegate* delegate,
         TfLiteBufferHandle* handle) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_10(mht_10_v, 461, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "lambda");
 *handle = kTfLiteNullBufferHandle; };
  // Store type-punned data SimpleDelegate structure.
  delegate_.data_ = static_cast<void*>(this);
  delegate_.flags = delegate_flags;
}

TfLiteRegistration SimpleDelegate::FakeFusedRegistration() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_11(mht_11_v, 470, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "SimpleDelegate::FakeFusedRegistration");

  TfLiteRegistration reg = {nullptr};
  reg.custom_name = "fake_fused_op";

  // Different flavors of the delegate kernel's Invoke(), dependent on
  // testing parameters.
  if (fail_delegate_node_invoke_) {
    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
      return kTfLiteError;
    };
  } else {
    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
      // Compute output data as elementwise sum of the two input arguments:
      //   func(x, y) = x + y
      // or for a single argument compute 2 * x:
      //   func(x) = x + x
      const TfLiteTensor* a0;
      const TfLiteTensor* a1;
      if (node->inputs->size == 2) {
        a0 = GetInput(context, node, 0);
        a1 = GetInput(context, node, 1);
      } else {
        a0 = GetInput(context, node, 0);
        a1 = a0;
      }
      TfLiteTensor* out;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &out));
      int num = 1;
      for (int i = 0; i < a0->dims->size; ++i) {
        num *= a0->dims->data[i];
      }
      for (int i = 0; i < num; i++) {
        out->data.f[i] = a0->data.f[i] + a1->data.f[i];
      }
      if (out->buffer_handle != kTfLiteNullBufferHandle) {
        // Make the data stale so that CopyFromBufferHandle can be invoked
        out->data_is_stale = true;
      }
      return kTfLiteOk;
    };
  }

  // Different flavors of the delegate kernel's Prepare(), dependent on
  // testing parameters.
  if (automatic_shape_propagation_) {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_12(mht_12_v, 518, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "lambda");

      // Shapes should already by propagated by the runtime, just need to
      // check.
      const TfLiteTensor* input1;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input1));
      TfLiteTensor* output;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
      const int input_dims_size = input1->dims->size;
      TF_LITE_ENSURE(context, output->dims->size == input_dims_size);
      for (int i = 0; i < input_dims_size; ++i) {
        TF_LITE_ENSURE(context, output->dims->data[i] == input1->dims->data[i]);
      }
      return kTfLiteOk;
    };
  } else if (fail_delegate_node_prepare_) {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_13(mht_13_v, 536, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "lambda");

      return kTfLiteError;
    };
  } else if (set_output_tensor_dynamic_) {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_14(mht_14_v, 543, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "lambda");

      TfLiteTensor* output;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
      SetTensorToDynamic(output);
      return kTfLiteOk;
    };
  } else {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_15(mht_15_v, 553, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "lambda");

      // Set output size to input size
      const TfLiteTensor* input1;
      const TfLiteTensor* input2;
      if (node->inputs->size == 2) {
        input1 = GetInput(context, node, 0);
        input2 = GetInput(context, node, 1);
      } else {
        input1 = GetInput(context, node, 0);
        input2 = input1;
      }
      TfLiteTensor* output;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

      TF_LITE_ENSURE_STATUS(context->ResizeTensor(
          context, output, TfLiteIntArrayCopy(input1->dims)));
      return kTfLiteOk;
    };
  }

  return reg;
}

std::unique_ptr<SimpleDelegate>
SimpleDelegate::DelegateWithRuntimeShapePropagation(
    const std::vector<int>& nodes, int64_t delegate_flags,
    int min_ops_per_subset) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_16(mht_16_v, 582, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "SimpleDelegate::DelegateWithRuntimeShapePropagation");

  return std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      nodes, delegate_flags, false /**fail_node_prepare**/,
      min_ops_per_subset /**min_ops_per_subset**/, false /**fail_node_invoke**/,
      true /**automatic_shape_propagation**/));
}

std::unique_ptr<SimpleDelegate> SimpleDelegate::DelegateWithDynamicOutput(
    const std::vector<int>& nodes) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_17(mht_17_v, 593, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "SimpleDelegate::DelegateWithDynamicOutput");

  // All params default except nodes & set_output_tensor_dynamic.
  return std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      nodes, kTfLiteDelegateFlagsAllowDynamicTensors,
      false /**fail_node_prepare**/, 0 /**min_ops_per_subset**/,
      false /**fail_node_invoke**/, false /**automatic_shape_propagation**/,
      true /**custom_op**/, true /**set_output_tensor_dynamic**/));
}

void TestFP16Delegation::SetUp() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_18(mht_18_v, 605, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "TestFP16Delegation::SetUp");

  interpreter_.reset(new Interpreter);
  interpreter_->AddTensors(13);
  interpreter_->SetInputs({0});
  interpreter_->SetOutputs({12});

  float16_const_ = Eigen::half(2.f);

  // TENSORS.
  TfLiteQuantizationParams quant;
  // Input.
  interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {1}, quant);
  // fp16 constant, dequantize output, Add0 output.
  interpreter_->SetTensorParametersReadOnly(
      1, kTfLiteFloat16, "", {1}, quant,
      reinterpret_cast<const char*>(&float16_const_), sizeof(TfLiteFloat16));
  interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {1}, quant);
  interpreter_->SetTensorParametersReadWrite(3, kTfLiteFloat32, "", {1}, quant);
  // fp16 constant, dequantize output, Add1 output.
  interpreter_->SetTensorParametersReadOnly(
      4, kTfLiteFloat16, "", {1}, quant,
      reinterpret_cast<const char*>(&float16_const_), sizeof(TfLiteFloat16));
  interpreter_->SetTensorParametersReadWrite(5, kTfLiteFloat32, "", {1}, quant);
  interpreter_->SetTensorParametersReadWrite(6, kTfLiteFloat32, "", {1}, quant);
  // fp16 constant, dequantize output, Mul0 output.
  interpreter_->SetTensorParametersReadOnly(
      7, kTfLiteFloat16, "", {1}, quant,
      reinterpret_cast<const char*>(&float16_const_), sizeof(TfLiteFloat16));
  interpreter_->SetTensorParametersReadWrite(8, kTfLiteFloat32, "", {1}, quant);
  interpreter_->SetTensorParametersReadWrite(9, kTfLiteFloat32, "", {1}, quant);
  // fp16 constant, dequantize output, Add2 output.
  interpreter_->SetTensorParametersReadOnly(
      10, kTfLiteFloat16, "", {1}, quant,
      reinterpret_cast<const char*>(&float16_const_), sizeof(TfLiteFloat16));
  interpreter_->SetTensorParametersReadWrite(11, kTfLiteFloat32, "", {1},
                                             quant);
  interpreter_->SetTensorParametersReadWrite(12, kTfLiteFloat32, "", {1},
                                             quant);

  // NODES.
  auto* add_reg = ops::builtin::Register_ADD();
  auto* mul_reg = ops::builtin::Register_MUL();
  auto* deq_reg = ops::builtin::Register_DEQUANTIZE();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  deq_reg->builtin_code = kTfLiteBuiltinDequantize;
  mul_reg->builtin_code = kTfLiteBuiltinMul;
  TfLiteAddParams* builtin_data0 =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  TfLiteAddParams* builtin_data1 =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  TfLiteMulParams* builtin_data2 =
      reinterpret_cast<TfLiteMulParams*>(malloc(sizeof(TfLiteMulParams)));
  TfLiteAddParams* builtin_data3 =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data0->activation = kTfLiteActNone;
  builtin_data1->activation = kTfLiteActNone;
  builtin_data2->activation = kTfLiteActNone;
  builtin_data3->activation = kTfLiteActNone;
  interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr, deq_reg);
  interpreter_->AddNodeWithParameters({0, 2}, {3}, nullptr, 0, builtin_data0,
                                      add_reg);
  interpreter_->AddNodeWithParameters({4}, {5}, nullptr, 0, nullptr, deq_reg);
  interpreter_->AddNodeWithParameters({3, 5}, {6}, nullptr, 0, builtin_data1,
                                      add_reg);
  interpreter_->AddNodeWithParameters({7}, {8}, nullptr, 0, nullptr, deq_reg);
  interpreter_->AddNodeWithParameters({6, 8}, {9}, nullptr, 0, builtin_data2,
                                      mul_reg);
  interpreter_->AddNodeWithParameters({10}, {11}, nullptr, 0, nullptr, deq_reg);
  interpreter_->AddNodeWithParameters({9, 11}, {12}, nullptr, 0, builtin_data3,
                                      add_reg);
}

void TestFP16Delegation::VerifyInvoke() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_19(mht_19_v, 680, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "TestFP16Delegation::VerifyInvoke");

  std::vector<float> input = {3.0f};
  std::vector<float> expected_output = {16.0f};

  const int input_tensor_idx = interpreter_->inputs()[0];
  const int output_tensor_idx = interpreter_->outputs()[0];

  memcpy(interpreter_->typed_tensor<float>(input_tensor_idx), input.data(),
         sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output_tensor = interpreter_->tensor(output_tensor_idx);
  for (int i = 0; i < 1; ++i) {
    EXPECT_EQ(output_tensor->data.f[i], expected_output[i]) << i;
  }
}

TestFP16Delegation::FP16Delegate::FP16Delegate(int num_delegated_subsets,
                                               bool fail_node_prepare,
                                               bool fail_node_invoke)
    : num_delegated_subsets_(num_delegated_subsets),
      fail_delegate_node_prepare_(fail_node_prepare),
      fail_delegate_node_invoke_(fail_node_invoke) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_20(mht_20_v, 704, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "TestFP16Delegation::FP16Delegate::FP16Delegate");

  delegate_.Prepare = [](TfLiteContext* context,
                         TfLiteDelegate* delegate) -> TfLiteStatus {
    auto* fp16_delegate = static_cast<FP16Delegate*>(delegate->data_);
    // FP16 graph partitioning.
    delegates::IsNodeSupportedFn node_supported_fn =
        [=](TfLiteContext* context, TfLiteNode* node,
            TfLiteRegistration* registration,
            std::string* unsupported_details) -> bool {
      return registration->builtin_code == kTfLiteBuiltinAdd;
    };
    delegates::FP16GraphPartitionHelper partition_helper(context,
                                                         node_supported_fn);
    TfLiteIntArray* nodes_to_separate = nullptr;
    if (partition_helper.Partition(nullptr) != kTfLiteOk) {
      nodes_to_separate = TfLiteIntArrayCreate(0);
    } else {
      std::vector<int> ops_to_replace =
          partition_helper.GetNodesOfFirstNLargestPartitions(
              fp16_delegate->num_delegated_subsets());
      nodes_to_separate = ConvertVectorToTfLiteIntArray(ops_to_replace);
    }

    context->ReplaceNodeSubsetsWithDelegateKernels(
        context, fp16_delegate->FakeFusedRegistration(), nodes_to_separate,
        delegate);
    TfLiteIntArrayFree(nodes_to_separate);
    return kTfLiteOk;
  };
  delegate_.CopyFromBufferHandle =
      [](TfLiteContext* context, TfLiteDelegate* delegate,
         TfLiteBufferHandle buffer_handle,
         TfLiteTensor* output) -> TfLiteStatus { return kTfLiteOk; };
  delegate_.FreeBufferHandle = nullptr;
  delegate_.CopyToBufferHandle = nullptr;
  // Store type-punned data SimpleDelegate structure.
  delegate_.data_ = static_cast<void*>(this);
  delegate_.flags = kTfLiteDelegateFlagsNone;
}

TfLiteRegistration TestFP16Delegation::FP16Delegate::FakeFusedRegistration() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_21(mht_21_v, 747, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "TestFP16Delegation::FP16Delegate::FakeFusedRegistration");

  TfLiteRegistration reg = {nullptr};
  reg.custom_name = "fake_fp16_add_op";

  // Different flavors of the delegate kernel's Invoke(), dependent on
  // testing parameters.
  if (fail_delegate_node_invoke_) {
    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
      return kTfLiteError;
    };
  } else {
    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
      float output = 0;
      for (int i = 0; i < node->inputs->size; ++i) {
        const TfLiteTensor* input_tensor = GetInput(context, node, i);
        if (input_tensor->type == kTfLiteFloat32) {
          output += input_tensor->data.f[0];
        } else {
          // All constants are 2.
          output += 2;
        }
      }
      TfLiteTensor* out = GetOutput(context, node, 0);
      out->data.f[0] = output;
      return kTfLiteOk;
    };
  }

  // Different flavors of the delegate kernel's Prepare(), dependent on
  // testing parameters.
  if (fail_delegate_node_prepare_) {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_22(mht_22_v, 781, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "lambda");

      return kTfLiteError;
    };
  } else {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTcc mht_23(mht_23_v, 788, "", "./tensorflow/lite/delegates/delegate_test_util.cc", "lambda");

      // Set output size to input size
      const TfLiteTensor* input = GetInput(context, node, 0);
      TfLiteTensor* output = GetOutput(context, node, 0);
      TF_LITE_ENSURE_STATUS(context->ResizeTensor(
          context, output, TfLiteIntArrayCopy(input->dims)));
      return kTfLiteOk;
    };
  }

  return reg;
}

}  // namespace test_utils
}  // namespace delegates
}  // namespace tflite
