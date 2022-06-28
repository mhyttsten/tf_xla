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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc() {
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
#include "tensorflow/lite/delegates/utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace delegates {
namespace {

TEST(UtilsTest, CreateNewTensorWithDifferentTypeTest) {
  std::vector<TfLiteTensor> tensors(2);
  // Data about original tensor.
  // The same shape should be reflected in tensors[1] later.
  tensors[0].dims = TfLiteIntArrayCreate(2);
  tensors[0].dims->data[0] = 2;
  tensors[0].dims->data[1] = 3;
  tensors[0].type = kTfLiteFloat32;
  // To simulate a valid TFLite Context.
  TfLiteContext context;
  context.AddTensors = [](struct TfLiteContext*, int tensors_to_add,
                          int* first_new_tensor_index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/delegates/utils_test.cc", "lambda");

    // The util should be adding exactly one tensor to the graph.
    if (tensors_to_add != 1) {
      return kTfLiteError;
    }
    // This ensures that the 'new tensor' is the second tensor in the vector
    // above.
    *first_new_tensor_index = 1;
    return kTfLiteOk;
  };
  context.ResizeTensor = [](struct TfLiteContext*, TfLiteTensor* tensor,
                            TfLiteIntArray* new_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/delegates/utils_test.cc", "lambda");

    // Ensure dimensions are the same as the original tensor.
    if (new_size->size != 2 || new_size->data[0] != 2 || new_size->data[1] != 3)
      return kTfLiteError;
    tensor->dims = new_size;
    return kTfLiteOk;
  };
  context.tensors = tensors.data();

  TfLiteTensor* new_tensor = nullptr;
  int new_tensor_index = -1;
  EXPECT_EQ(CreateNewTensorWithDifferentType(
                &context, /**original_tensor_index**/ 0,
                /**new_type**/ kTfLiteUInt8, &new_tensor, &new_tensor_index),
            kTfLiteOk);
  EXPECT_EQ(new_tensor_index, 1);
  EXPECT_NE(new_tensor, nullptr);
  EXPECT_NE(new_tensor->dims, nullptr);
  EXPECT_EQ(new_tensor->type, kTfLiteUInt8);
  EXPECT_EQ(new_tensor->allocation_type, kTfLiteArenaRw);

  // Cleanup.
  TfLiteIntArrayFree(tensors[0].dims);
  TfLiteIntArrayFree(tensors[1].dims);
}

// A mock TfLiteContext to be used for GraphPartitionHelperTest.
class MockTfLiteContext : public TfLiteContext {
 public:
  MockTfLiteContext() : TfLiteContext({0}) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_2(mht_2_v, 253, "", "./tensorflow/lite/delegates/utils_test.cc", "MockTfLiteContext");

    // Simply create a 10-node execution plan.
    exec_plan_ = TfLiteIntArrayCreate(10);
    for (int i = 0; i < 10; ++i) exec_plan_->data[i] = i;

    // Create {1}, {0,3,7,8}, {2,4,9}, {5,6} 4 partitions.
    TfLiteDelegateParams params1({nullptr});
    params1.nodes_to_replace = TfLiteIntArrayCreate(1);
    params1.nodes_to_replace->data[0] = 1;
    delegate_params_.emplace_back(params1);

    TfLiteDelegateParams params2({nullptr});
    params2.nodes_to_replace = TfLiteIntArrayCreate(4);
    params2.nodes_to_replace->data[0] = 0;
    params2.nodes_to_replace->data[1] = 3;
    params2.nodes_to_replace->data[2] = 7;
    params2.nodes_to_replace->data[3] = 8;
    delegate_params_.emplace_back(params2);

    TfLiteDelegateParams params3({nullptr});
    params3.nodes_to_replace = TfLiteIntArrayCreate(3);
    params3.nodes_to_replace->data[0] = 2;
    params3.nodes_to_replace->data[1] = 4;
    params3.nodes_to_replace->data[2] = 9;
    delegate_params_.emplace_back(params3);

    TfLiteDelegateParams params4({nullptr});
    params4.nodes_to_replace = TfLiteIntArrayCreate(2);
    params4.nodes_to_replace->data[0] = 5;
    params4.nodes_to_replace->data[1] = 6;
    delegate_params_.emplace_back(params4);

    // We need to mock the following 3 functions inside TfLiteContext object
    // that are used by GraphPartitionHelper implementation.
    this->GetExecutionPlan = MockGetExecutionPlan;
    this->GetNodeAndRegistration = MockGetNodeAndRegistration;
    this->PreviewDelegatePartitioning = MockPreviewDelegatePartitioning;
  }
  ~MockTfLiteContext() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_3(mht_3_v, 294, "", "./tensorflow/lite/delegates/utils_test.cc", "~MockTfLiteContext");

    TfLiteIntArrayFree(exec_plan_);
    for (auto params : delegate_params_) {
      TfLiteIntArrayFree(params.nodes_to_replace);
      TfLiteIntArrayFree(params.input_tensors);
      TfLiteIntArrayFree(params.output_tensors);
    }
  }

  TfLiteIntArray* exec_plan() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_4(mht_4_v, 306, "", "./tensorflow/lite/delegates/utils_test.cc", "exec_plan");
 return exec_plan_; }
  TfLiteNode* node() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_5(mht_5_v, 310, "", "./tensorflow/lite/delegates/utils_test.cc", "node");
 return &node_; }
  TfLiteRegistration* registration() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_6(mht_6_v, 314, "", "./tensorflow/lite/delegates/utils_test.cc", "registration");
 return &registration_; }
  TfLiteDelegateParams* delegate_params() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_7(mht_7_v, 318, "", "./tensorflow/lite/delegates/utils_test.cc", "delegate_params");
 return &delegate_params_.front(); }
  int num_delegate_params() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_8(mht_8_v, 322, "", "./tensorflow/lite/delegates/utils_test.cc", "num_delegate_params");
 return delegate_params_.size(); }

 private:
  static TfLiteStatus MockGetExecutionPlan(TfLiteContext* context,
                                           TfLiteIntArray** execution_plan) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_9(mht_9_v, 329, "", "./tensorflow/lite/delegates/utils_test.cc", "MockGetExecutionPlan");

    MockTfLiteContext* mock = reinterpret_cast<MockTfLiteContext*>(context);
    *execution_plan = mock->exec_plan();
    return kTfLiteOk;
  }

  static TfLiteStatus MockGetNodeAndRegistration(
      TfLiteContext* context, int node_index, TfLiteNode** node,
      TfLiteRegistration** registration) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_10(mht_10_v, 340, "", "./tensorflow/lite/delegates/utils_test.cc", "MockGetNodeAndRegistration");

    MockTfLiteContext* mock = reinterpret_cast<MockTfLiteContext*>(context);
    *node = mock->node();
    *registration = mock->registration();
    return kTfLiteOk;
  }

  static TfLiteStatus MockPreviewDelegatePartitioning(
      TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
      TfLiteDelegateParams** partition_params_array, int* num_partitions) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_11(mht_11_v, 352, "", "./tensorflow/lite/delegates/utils_test.cc", "MockPreviewDelegatePartitioning");

    MockTfLiteContext* mock = reinterpret_cast<MockTfLiteContext*>(context);
    *partition_params_array = mock->delegate_params();
    *num_partitions = mock->num_delegate_params();
    return kTfLiteOk;
  }

  // The execution plan of this mocked TfLiteContext object.
  TfLiteIntArray* exec_plan_;

  // For simplicity, the mocked graph has only type of node and one
  // registration.
  TfLiteNode node_;
  TfLiteRegistration registration_;

  // The TfLiteDelegateParams object that's manually populated inside the mocked
  // TfLiteContext::PreviewDelegatePartitioning.
  std::vector<TfLiteDelegateParams> delegate_params_;
};

bool IsNodeSupported(TfLiteContext* context, TfLiteNode* node,
                     TfLiteRegistration* registration,
                     std::string* unsupported_details) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_12(mht_12_v, 377, "", "./tensorflow/lite/delegates/utils_test.cc", "IsNodeSupported");

  return true;
}

std::vector<int> GetNodesToReplaceFromPartitions(
    const std::vector<TfLiteDelegateParams*>& partitions) {
  std::vector<int> nodes;
  for (const auto p : partitions) {
    nodes.insert(nodes.end(), p->nodes_to_replace->data,
                 p->nodes_to_replace->data + p->nodes_to_replace->size);
  }
  return nodes;
}

TEST(GraphPartitionHelper, CheckPartitions) {
  // The mocked TfLiteContext has 4 partitions: {1}, {0,3,7,8}, {2,4,9}, {5,6}.
  MockTfLiteContext mocked_context;
  GraphPartitionHelper helper(&mocked_context, IsNodeSupported);
  EXPECT_EQ(kTfLiteOk, helper.Partition(nullptr));
  EXPECT_EQ(10, helper.num_total_nodes());
  EXPECT_EQ(4, helper.num_partitions());

  auto partitions = helper.GetFirstNLargestPartitions(1, 0);
  EXPECT_EQ(1, partitions.size());
  auto nodes = GetNodesToReplaceFromPartitions(partitions);
  EXPECT_THAT(nodes, testing::ElementsAreArray({0, 3, 7, 8}));

  // Get the largest partition but requiring at least 5 nodes, so empty result.
  partitions = helper.GetFirstNLargestPartitions(1, 5);
  EXPECT_TRUE(partitions.empty());

  partitions = helper.GetFirstNLargestPartitions(10, 3);
  EXPECT_EQ(2, partitions.size());
  EXPECT_EQ(4, partitions[0]->nodes_to_replace->size);
  EXPECT_EQ(3, partitions[1]->nodes_to_replace->size);
  nodes = GetNodesToReplaceFromPartitions(partitions);
  EXPECT_THAT(nodes, testing::ElementsAreArray({0, 3, 7, 8, 2, 4, 9}));
}

TfLiteStatus ErrorGetExecutionPlan(TfLiteContext* context,
                                   TfLiteIntArray** execution_plan) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_13(mht_13_v, 420, "", "./tensorflow/lite/delegates/utils_test.cc", "ErrorGetExecutionPlan");

  return kTfLiteError;
}

void EmptyReportError(TfLiteContext* context, const char* format, ...) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutils_testDTcc mht_14(mht_14_v, 428, "", "./tensorflow/lite/delegates/utils_test.cc", "EmptyReportError");
}

TEST(GraphPartitionHelper, CheckPrepareErrors) {
  TfLiteContext error_context({0});
  error_context.GetExecutionPlan = ErrorGetExecutionPlan;
  error_context.ReportError = EmptyReportError;
  GraphPartitionHelper helper(&error_context, IsNodeSupported);
  EXPECT_EQ(kTfLiteError, helper.Partition(nullptr));
}

TEST(GraphPartitionHelper, CheckPartitionsWithSupportedNodeList) {
  // The mocked TfLiteContext has 4 partitions: {1}, {0,3,7,8}, {2,4,9}, {5,6}.
  // So, we simply create a list of supported nodes as {0,1,2,...,8,9}
  MockTfLiteContext mocked_context;
  std::vector<int> supported_nodes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  GraphPartitionHelper helper(&mocked_context, supported_nodes);
  EXPECT_EQ(kTfLiteOk, helper.Partition(nullptr));
  EXPECT_EQ(10, helper.num_total_nodes());
  EXPECT_EQ(4, helper.num_partitions());

  auto partitions = helper.GetFirstNLargestPartitions(1, 0);
  EXPECT_EQ(1, partitions.size());
  auto nodes = GetNodesToReplaceFromPartitions(partitions);
  EXPECT_THAT(nodes, testing::ElementsAreArray({0, 3, 7, 8}));

  // Get the largest partition but requiring at least 5 nodes, so empty result.
  partitions = helper.GetFirstNLargestPartitions(1, 5);
  EXPECT_TRUE(partitions.empty());

  partitions = helper.GetFirstNLargestPartitions(10, 3);
  EXPECT_EQ(2, partitions.size());
  EXPECT_EQ(4, partitions[0]->nodes_to_replace->size);
  EXPECT_EQ(3, partitions[1]->nodes_to_replace->size);
  nodes = GetNodesToReplaceFromPartitions(partitions);
  EXPECT_THAT(nodes, testing::ElementsAreArray({0, 3, 7, 8, 2, 4, 9}));
}

}  // namespace
}  // namespace delegates
}  // namespace tflite
