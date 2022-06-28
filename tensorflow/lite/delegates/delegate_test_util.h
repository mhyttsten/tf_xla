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
#ifndef TENSORFLOW_LITE_DELEGATES_DELEGATE_TEST_UTIL_
#define TENSORFLOW_LITE_DELEGATES_DELEGATE_TEST_UTIL_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTh() {
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

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace delegates {
namespace test_utils {

// Build a kernel registration for a custom addition op that adds its two
// tensor inputs to produce a tensor output.
TfLiteRegistration AddOpRegistration();

class SimpleDelegate {
 public:
  // Create a simple implementation of a TfLiteDelegate. We use the C++ class
  // SimpleDelegate and it can produce a handle TfLiteDelegate that is
  // value-copyable and compatible with TfLite.
  //
  // Parameters:
  //   nodes: Indices of the graph nodes that the delegate will handle.
  //   fail_node_prepare: To simulate failure of Delegate node's Prepare().
  //   min_ops_per_subset: If >0, partitioning preview is used to choose only
  //     those subsets with min_ops_per_subset number of nodes.
  //   fail_node_invoke: To simulate failure of Delegate node's Invoke().
  //   automatic_shape_propagation: This assumes that the runtime will
  //     propagate shapes using the original execution plan.
  //   custom_op: If true, the graph nodes specified in the 'nodes' parameter
  //     should be custom ops with name "my_add"; if false, they should be
  //     the builtin ADD operator.
  //   set_output_tensor_dynamic: If True, this delegate sets output tensor to
  //     as dynamic during kernel Prepare.
  explicit SimpleDelegate(const std::vector<int>& nodes,
                          int64_t delegate_flags = kTfLiteDelegateFlagsNone,
                          bool fail_node_prepare = false,
                          int min_ops_per_subset = 0,
                          bool fail_node_invoke = false,
                          bool automatic_shape_propagation = false,
                          bool custom_op = true,
                          bool set_output_tensor_dynamic = false);

  static std::unique_ptr<SimpleDelegate> DelegateWithRuntimeShapePropagation(
      const std::vector<int>& nodes, int64_t delegate_flags,
      int min_ops_per_subset);

  static std::unique_ptr<SimpleDelegate> DelegateWithDynamicOutput(
      const std::vector<int>& nodes);

  TfLiteRegistration FakeFusedRegistration();

  TfLiteDelegate* get_tf_lite_delegate() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTh mht_0(mht_0_v, 243, "", "./tensorflow/lite/delegates/delegate_test_util.h", "get_tf_lite_delegate");
 return &delegate_; }

  int min_ops_per_subset() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTh mht_1(mht_1_v, 248, "", "./tensorflow/lite/delegates/delegate_test_util.h", "min_ops_per_subset");
 return min_ops_per_subset_; }

 private:
  std::vector<int> nodes_;
  TfLiteDelegate delegate_;
  bool fail_delegate_node_prepare_ = false;
  int min_ops_per_subset_ = 0;
  bool fail_delegate_node_invoke_ = false;
  bool automatic_shape_propagation_ = false;
  bool custom_op_ = true;
  bool set_output_tensor_dynamic_ = false;
};

// Base class for single/multiple delegate tests.
// Friend of Interpreter to access private methods.
class TestDelegation {
 protected:
  TfLiteStatus RemoveAllDelegates() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTh mht_2(mht_2_v, 268, "", "./tensorflow/lite/delegates/delegate_test_util.h", "RemoveAllDelegates");

    return interpreter_->RemoveAllDelegates();
  }

  void SetUpSubgraph(Subgraph* subgraph);
  void AddSubgraphs(int subgraphs_to_add,
                    int* first_new_subgraph_index = nullptr);

  std::unique_ptr<Interpreter> interpreter_;
};

// Tests scenarios involving a single delegate.
class TestDelegate : public TestDelegation, public ::testing::Test {
 protected:
  void SetUp() override;

  void TearDown() override;

  TfLiteBufferHandle last_allocated_handle_ = kTfLiteNullBufferHandle;

  TfLiteBufferHandle AllocateBufferHandle() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTh mht_3(mht_3_v, 291, "", "./tensorflow/lite/delegates/delegate_test_util.h", "AllocateBufferHandle");
 return ++last_allocated_handle_; }

  std::unique_ptr<SimpleDelegate> delegate_, delegate2_;
};

// Tests scenarios involving two delegates, parametrized by the first & second
// delegate's flags.
class TestTwoDelegates
    : public TestDelegation,
      public ::testing::TestWithParam<
          std::pair<TfLiteDelegateFlags, TfLiteDelegateFlags>> {
 protected:
  void SetUp() override;

  void TearDown() override;

  std::unique_ptr<SimpleDelegate> delegate_, delegate2_;
};

// Tests delegate functionality related to FP16 graphs.
// Model architecture:
// 1->DEQ->2   4->DEQ->5   7->DEQ->8   10->DEQ->11
//         |           |           |            |
// 0----->ADD->3----->ADD->6----->MUL->9------>ADD-->12
// Input: 0, Output:12.
// All constants are 2, so the function is: (x + 2 + 2) * 2 + 2 = 2x + 10
//
// Delegate only supports ADD, so can have up to two delegated partitions.
// TODO(b/156707497): Add more cases here once we have landed CPU kernels
// supporting FP16.
class TestFP16Delegation : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override;

  void VerifyInvoke();

  void TearDown() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTh mht_4(mht_4_v, 330, "", "./tensorflow/lite/delegates/delegate_test_util.h", "TearDown");
 interpreter_.reset(); }

 protected:
  class FP16Delegate {
   public:
    // Uses FP16GraphPartitionHelper to accept ADD nodes with fp16 input.
    explicit FP16Delegate(int num_delegated_subsets,
                          bool fail_node_prepare = false,
                          bool fail_node_invoke = false);

    TfLiteRegistration FakeFusedRegistration();

    TfLiteDelegate* get_tf_lite_delegate() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTh mht_5(mht_5_v, 345, "", "./tensorflow/lite/delegates/delegate_test_util.h", "get_tf_lite_delegate");
 return &delegate_; }

    int num_delegated_subsets() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSdelegate_test_utilDTh mht_6(mht_6_v, 350, "", "./tensorflow/lite/delegates/delegate_test_util.h", "num_delegated_subsets");
 return num_delegated_subsets_; }

   private:
    TfLiteDelegate delegate_;
    int num_delegated_subsets_;
    bool fail_delegate_node_prepare_ = false;
    bool fail_delegate_node_invoke_ = false;
  };

  std::unique_ptr<Interpreter> interpreter_;
  std::unique_ptr<FP16Delegate> delegate_;
  Eigen::half float16_const_;
};

}  // namespace test_utils
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_DELEGATE_TEST_UTIL_
