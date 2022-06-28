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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_op_testDTcc() {
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

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "reduce_dataset";

class ReduceDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ReduceDatasetParams(T input_dataset_params, std::vector<Tensor> initial_state,
                      std::vector<Tensor> other_arguments,
                      FunctionDefHelper::AttrValueWrapper func,
                      std::vector<FunctionDef> func_lib,
                      DataTypeVector type_state, DataTypeVector type_arguments,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      bool use_inter_op_parallelism, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        initial_state_(std::move(initial_state)),
        other_arguments_(std::move(other_arguments)),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_state_(std::move(type_state)),
        type_arguments_(std::move(type_arguments)),
        use_inter_op_parallelism_(use_inter_op_parallelism) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_op_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/data/reduce_dataset_op_test.cc", "ReduceDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors = initial_state_;
    input_tensors.insert(input_tensors.end(), other_arguments_.begin(),
                         other_arguments_.end());
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_op_testDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/kernels/data/reduce_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    input_names->emplace_back("input_dataset");
    for (int i = 0; i < initial_state_.size(); ++i) {
      input_names->emplace_back(strings::StrCat("initial_state_", i));
    }
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(strings::StrCat("other_arguments_", i));
    }
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_op_testDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/kernels/data/reduce_dataset_op_test.cc", "GetAttributes");

    attr_vector->clear();
    *attr_vector = {{"f", func_},
                    {"Tstate", type_state_},
                    {"Targuments", type_arguments_},
                    {"output_types", output_dtypes_},
                    {"output_shapes", output_shapes_},
                    {"use_inter_op_parallelism", use_inter_op_parallelism_},
                    {"metadata", ""}};
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_op_testDTcc mht_3(mht_3_v, 260, "", "./tensorflow/core/kernels/data/reduce_dataset_op_test.cc", "dataset_type");
 return "Reduce"; }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

 private:
  std::vector<Tensor> initial_state_;
  std::vector<Tensor> other_arguments_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_state_;
  DataTypeVector type_arguments_;
  bool use_inter_op_parallelism_;
};

class ReduceDatasetOpTest : public DatasetOpsTestBase {};

// Test case 1: the input function has one output.
ReduceDatasetParams ReduceDatasetParams1() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_op_testDTcc mht_4(mht_4_v, 280, "", "./tensorflow/core/kernels/data/reduce_dataset_op_test.cc", "ReduceDatasetParams1");

  return ReduceDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(0, 10, 1),
      /*initial_state=*/CreateTensors<int64_t>(TensorShape({}), {{1}}),
      /*other_arguments=*/{},
      /*func=*/FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XAddY()},
      /*type_state=*/{DT_INT64},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*node_name=*/kNodeName);
}

// Test case 2: the reduce function has two inputs and two outputs. As the
// number of components of initial_state need to match with the reduce function
// outputs, the initial_state will have two components. It results in that
// the components of initial_state will be all the inputs for the reduce
// function, and the input dataset will not be involved in the
// reduce/aggregation process.
ReduceDatasetParams ReduceDatasetParams2() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_op_testDTcc mht_5(mht_5_v, 304, "", "./tensorflow/core/kernels/data/reduce_dataset_op_test.cc", "ReduceDatasetParams2");

  return ReduceDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(1, 10, 1),
      /*initial_state=*/CreateTensors<int64_t>(TensorShape({}), {{1}, {1}}),
      /*other_arguments=*/{},
      /*func=*/
      FunctionDefHelper::FunctionRef("XPlusOneXTimesY", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XPlusOneXTimesY()},
      /*type_state=*/{DT_INT64, DT_INT64},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*node_name=*/kNodeName);
}

// Test case 3: the input dataset has no outputs, so the reduce dataset just
// returns the initial state.
ReduceDatasetParams ReduceDatasetParams3() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_op_testDTcc mht_6(mht_6_v, 325, "", "./tensorflow/core/kernels/data/reduce_dataset_op_test.cc", "ReduceDatasetParams3");

  return ReduceDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(0, 0, 1),
      /*initial_state=*/CreateTensors<int64_t>(TensorShape({}), {{1}, {3}}),
      /*other_arguments=*/{},
      /*func=*/
      FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XAddY()},
      /*type_state=*/{DT_INT64, DT_INT64},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<ReduceDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/
           ReduceDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{46}})},
          {/*dataset_params=*/ReduceDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}),
                                  {{10}, {1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9}})},
          {/*dataset_params=*/
           ReduceDatasetParams3(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{1}, {3}})}};
}

class ParameterizedReduceDatasetOpTest
    : public ReduceDatasetOpTest,
      public ::testing::WithParamInterface<
          GetNextTestCase<ReduceDatasetParams>> {};

TEST_P(ParameterizedReduceDatasetOpTest, Compute) {
  auto test_case = GetParam();
  TF_ASSERT_OK(InitializeRuntime(test_case.dataset_params));
  std::vector<Tensor> output;
  TF_ASSERT_OK(RunDatasetOp(test_case.dataset_params, &output));
  TF_EXPECT_OK(
      ExpectEqual(test_case.expected_outputs, output, /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(ReduceDatasetOpTest, ParameterizedReduceDatasetOpTest,
                         ::testing::ValuesIn(GetNextTestCases()));

}  // namespace
}  // namespace data
}  // namespace tensorflow
