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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/tensor_dataset_op.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "tensor_dataset";

class TensorDatasetParams : public DatasetParams {
 public:
  TensorDatasetParams(std::vector<Tensor> components, string node_name)
      : DatasetParams(TensorDtypes(components), TensorShapes(components),
                      std::move(node_name)),
        components_(std::move(components)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_dataset_op_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/data/tensor_dataset_op_test.cc", "TensorDatasetParams");
}

  std::vector<Tensor> GetInputTensors() const override { return components_; }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_dataset_op_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/kernels/data/tensor_dataset_op_test.cc", "GetInputNames");

    input_names->reserve(components_.size());
    for (int i = 0; i < components_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(TensorDatasetOp::kComponents, "_", i));
    }
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_dataset_op_testDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/kernels/data/tensor_dataset_op_test.cc", "GetAttributes");

    *attr_vector = {{"Toutput_types", output_dtypes_},
                    {"output_shapes", output_shapes_},
                    {"metadata", ""}};
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_dataset_op_testDTcc mht_3(mht_3_v, 234, "", "./tensorflow/core/kernels/data/tensor_dataset_op_test.cc", "dataset_type");
 return TensorDatasetOp::kDatasetType; }

 private:
  DataTypeVector TensorDtypes(const std::vector<Tensor>& input_components) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_dataset_op_testDTcc mht_4(mht_4_v, 240, "", "./tensorflow/core/kernels/data/tensor_dataset_op_test.cc", "TensorDtypes");

    DataTypeVector dtypes;
    for (const auto& component : input_components) {
      dtypes.emplace_back(component.dtype());
    }
    return dtypes;
  }

  std::vector<PartialTensorShape> TensorShapes(
      const std::vector<Tensor>& input_components) {
    std::vector<PartialTensorShape> shapes;
    for (const auto& component : input_components) {
      shapes.emplace_back(component.shape());
    }
    return shapes;
  }

 public:
  std::vector<Tensor> components_;
};

class TensorDatasetOpTest : public DatasetOpsTestBase {};

std::vector<Tensor> PlainTensors() {
  return {CreateTensor<int64_t>(TensorShape({}), {1}),
          CreateTensor<int64_t>(TensorShape({1, 3}), {1, 2, 3}),
          CreateTensor<double>(TensorShape({}), {37.0}),
          CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})};
}

// Test case 1: test a dataset that represents a single tuple of plain tensors.
TensorDatasetParams PlainTensorDatasetParams() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_dataset_op_testDTcc mht_5(mht_5_v, 274, "", "./tensorflow/core/kernels/data/tensor_dataset_op_test.cc", "PlainTensorDatasetParams");

  return {/*components=*/PlainTensors(),
          /*node_name=*/kNodeName};
}

// Test case 2: test a dataset that represents a tuple of nested tensors.
TensorDatasetParams NestedTensorDatasetParams() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_dataset_op_testDTcc mht_6(mht_6_v, 283, "", "./tensorflow/core/kernels/data/tensor_dataset_op_test.cc", "NestedTensorDatasetParams");

  return {/*components=*/
          {CreateTensor<Variant>(TensorShape({}),
                                 {CreateTensor<double>(TensorShape({2, 2}),
                                                       {1.0, 2.0, 3.0, 4.0})}),
           CreateTensor<Variant>(
               TensorShape({}),
               {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
           CreateTensor<int64_t>(TensorShape({1, 3}), {1, 2, 3})},
          /*node_name=*/kNodeName};
}

std::vector<GetNextTestCase<TensorDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/PlainTensorDatasetParams(),
           /*expected_outputs=*/PlainTensors()},
          {/*dataset_params=*/NestedTensorDatasetParams(),
           /*expected_outputs=*/
           {CreateTensor<Variant>(TensorShape({}),
                                  {CreateTensor<double>(TensorShape({2, 2}),
                                                        {1.0, 2.0, 3.0, 4.0})}),
            CreateTensor<Variant>(
                TensorShape({}),
                {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
            CreateTensor<int64_t>(TensorShape({1, 3}), {1, 2, 3})}}};
}

class ParameterizedGetNextTest : public TensorDatasetOpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<TensorDatasetParams>> {};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  TF_EXPECT_OK(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence));
  ASSERT_FALSE(end_of_sequence);
  EXPECT_EQ(out_tensors.size(), test_case.expected_outputs.size());
  for (int i = 0; i < out_tensors.size(); ++i) {
    if (out_tensors[i].dtype() == DT_VARIANT) {
      // Currently `ExpectEqual()` does not support the variant tensor
      // yet, so we manually cast the variant to numeric/string tensor.
      const Tensor* output = out_tensors[i].scalar<Variant>()().get<Tensor>();
      const Tensor* expected_output =
          test_case.expected_outputs[i].scalar<Variant>()().get<Tensor>();
      TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
    } else {
      TF_EXPECT_OK(ExpectEqual(out_tensors[i], test_case.expected_outputs[i]));
    }
  }
  TF_EXPECT_OK(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence));
  EXPECT_TRUE(end_of_sequence);
  EXPECT_TRUE(out_tensors.empty());
}

INSTANTIATE_TEST_CASE_P(TensorDatasetOpTest, ParameterizedGetNextTest,
                        ::testing::ValuesIn(GetNextTestCases()));

TEST_F(TensorDatasetOpTest, DatasetTypeString) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(TensorDatasetOp::kDatasetType)));
}

TEST_F(TensorDatasetOpTest, DatasetNodeName) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(TensorDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(TensorDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(dataset_params.output_shapes()));
}

TEST_F(TensorDatasetOpTest, Cardinality) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(1));
}

TEST_F(TensorDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(TensorDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(dataset_params.output_shapes()));
}

TEST_F(TensorDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = PlainTensorDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      "FromTensor", dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<TensorDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/PlainTensorDatasetParams(),
           /*breakpoints=*/{0, 1, 2},
           /*expected_outputs=*/
           PlainTensors()},
          {/*dataset_params=*/NestedTensorDatasetParams(),
           /*breakpoints=*/{0, 1, 2},
           /*expected_outputs=*/
           {CreateTensor<Variant>(TensorShape({}),
                                  {CreateTensor<double>(TensorShape({2, 2}),
                                                        {1.0, 2.0, 3.0, 4.0})}),
            CreateTensor<Variant>(
                TensorShape({}),
                {CreateTensor<tstring>(TensorShape({1, 2}), {"a", "b"})}),
            CreateTensor<int64_t>(TensorShape({1, 3}), {1, 2, 3})}}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public TensorDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<TensorDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, SaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  const std::vector<int>& breakpoints = test_case.breakpoints;
  int cardinality = 1;
  for (int breakpoint : breakpoints) {
    VariantTensorDataWriter writer;
    TF_EXPECT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 test_case.dataset_params.iterator_prefix(),
                                 *dataset_, &iterator_));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }

    if (breakpoint >= cardinality) {
      EXPECT_TRUE(end_of_sequence);
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }

  EXPECT_EQ(out_tensors.size(), test_case.expected_outputs.size());
  for (int i = 0; i < out_tensors.size(); ++i) {
    if (out_tensors[i].dtype() == DT_VARIANT) {
      // Currently `ExpectEqual()` does not support the variant tensor
      // yet, so we manually cast the variant to numeric/string tensor.
      const Tensor* output = out_tensors[i].scalar<Variant>()().get<Tensor>();
      const Tensor* expected_output =
          test_case.expected_outputs[i].scalar<Variant>()().get<Tensor>();
      TF_EXPECT_OK(ExpectEqual(*output, *expected_output));
    } else {
      TF_EXPECT_OK(ExpectEqual(out_tensors[i], test_case.expected_outputs[i]));
    }
  }
}

INSTANTIATE_TEST_CASE_P(TensorDatasetOpTest,
                        ParameterizedIteratorSaveAndRestoreTest,
                        ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

TEST_F(TensorDatasetOpTest, Splitting) {
  auto params = PlainTensorDatasetParams();
  TF_ASSERT_OK(InitializeRuntime(params));
  TF_EXPECT_OK(CheckSplitProviderFullIteration(
      params, /*expected_outputs=*/PlainTensors()));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/3, /*shard_index=*/2,
      /*expected_outputs=*/CreateTensors<int64_t>(TensorShape({}), {})));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/3, /*shard_index=*/0,
      /*expected_outputs=*/PlainTensors()));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
