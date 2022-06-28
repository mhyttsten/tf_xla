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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/parallel_map_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/name_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "parallel_map_dataset";
constexpr int kOpVersion = 2;

class ParallelMapDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ParallelMapDatasetParams(
      T input_dataset_params, std::vector<Tensor> other_arguments,
      int num_parallel_calls, FunctionDefHelper::AttrValueWrapper func,
      std::vector<FunctionDef> func_lib, DataTypeVector type_arguments,
      const DataTypeVector& output_dtypes,
      const std::vector<PartialTensorShape>& output_shapes,
      bool use_inter_op_parallelism, const std::string& deterministic,
      bool preserve_cardinality, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        num_parallel_calls_(num_parallel_calls),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)),
        use_inter_op_parallelism_(use_inter_op_parallelism),
        deterministic_(deterministic),
        preserve_cardinality_(preserve_cardinality) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("deterministic: \"" + deterministic + "\"");
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "ParallelMapDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    op_version_ = kOpVersion;
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    iterator_prefix_ = name_utils::IteratorPrefix(
        input_dataset_params.dataset_type(),
        input_dataset_params.iterator_prefix(), params);
  }

  std::vector<Tensor> GetInputTensors() const override {
    auto input_tensors = other_arguments_;
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {num_parallel_calls_}));
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "GetInputNames");

    input_names->emplace_back(ParallelMapDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(ParallelMapDatasetOp::kOtherArguments, "_", i));
    }
    input_names->emplace_back(ParallelMapDatasetOp::kNumParallelCalls);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "GetAttributes");

    *attr_vector = {{"f", func_},
                    {"Targuments", type_arguments_},
                    {"output_shapes", output_shapes_},
                    {"output_types", output_dtypes_},
                    {"use_inter_op_parallelism", use_inter_op_parallelism_},
                    {"deterministic", deterministic_},
                    {"preserve_cardinality", preserve_cardinality_},
                    {"metadata", ""}};
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_3(mht_3_v, 263, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "dataset_type");

    return ParallelMapDatasetOp::kDatasetType;
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

 private:
  std::vector<Tensor> other_arguments_;
  int num_parallel_calls_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
  bool use_inter_op_parallelism_;
  std::string deterministic_;
  bool preserve_cardinality_;
};

class ParallelMapDatasetOpTest : public DatasetOpsTestBase {};

FunctionDefHelper::AttrValueWrapper MapFunc(const string& func_name,
                                            const DataType& dtype) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_4(mht_4_v, 287, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "MapFunc");

  return FunctionDefHelper::FunctionRef(func_name, {{"T", dtype}});
}

// test case 1: num_parallel_calls = 1, use_inter_op_parallelism = false,
// deterministic = true, preserve_cardinality = false, MapFunc = XTimesTwo
ParallelMapDatasetParams ParallelMapDatasetParams1() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_5(mht_5_v, 296, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "ParallelMapDatasetParams1");

  return ParallelMapDatasetParams(
      RangeDatasetParams(0, 10, 3),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/1,
      /*func=*/MapFunc("XTimesTwo", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*preserve_cardinality=*/false,
      /*node_name=*/kNodeName);
}

// test case 2: num_parallel_calls = 2, use_inter_op_parallelism = true,
// deterministic = false, preserve_cardinality = true, MapFunc = XTimesTwo
ParallelMapDatasetParams ParallelMapDatasetParams2() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_6(mht_6_v, 317, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "ParallelMapDatasetParams2");

  return ParallelMapDatasetParams(
      RangeDatasetParams(0, 10, 3),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/2,
      /*func=*/MapFunc("XTimesTwo", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*preserve_cardinality=*/true,
      /*node_name=*/kNodeName);
}

// test case 3: num_parallel_calls = 3, use_inter_op_parallelism = true,
// deterministic = true, preserve_cardinality = false, MapFunc = XTimesFour
ParallelMapDatasetParams ParallelMapDatasetParams3() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_7(mht_7_v, 338, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "ParallelMapDatasetParams3");

  return ParallelMapDatasetParams(
      RangeDatasetParams(0, 10, 3),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/3,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*preserve_cardinality=*/false,
      /*node_name=*/kNodeName);
}

// test case 4: num_parallel_calls = 4, use_inter_op_parallelism = false,
// deterministic = true, preserve_cardinality = false, MapFunc = XTimesTwo
ParallelMapDatasetParams ParallelMapDatasetParams4() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_8(mht_8_v, 359, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "ParallelMapDatasetParams4");

  return ParallelMapDatasetParams(
      RangeDatasetParams(0, 10, 3),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/4,
      /*func=*/MapFunc("XTimesTwo", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*preserve_cardinality=*/false,
      /*node_name=*/kNodeName);
}

// test case 5: num_parallel_calls = kAutotune, use_inter_op_parallelism = true,
// deterministic = false, preserve_cardinality = true, MapFunc = XTimesFour
ParallelMapDatasetParams ParallelMapDatasetParams5() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_9(mht_9_v, 380, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "ParallelMapDatasetParams5");

  return ParallelMapDatasetParams(
      RangeDatasetParams(0, 10, 3),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/model::kAutotune,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*preserve_cardinality=*/true,
      /*node_name=*/kNodeName);
}

// test case 6: num_parallel_calls = 4, use_inter_op_parallelism = true,
// deterministic = true, preserve_cardinality = false, MapFunc = XTimesFour
ParallelMapDatasetParams ParallelMapDatasetParams6() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_10(mht_10_v, 401, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "ParallelMapDatasetParams6");

  return ParallelMapDatasetParams(
      RangeDatasetParams(0, 10, 3),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/4,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*preserve_cardinality=*/false,
      /*node_name=*/kNodeName);
}

// TODO(feihugis): make this test case work.
// test case 7: num_parallel_calls = 2, use_inter_op_parallelism = false,
// deterministic = true, preserve_cardinality = false, MapFunc = XTimesFour
ParallelMapDatasetParams ParallelMapDatasetParams7() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_11(mht_11_v, 423, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "ParallelMapDatasetParams7");

  return ParallelMapDatasetParams(
      RangeDatasetParams(0, 10, 3),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/2,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*preserve_cardinality=*/false,
      /*node_name=*/kNodeName);
}

// TODO(feihugis): make this test case work.
// test case 8: num_parallel_calls = kAutotune, use_inter_op_parallelism =
// false, deterministic = false, preserve_cardinality = true, MapFunc =
// XTimesFour
ParallelMapDatasetParams ParallelMapDatasetParams8() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_12(mht_12_v, 446, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "ParallelMapDatasetParams8");

  return ParallelMapDatasetParams(
      RangeDatasetParams(0, 10, 3),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/model::kAutotune,
      /*func=*/MapFunc("XTimesFour", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo(), test::function::XTimesFour()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/false,
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*preserve_cardinality=*/true,
      /*node_name=*/kNodeName);
}

ParallelMapDatasetParams ParallelMapDatasetParamsWithInvalidNumParallelCalls() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_map_dataset_op_testDTcc mht_13(mht_13_v, 465, "", "./tensorflow/core/kernels/data/parallel_map_dataset_op_test.cc", "ParallelMapDatasetParamsWithInvalidNumParallelCalls");

  return ParallelMapDatasetParams(
      RangeDatasetParams(0, 10, 3),
      /*other_arguments=*/{},
      /*num_parallel_calls=*/-4,
      /*func=*/MapFunc("XTimesTwo", DT_INT64),
      /*func_lib*/ {test::function::XTimesTwo()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*use_inter_op_parallelism=*/true,
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*preserve_cardinality=*/true,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<ParallelMapDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/ParallelMapDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {6}, {12}, {18}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelMapDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {6}, {12}, {18}}),
           /*compare_order=*/false},
          {/*dataset_params=*/ParallelMapDatasetParams3(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {12}, {24}, {36}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelMapDatasetParams4(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {6}, {12}, {18}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelMapDatasetParams5(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {12}, {24}, {36}}),
           /*compare_order=*/false},
          {/*dataset_params=*/
           ParallelMapDatasetParams6(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {12}, {24}, {36}}),
           /*compare_order=*/true}};
}

ITERATOR_GET_NEXT_TEST_P(ParallelMapDatasetOpTest, ParallelMapDatasetParams,
                         GetNextTestCases())

TEST_F(ParallelMapDatasetOpTest, DatasetNodeName) {
  auto dataset_params = ParallelMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ParallelMapDatasetOpTest, DatasetTypeString) {
  auto dataset_params = ParallelMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::OpNameParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(ParallelMapDatasetOp::kDatasetType, params)));
}

TEST_F(ParallelMapDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = ParallelMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(ParallelMapDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = ParallelMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

std::vector<CardinalityTestCase<ParallelMapDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/ParallelMapDatasetParams1(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelMapDatasetParams2(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/ParallelMapDatasetParams3(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelMapDatasetParams4(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelMapDatasetParams5(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/ParallelMapDatasetParams6(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(ParallelMapDatasetOpTest, ParallelMapDatasetParams,
                           CardinalityTestCases())

TEST_F(ParallelMapDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = ParallelMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(ParallelMapDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = ParallelMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(ParallelMapDatasetOpTest, IteratorPrefix) {
  auto dataset_params = ParallelMapDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::IteratorPrefixParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckIteratorPrefix(
      name_utils::IteratorPrefix(ParallelMapDatasetOp::kDatasetType,
                                 dataset_params.iterator_prefix(), params)));
}

std::vector<IteratorSaveAndRestoreTestCase<ParallelMapDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/ParallelMapDatasetParams1(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {6}, {12}, {18}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelMapDatasetParams2(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {6}, {12}, {18}}),
           /*compare_order=*/false},
          {/*dataset_params=*/ParallelMapDatasetParams3(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {12}, {24}, {36}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelMapDatasetParams4(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {6}, {12}, {18}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelMapDatasetParams5(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {12}, {24}, {36}}),
           /*compare_order=*/false},
          {/*dataset_params=*/
           ParallelMapDatasetParams6(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{}, {{0}, {12}, {24}, {36}}),
           /*compare_order=*/true}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ParallelMapDatasetOpTest,
                                 ParallelMapDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(ParallelMapDatasetOpTest, InvalidNumParallelCalls) {
  auto dataset_params = ParallelMapDatasetParamsWithInvalidNumParallelCalls();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
