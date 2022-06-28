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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/map_defun_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "map_defun";
constexpr char kOpName[] = "MapDefun";

class MapDefunOpParams : public DatasetParams {
 public:
  MapDefunOpParams(std::vector<Tensor> arguments,
                   std::vector<Tensor> captured_inputs,
                   DataTypeVector type_arguments, DataTypeVector type_captured,
                   DataTypeVector output_dtypes,
                   std::vector<PartialTensorShape> output_shapes,
                   FunctionDefHelper::AttrValueWrapper func,
                   std::vector<FunctionDef> func_lib,
                   int max_intra_op_parallelism, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        arguments_(std::move(arguments)),
        captured_inputs_(std::move(captured_inputs)),
        type_arguments_(std::move(type_arguments)),
        type_captured_(std::move(type_captured)),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        max_intra_op_parallelism_(max_intra_op_parallelism) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "MapDefunOpParams");
}

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors = arguments_;
    input_tensors.insert(input_tensors.end(), captured_inputs_.begin(),
                         captured_inputs_.end());
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "GetInputNames");

    input_names->clear();

    input_names->reserve(arguments_.size() + captured_inputs_.size());
    for (int i = 0; i < arguments_.size(); ++i) {
      input_names->emplace_back(
          strings::StrCat(MapDefunOp::kArguments, "_", i));
    }
    for (int i = 0; i < captured_inputs_.size(); ++i) {
      input_names->emplace_back(
          strings::StrCat(MapDefunOp::kCapturedInputs, "_", i));
    }
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "GetAttributes");

    *attr_vector = {
        {MapDefunOp::kTarguments, type_arguments_},
        {MapDefunOp::kTcaptured, type_captured_},
        {MapDefunOp::kOutputShapes, output_shapes_},
        {MapDefunOp::kOutputTypes, output_dtypes_},
        {MapDefunOp::kFunc, func_},
        {MapDefunOp::kMaxIntraOpParallelism, max_intra_op_parallelism_}};
    return Status::OK();
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "dataset_type");
 return "MapDef"; }

 private:
  std::vector<Tensor> arguments_;
  std::vector<Tensor> captured_inputs_;
  DataTypeVector type_arguments_;
  DataTypeVector type_captured_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  int max_intra_op_parallelism_;
};

class MapDefunOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `MapDefun` op kernel
  Status CreateMapDefunOpKernel(const MapDefunOpParams& params,
                                std::unique_ptr<OpKernel>* map_defun_kernel) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_4(mht_4_v, 276, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "CreateMapDefunOpKernel");

    std::vector<string> input_namess;
    TF_RETURN_IF_ERROR(params.GetInputNames(&input_namess));
    AttributeVector attributes;
    TF_RETURN_IF_ERROR(params.GetAttributes(&attributes));

    NodeDef node_def =
        test::function::NDef(kNodeName, kOpName, input_namess, attributes);
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, map_defun_kernel));
    return Status::OK();
  }

  // Creates a new `MapDefun` op kernel context.
  Status CreateMapDefunContext(OpKernel* const op_kernel,
                               gtl::InlinedVector<TensorValue, 4>* const inputs,
                               std::unique_ptr<OpKernelContext>* context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_5(mht_5_v, 294, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "CreateMapDefunContext");

    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  MapDefunOpParams map_defun_op_params;
  std::vector<Tensor> expected_outputs;
};

// Test case 1: one input for the map function with no captured inputs.
TestCase TestCase1() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_6(mht_6_v, 310, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "TestCase1");

  return {/*map_defun_op_params=*/
          MapDefunOpParams(
              /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                                   {0, 1, 2, 3, 4, 5})},
              /*captured_inputs=*/{},
              /*type_arguments=*/{DT_INT64},
              /*type_captured=*/{},
              /*output_dtypes=*/{DT_INT64},
              /*output_shapes=*/{PartialTensorShape({2})},
              /*func=*/
              {FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_INT64}})},
              /*func_lib=*/{test::function::XTimesTwo()},
              /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
          /*expected_outputs=*/
          {CreateTensor<int64_t>(TensorShape({3, 2}), {0, 2, 4, 6, 8, 10})}};
}

// Test case 2: two inputs for the map function with no captured inputs.
TestCase TestCase2() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_7(mht_7_v, 332, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "TestCase2");

  return {
      /*map_defun_op_params=*/
      MapDefunOpParams(
          /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                               {0, 1, 2, 3, 4, 5}),
                         CreateTensor<int64_t>(TensorShape({3, 2}),
                                               {0, 10, 20, 30, 40, 50})},
          /*captured_inputs=*/{},
          /*type_arguments=*/{DT_INT64, DT_INT64},
          /*type_captured=*/{},
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({2})},
          /*func=*/
          {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
          /*func_lib=*/{test::function::XAddY()},
          /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
      /*expected_outputs=*/
      {CreateTensor<int64_t>(TensorShape({3, 2}), {0, 11, 22, 33, 44, 55})}};
}

// Test case 3: two inputs for the map function with one captured input.
TestCase TestCase3() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_8(mht_8_v, 357, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "TestCase3");

  return {/*map_defun_op_params=*/
          MapDefunOpParams(
              /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                                   {0, 1, 2, 3, 4, 5})},
              /*captured_inputs=*/
              {CreateTensor<int64_t>(TensorShape({2}), {10, 100})},
              /*type_arguments=*/{DT_INT64},
              /*type_captured=*/{DT_INT64},
              /*output_dtypes=*/{DT_INT64},
              /*output_shapes=*/{PartialTensorShape({2})},
              /*func=*/
              {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
              /*func_lib=*/{test::function::XAddY()},
              /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
          /*expected_outputs=*/
          {CreateTensor<int64_t>(TensorShape({3, 2}),
                                 {10, 101, 12, 103, 14, 105})}};
}

TestCase InvalidOutputTypes() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_9(mht_9_v, 380, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "InvalidOutputTypes");

  return {/*map_defun_op_params=*/
          MapDefunOpParams(
              /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                                   {0, 1, 2, 3, 4, 5})},
              /*captured_inputs=*/
              {CreateTensor<int64_t>(TensorShape({2}), {10, 100})},
              /*type_arguments=*/{DT_INT64},
              /*type_captured=*/{DT_INT64},
              /*output_dtypes=*/{DT_FLOAT},
              /*output_shapes=*/{PartialTensorShape({2})},
              /*func=*/
              {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
              /*func_lib=*/{test::function::XAddY()},
              /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
          /*expected_outputs=*/{}};
}

TestCase InvalidOutputShapes() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_10(mht_10_v, 401, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "InvalidOutputShapes");

  return {/*map_defun_op_params=*/
          MapDefunOpParams(
              /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                                   {0, 1, 2, 3, 4, 5})},
              /*captured_inputs=*/
              {CreateTensor<int64_t>(TensorShape({2}), {10, 100})},
              /*type_arguments=*/{DT_INT64},
              /*type_captured=*/{DT_INT64},
              /*output_dtypes=*/{DT_INT64},
              /*output_shapes=*/{PartialTensorShape({2, 2})},
              /*func=*/
              {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
              /*func_lib=*/{test::function::XAddY()},
              /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
          /*expected_outputs=*/{}};
}

TestCase InvalidInputs() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_op_testDTcc mht_11(mht_11_v, 422, "", "./tensorflow/core/kernels/data/map_defun_op_test.cc", "InvalidInputs");

  return {/*map_defun_op_params=*/
          MapDefunOpParams(
              /*arguments=*/{CreateTensor<int64_t>(TensorShape({3, 2}),
                                                   {0, 1, 2, 3, 4, 5}),
                             CreateTensor<int64_t>(TensorShape({2, 2}),
                                                   {0, 1, 2, 3})},
              /*captured_inputs=*/{},
              /*type_arguments=*/{DT_INT64, DT_INT64},
              /*type_captured=*/{},
              /*output_dtypes=*/{DT_INT64},
              /*output_shapes=*/{PartialTensorShape({2})},
              /*func=*/
              {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
              /*func_lib=*/{test::function::XAddY()},
              /*max_intra_op_parallelism=*/2, /*node_name=*/kNodeName),
          /*expected_outputs=*/{}};
}

class ParameterizedMapDefunOpTest
    : public MapDefunOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedMapDefunOpTest, NormalTests) {
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitializeRuntime(test_case.map_defun_op_params));
  auto input_tensors = test_case.map_defun_op_params.GetInputTensors();
  gtl::InlinedVector<TensorValue, 4> input_values;
  for (auto& input : input_tensors) {
    input_values.push_back(TensorValue(&input));
  }
  std::unique_ptr<OpKernel> map_defun_kernel;
  TF_ASSERT_OK(
      CreateMapDefunOpKernel(test_case.map_defun_op_params, &map_defun_kernel));
  std::unique_ptr<OpKernelContext> context;
  TF_ASSERT_OK(
      CreateMapDefunContext(map_defun_kernel.get(), &input_values, &context));
  TF_ASSERT_OK(RunOpKernel(map_defun_kernel.get(), context.get()));

  EXPECT_EQ(context->num_outputs(), test_case.expected_outputs.size());
  for (int i = 0; i < context->num_outputs(); ++i) {
    TF_EXPECT_OK(ExpectEqual(*context->mutable_output(i),
                             test_case.expected_outputs[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(MapDefunOpTest, ParameterizedMapDefunOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3()})));

TEST_F(MapDefunOpTest, InvalidArguments) {
  std::vector<TestCase> test_cases = {InvalidOutputTypes(),
                                      InvalidOutputShapes(), InvalidInputs()};
  for (auto& test_case : test_cases) {
    TF_ASSERT_OK(InitializeRuntime(test_case.map_defun_op_params));
    auto input_tensors = test_case.map_defun_op_params.GetInputTensors();
    gtl::InlinedVector<TensorValue, 4> input_values;
    for (auto& input : input_tensors) {
      input_values.push_back(TensorValue(&input));
    }
    std::unique_ptr<OpKernel> map_defun_kernel;
    TF_ASSERT_OK(CreateMapDefunOpKernel(test_case.map_defun_op_params,
                                        &map_defun_kernel));
    std::unique_ptr<OpKernelContext> context;
    TF_ASSERT_OK(
        CreateMapDefunContext(map_defun_kernel.get(), &input_values, &context));
    EXPECT_EQ(RunOpKernel(map_defun_kernel.get(), context.get()).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
