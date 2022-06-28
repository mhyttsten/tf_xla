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
class MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_op_testDTcc() {
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

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

template <typename VALUE_TYPE>
struct ShapeAndValues {
  TensorShape shape;
  std::vector<VALUE_TYPE> values;
};

template <typename VALUE_TYPE>
ShapeAndValues<VALUE_TYPE> createVector(const std::vector<VALUE_TYPE>& values) {
  TensorShape shape({static_cast<int64_t>(values.size())});
  return {shape, values};
}

template <typename VALUE_TYPE>
ShapeAndValues<VALUE_TYPE> createScalar(const VALUE_TYPE& values) {
  TensorShape shape({});
  return {shape, {values}};
}

class RaggedTensorToTensorOpTest : public ::tensorflow::OpsTestBase {
 protected:
  // Builds the tensorflow test graph for RaggedTensorToTensor.
  template <typename VALUE_TYPE, typename INDEX_TYPE>
  void BuildRaggedTensorToTensorGraph(
      const TensorShape& shape, const std::vector<string>& row_partition_types,
      const ShapeAndValues<VALUE_TYPE>& values,
      const ShapeAndValues<VALUE_TYPE>& default_value,
      const std::vector<ShapeAndValues<INDEX_TYPE>>& row_partition_tensors) {
    const auto& value_dtype = DataTypeToEnum<VALUE_TYPE>::v();
    const auto& index_dtype = DataTypeToEnum<INDEX_TYPE>::v();
    int num_row_partition_tensors = row_partition_tensors.size();
    TF_ASSERT_OK(
        NodeDefBuilder("tested_op", "RaggedTensorToTensor")
            .Attr("T", value_dtype)
            .Attr("Tindex", index_dtype)
            .Attr("num_row_partition_tensors", num_row_partition_tensors)
            .Attr("row_partition_types", row_partition_types)
            .Input(FakeInput(index_dtype))
            .Input(FakeInput(value_dtype))  // values
            .Input(FakeInput(value_dtype))  // default_value
            .Input(FakeInput(num_row_partition_tensors,
                             index_dtype))  // row_partition_tensors
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    {
      std::vector<INDEX_TYPE> shape_as_vector;
      for (const auto& dim : shape.dim_sizes()) {
        shape_as_vector.push_back(dim);
      }
      ShapeAndValues<INDEX_TYPE> shape_as_tensor =
          createVector(shape_as_vector);
      AddInputFromArray<INDEX_TYPE>(shape_as_tensor.shape,
                                    shape_as_tensor.values);
    }
    AddInputFromArray<VALUE_TYPE>(values.shape, values.values);
    AddInputFromArray<VALUE_TYPE>(default_value.shape, default_value.values);

    for (const auto& row_partition_tensor : row_partition_tensors) {
      AddInputFromArray<INDEX_TYPE>(row_partition_tensor.shape,
                                    row_partition_tensor.values);
    }
  }
};

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor) {
  // indices = [2, 1, 0, 3]
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  // params.shape = [4, None]
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({4, 4}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {createScalar<int32>(4), createVector<int32>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
      // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({.1, .2, .3, 1.5, 1.5, 1.5, 1.5, 1.5, .4, .5, .6,
                             .7, .8, .9, 1.5, 1.5},
                            TensorShape({4, 4})),
      0.01);
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensorRowSplits) {
  // indices = [2, 1, 0, 3]
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({4, 4}),  // shape
      {"ROW_SPLITS"},       // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),               // default_value
      {createVector<int32>({0, 3, 3, 7, 9})}  // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({.1, .2, .3, 1.5, 1.5, 1.5, 1.5, 1.5, .4, .5, .6,
                             .7, .8, .9, 1.5, 1.5},
                            TensorShape({4, 4})),
      0.01);
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_3DParams) {
  // params = [
  //           [[]],
  //           [[.1, .2], [.3]],
  //           [],
  //           [[.4, .5], [.6, .7, .8]],
  //           [[.9]]
  //          ]
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({5, 2, 3}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {
          createScalar<int32>(5),
          createVector<int32>({0, 1, 1, 3, 3, 4}),
          createVector<int32>({1, 1, 2, 3, 3, 4, 4, 4, 5}),
      }  // row_partition_tensors
  );
  TF_ASSERT_OK(RunOpKernel());

  // Expected = [
  //              [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
  //              [[.1, .2, 1.5], [.3, 1.5, 1.5]],
  //              [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
  //              [[.4, .5, 1.5], [.6, .7, .8]],
  //              [[.9, 1.5, 1.5], [1.5, 1.5, 1.5]]
  //            ]
  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .1,  .2,  1.5, .3,
                             1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .4,  .5,
                             1.5, .6,  .7,  .8,  .9,  1.5, 1.5, 1.5, 1.5, 1.5},
                            TensorShape({5, 2, 3})),
      0.1);
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_3DParamsRowSplits) {
  // params = [
  //           [[]],
  //           [[.1, .2], [.3]],
  //           [],
  //           [[.4, .5], [.6, .7, .8]],
  //           [[.9]]
  //          ]
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({5, 2, 3}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {
          createVector<int32>({0, 1, 3, 3, 5, 6}),
          createVector<int32>({0, 0, 2, 3, 5, 8, 9}),
      }  // row_partition_tensors
  );
  TF_ASSERT_OK(RunOpKernel());

  // Expected = [
  //              [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
  //              [[.1, .2, 1.5], [.3, 1.5, 1.5]],
  //              [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
  //              [[.4, .5, 1.5], [.6, .7, .8]],
  //              [[.9, 1.5, 1.5], [1.5, 1.5, 1.5]]
  //            ]
  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .1,  .2,  1.5, .3,
                             1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .4,  .5,
                             1.5, .6,  .7,  .8,  .9,  1.5, 1.5, 1.5, 1.5, 1.5},
                            TensorShape({5, 2, 3})),
      0.1);
}

// test_three_dimensional_ragged fails, want to try it at a lower level.
TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_3DParamsRowSplits2) {
  // params = [
  //           [[0, 1, 2], []],
  //           [],
  //           [[3]]
  //          ]
  BuildRaggedTensorToTensorGraph<int64_t, int64_t>(
      TensorShape({3, 2, 3}),               // shape
      {"ROW_SPLITS", "ROW_SPLITS"},         // row_partition_types
      createVector<int64_t>({0, 1, 2, 3}),  // values
      createScalar<int64_t>(5),             // default_value
      {
          createVector<int64_t>({0, 2, 2, 3}),
          createVector<int64_t>({0, 3, 3, 4}),
      }  // row_partition_tensors
  );
  TF_ASSERT_OK(RunOpKernel());

  // Expected = [
  //              [[0, 1, 2], [5, 5, 5]],
  //              [[5, 5, 5], [5, 5, 5]],
  //              [[3, 5, 5], [5, 5, 5]]
  //            ]
  test::ExpectTensorEqual<int64_t>(
      *GetOutput(0), test::AsTensor<int64_t>(
                         {0, 1, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5},
                         TensorShape({3, 2, 3})));
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_4DParams) {
  // Input:    [[],
  //            [
  //             [[1, 2], [3, 4], [5, 6]],
  //             [[7, 8]]
  //            ],
  //            [[]],
  //            []
  // ]
  // params.shape = [3, 2, 3, 2]
  BuildRaggedTensorToTensorGraph<int32, int32>(
      TensorShape({4, 2, 3, 2}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},                               // row_partition_types
      createVector<int32>({1, 2, 3, 4, 5, 6, 7, 8}),  // values
      createScalar<int32>(15),                        // default_value
      {createScalar<int32>(5), createVector<int32>({0, 1, 1}),
       createVector<int32>({1, 1, 1, 2}),
       createVector<int32>({0, 0, 1, 1, 2, 2, 3, 3})}  // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());
  // params = [
  //           [
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ],
  //           [
  //             [[1, 2], [3, 4], [5, 6]],
  //             [[7, 8], [15, 15], [15,15]],
  //           ],
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ],
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ]
  // params.shape = [3, 2, 3, 2]
  test::ExpectTensorEqual<int32>(
      *GetOutput(0),
      test::AsTensor<int32>(
          {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 1,  2,  3,  4,
           5,  6,  7,  8,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
           15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15},
          TensorShape({4, 2, 3, 2})));
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_4DParamsRowSplit) {
  // Input:    [[],
  //            [
  //             [[1, 2], [3, 4], [5, 6]],
  //             [[7, 8]]
  //            ],
  //            [[]],
  //            []
  // ]
  // params.shape = [3, 2, 3, 2]
  BuildRaggedTensorToTensorGraph<int32, int32>(
      TensorShape({4, 2, 3, 2}),  // shape
      {"ROW_SPLITS", "ROW_SPLITS", "ROW_SPLITS"},
      // row_partition_types
      createVector<int32>({1, 2, 3, 4, 5, 6, 7, 8}),  // values
      createScalar<int32>(15),                        // default_value
      {createVector<int32>({0, 1, 3}), createVector<int32>({0, 0, 3, 4}),
       createVector<int32>({0, 2, 4, 6, 8})}  // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());
  // params = [
  //           [
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ],
  //           [
  //             [[1, 2], [3, 4], [5, 6]],
  //             [[7, 8], [15, 15], [15,15]],
  //           ],
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ],
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ]
  // params.shape = [3, 2, 3, 2]
  test::ExpectTensorEqual<int32>(
      *GetOutput(0),
      test::AsTensor<int32>(
          {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 1,  2,  3,  4,
           5,  6,  7,  8,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
           15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15},
          TensorShape({4, 2, 3, 2})));
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensorContractExpanded) {
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({3, 5}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {createScalar<int32>(4), createVector<int32>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
      // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({.1, .2, .3, 1.5, 1.5,     //
                             1.5, 1.5, 1.5, 1.5, 1.5,  //
                             .4, .5, .6, .7, 1.5},     //
                            TensorShape({3, 5})),
      0.01);
}

// Adds a dense dimension.
TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensorContractExpandedDense) {
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({3, 5, 2}),              // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      ShapeAndValues<float>{TensorShape({9, 2}),
                            {.1, 1.1, .2, 1.2, .3, 1.3, .4, 1.4, .5, 1.5, .6,
                             1.6, .7, 1.7, .8, 1.8, .9, 1.9}},  // values
      createScalar<float>(1.5),                                 // default_value
      {createScalar<int32>(4), createVector<int32>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
      // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>(
          {.1,  1.1, .2,  1.2, .3,  1.3, 1.5, 1.5, 1.5, 1.5,   //
           1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,   //
           .4,  1.4, .5,  1.5, .6,  1.6, .7,  1.7, 1.5, 1.5},  //
          TensorShape({3, 5, 2})),
      0.01);
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensorConstrained) {
  // params = [[.1, .2, .3],
  //           [],
  //           [.4, .5, .6, .7],
  //           [.8, .9]]
  // constrained to (3, 3)
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({3, 3}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {createScalar<int32>(4), createVector<int32>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
      // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorNear<float>(*GetOutput(0),
                                test::AsTensor<float>(
                                    {
                                        //
                                        .1, .2, .3,     //
                                        1.5, 1.5, 1.5,  //
                                        .4, .5, .6      //
                                    },
                                    TensorShape({3, 3})),
                                0.01);
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_3DParamsConstrained) {
  // params = [
  //           [[]],
  //           [[.1, .2], [.3]],
  //           [],
  //           [[.4, .5], [.6, .7, .8]],
  //           [[.9]]
  //          ]
  // params.shape = [5, None, None]
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({4, 1, 2}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {
          createScalar<int32>(5),
          createVector<int32>({0, 1, 1, 3, 3, 4}),
          createVector<int32>({1, 1, 2, 3, 3, 4, 4, 4, 5}),
      }  // row_partition_tensors
  );
  TF_ASSERT_OK(RunOpKernel());

  // Expected = [
  //              [[1.5, 1.5]],
  //              [[.1, .2]],
  //              [[1.5, 1.5]],
  //              [[.4, .5]],
  //            ]
  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({1.5, 1.5, .1, .2, 1.5, 1.5, .4, .5},
                            TensorShape({4, 1, 2})),
      0.01);
}

// Seg fault but removing this does not make the problem go away.
// This tests is labeled as flaky. Removing it to find out.
TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_4DParamsConstrained) {
  // Input:    [[],
  //            [
  //             [[1, 2], [3, 4], [5, 6]],
  //             [[7, 8]]
  //            ],
  //            [[]],
  //            []
  // ]
  // params.shape = [3, 2, 3, 2]
  BuildRaggedTensorToTensorGraph<int32, int32>(
      TensorShape({2, 2, 2, 2}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},                               // row_partition_types
      createVector<int32>({1, 2, 3, 4, 5, 6, 7, 8}),  // values
      createScalar<int32>(15),                        // default_value
      {createScalar<int32>(5), createVector<int32>({0, 1, 1}),
       createVector<int32>({1, 1, 1, 2}),
       createVector<int32>({0, 0, 1, 1, 2, 2, 3, 3})}  // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());
  // params = [
  //           [
  //             [[15,15],[15,15]],
  //             [[15,15],[15,15]],
  //           ],
  //           [
  //             [[1, 2], [3, 4]],
  //             [[7, 8], [15, 15]],
  //           ],
  //          ]
  // params.shape = [3, 2, 3, 2]
  test::ExpectTensorEqual<int32>(*GetOutput(0), test::AsTensor<int32>(
                                                    {
                                                        15, 15, 15, 15,  //
                                                        15, 15, 15, 15,  //
                                                        1, 2, 3, 4,      //
                                                        7, 8, 15, 15,    //
                                                    },
                                                    TensorShape({2, 2, 2, 2})));
}

TEST_F(RaggedTensorToTensorOpTest, ShapeWrongDimensions) {
  BuildRaggedTensorToTensorGraph<int32, int32>(
      TensorShape({10, 7, 10, 20}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},                   // row_partition_types
      createVector<int32>({1, 2, 3, 4}),  // values
      createScalar<int32>(15),            // default_value
      {createScalar<int32>(5), createVector<int32>({0, 1, 1}),
       createVector<int32>({1, 1, 1, 2})}  // row_partition_tensors
  );
  // Fails with an invalid argument.
  EXPECT_EQ(RunOpKernel().code(), errors::Code::INVALID_ARGUMENT);
}

class RaggedTensorToTensorOpUnknownShapeTest
    : public ::tensorflow::OpsTestBase {
 protected:
  std::unique_ptr<ShapeInferenceTestOp> op_;
  void SetAttributes(const gtl::ArraySlice<string> row_partition_types,
                     int num_row_partition_tensors) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_tensor_op_testDTcc mht_0(mht_0_v, 681, "", "./tensorflow/core/kernels/ragged_tensor_to_tensor_op_test.cc", "SetAttributes");

    op_ = absl::make_unique<ShapeInferenceTestOp>("RaggedTensorToTensor");
    SetAttrValue(row_partition_types,
                 &((*op_->node_def.mutable_attr())["row_partition_types"]));
    (*op_->node_def.mutable_attr())["num_row_partition_tensors"].set_i(
        num_row_partition_tensors);
  }
};

TEST_F(RaggedTensorToTensorOpUnknownShapeTest, ValueRowIDs) {
  SetAttributes(gtl::ArraySlice<string>{"FIRST_DIM_SIZE", "VALUE_ROWIDS"}, 2);

  INFER_OK(*op_, "?;?;?;?;?", "?");
  INFER_OK(*op_, "?;[6];[];[];[6]", "[?,?]");
  INFER_OK(*op_, "?;[6];?;[];[6]", "[?,?]");
  INFER_OK(*op_, "?;?;[];[];[6]", "?");
  INFER_OK(*op_, "?;[6];?;[];[6]", "[?,?]");
  INFER_OK(*op_, "?;[6,2];?;[];[6]", "[?,?,2]");
  INFER_OK(*op_, "?;[6,2];[2];[];[6]", "[?,?,2]");
  INFER_OK(*op_, "?;[6,2,7];[2,7];[];[6]", "[?,?,2,7]");
  INFER_ERROR(
      "default_value.shape=[3] and rt_input.flat_values.shape=[6,2] "
      "are incompatible",
      *op_, "?;[6,2];[3];[];[6]");
  INFER_ERROR(
      "default_value.shape=[2,2] and rt_input.flat_values.shape="
      "[6,2,1,2] are incompatible",
      *op_, "?;[6,2,1,2];[2,2];[];[6]");
  INFER_ERROR("must be a vector", *op_, "?;[6];[];[];[3,6]");
  INFER_ERROR("must be a scalar", *op_, "?;[6];[];[7];[3]");
}

TEST_F(RaggedTensorToTensorOpUnknownShapeTest, RowSplits) {
  // RaggedTensorToTensor(param_splits+, param_values, indices) -> [splits+,
  // values]
  SetAttributes(gtl::ArraySlice<string>{"ROW_SPLITS"}, 1);

  // value, default_value, ROW_SPLITS
  INFER_OK(*op_, "?;?;?;?", "?");
  INFER_OK(*op_, "?;[3];[];[6]", "[?,?]");
  INFER_OK(*op_, "?;?;?;?", "?");
  INFER_OK(*op_, "?;[3,2];[2];[6]", "[?,?,2]");
  INFER_OK(*op_, "?;[3,2,7];[2,7];[6]", "[?,?,2,7]");
  INFER_OK(*op_, "?;[3,2,7];[2,7];[6]", "[?,?,2,7]");
}

}  // namespace
}  // namespace tensorflow
