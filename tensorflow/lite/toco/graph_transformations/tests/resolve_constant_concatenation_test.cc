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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSresolve_constant_concatenation_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSresolve_constant_concatenation_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSresolve_constant_concatenation_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {
// A gmock matcher that check that elements of a float vector match to a given
// tolerance.
std::vector<testing::Matcher<float>> ArrayFloatNear(
    const std::vector<float>& values, float max_abs_error = 1e-5) {
  std::vector<testing::Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    matchers.emplace_back(testing::FloatNear(v, max_abs_error));
  }
  return matchers;
}
}  // namespace

// The following 3 tests make sure the concatenation operation on different axis
// values match TensorFlow results listed below:
//
// x0 = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
// x1 = [[[10, 11], [12, 13]], [[14, 15], [16, 17]]]
// x2 = [[[20, 21], [22, 23]], [[24, 25], [26, 27]]]
// x3 = [[[30, 31], [32, 33]], [[34, 35], [36, 37]]]
//
// ConcatAtAxis0 test:
// t0 = tf.concat([x0, x1, x2, x3], 0)
// [[[ 0  1]
//   [ 2  3]]
//
//  [[ 4  5]
//   [ 6  7]]
//
//  [[10 11]
//   [12 13]]
//
//  [[14 15]
//   [16 17]]
//
//  [[20 21]
//   [22 23]]
//
//  [[24 25]
//   [26 27]]
//
//  [[30 31]
//   [32 33]]
//
//  [[34 35]
//   [36 37]]]
//
// ConcatAtAxis1 test:
// t1 = tf.concat([x0, x1, x2, x3], 1)
// [[[ 0  1]
//   [ 2  3]
//   [10 11]
//   [12 13]
//   [20 21]
//   [22 23]
//   [30 31]
//   [32 33]]
//
//  [[ 4  5]
//   [ 6  7]
//   [14 15]
//   [16 17]
//   [24 25]
//   [26 27]
//   [34 35]
//   [36 37]]]
//
// ConcatAtAxis2 test:
// t2 = tf.concat([x0, x1, x2, x3], 2)
// [[[ 0  1 10 11 20 21 30 31]
//   [ 2  3 12 13 22 23 32 33]]
//
//  [[ 4  5 14 15 24 25 34 35]
//   [ 6  7 16 17 26 27 36 37]]]

class ResolveConstantConcatenationTest : public ::testing::Test {
 protected:
  ResolveConstantConcatenationTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSresolve_constant_concatenation_testDTcc mht_0(mht_0_v, 273, "", "./tensorflow/lite/toco/graph_transformations/tests/resolve_constant_concatenation_test.cc", "ResolveConstantConcatenationTest");
}

  // Prepare a hypothetical TOCO model with one Concatenation operator in it
  // together with 4 arrays as its inputs.
  // It receives the dimension of concatenation as input.
  void PrepareModel(Model* model, int axis) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSresolve_constant_concatenation_testDTcc mht_1(mht_1_v, 281, "", "./tensorflow/lite/toco/graph_transformations/tests/resolve_constant_concatenation_test.cc", "PrepareModel");

    const std::string output_name("concat_op_output");
    model->flags.add_output_arrays(output_name);
    std::vector<std::string> concat_input_names = {"array0", "array1", "array2",
                                                   "array3"};

    const int kDim = 3;
    const int kElementPerDim = 2;
    const int kBufSize = 8;
    const int kNumArrays = 4;
    static float in_buf[kNumArrays][kBufSize] = {
        {0., 1., 2., 3., 4., 5., 6., 7.},
        {10., 11., 12., 13., 14., 15., 16., 17.},
        {20., 21., 22., 23., 24., 25., 26., 27.},
        {30., 31., 32., 33., 34., 35., 36., 37.}};
    int cnt = 0;
    for (const std::string& concat_input_name : concat_input_names) {
      Array& in_array = model->GetOrCreateArray(concat_input_name);
      in_array.data_type = ArrayDataType::kFloat;

      // Initialize shape for the input array.
      Shape* in_array_shape = in_array.mutable_shape();
      std::vector<int>* in_array_shape_dim = in_array_shape->mutable_dims();
      for (int i = 0; i < kDim; i++) {
        in_array_shape_dim->push_back(kElementPerDim);
      }
      auto& in_array_buffer =
          in_array.GetMutableBuffer<toco::ArrayDataType::kFloat>();
      in_array_buffer.data.resize(kBufSize);
      float* buf_ptr =
          in_array.GetMutableBuffer<toco::ArrayDataType::kFloat>().data.data();
      std::copy(in_buf[cnt], in_buf[cnt] + kBufSize, buf_ptr);
      cnt++;
    }
    auto* concatenation_op = new ConcatenationOperator;
    concatenation_op->axis = axis;
    concatenation_op->inputs = concat_input_names;
    concatenation_op->outputs = {output_name};
    Array& out_array = model->GetOrCreateArray(concatenation_op->outputs[0]);
    out_array.data_type = ArrayDataType::kFloat;
    Shape* out_array_shape = out_array.mutable_shape();
    std::vector<int>* out_array_shape_dim = out_array_shape->mutable_dims();
    out_array_shape_dim->resize(kDim);
    for (int i = 0; i < kDim; i++) {
      if (i == axis) {
        (*out_array_shape_dim)[i] = kNumArrays * kElementPerDim;
      } else {
        (*out_array_shape_dim)[i] = kElementPerDim;
      }
    }
    model->operators.push_back(std::unique_ptr<Operator>(concatenation_op));
  }
};

TEST_F(ResolveConstantConcatenationTest, ConcatAtAxis0) {
  Model model;
  const int axis = 0;
  PrepareModel(&model, axis);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::ResolveConstantConcatenation);
  EXPECT_THAT(model.GetArrayMap().size(), 5);
  bool modified;
  ASSERT_TRUE((*graph_transformation_set.begin())
                  ->Run(&model, /*op_index=*/0, &modified)
                  .ok());
  EXPECT_THAT(model.GetArrayMap().size(), 1);

  const auto& concatenated_array = model.GetArray(model.flags.output_arrays(0));
  EXPECT_THAT(concatenated_array.GetBuffer<toco::ArrayDataType::kFloat>().data,
              ElementsAreArray(ArrayFloatNear(
                  {0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  10., 11., 12.,
                   13., 14., 15., 16., 17., 20., 21., 22., 23., 24., 25.,
                   26., 27., 30., 31., 32., 33., 34., 35., 36., 37.})));
}

TEST_F(ResolveConstantConcatenationTest, ConcatAtAxis1) {
  Model model;
  const int axis = 1;
  PrepareModel(&model, axis);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::ResolveConstantConcatenation);
  EXPECT_THAT(model.GetArrayMap().size(), 5);
  bool modified;
  ASSERT_TRUE((*graph_transformation_set.begin())
                  ->Run(&model, /*op_index=*/0, &modified)
                  .ok());
  EXPECT_THAT(model.GetArrayMap().size(), 1);

  auto& concatenated_array = (*model.GetArrayMap().begin()).second;
  EXPECT_THAT(concatenated_array->GetBuffer<toco::ArrayDataType::kFloat>().data,
              ElementsAreArray(ArrayFloatNear(
                  {0.,  1.,  2.,  3.,  10., 11., 12., 13., 20., 21., 22.,
                   23., 30., 31., 32., 33., 4.,  5.,  6.,  7.,  14., 15.,
                   16., 17., 24., 25., 26., 27., 34., 35., 36., 37.})));
}

TEST_F(ResolveConstantConcatenationTest, ConcatAtAxis2) {
  Model model;
  const int axis = 2;
  PrepareModel(&model, axis);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::ResolveConstantConcatenation);
  EXPECT_THAT(model.GetArrayMap().size(), 5);
  bool modified;
  ASSERT_TRUE((*graph_transformation_set.begin())
                  ->Run(&model, /*op_index=*/0, &modified)
                  .ok());
  EXPECT_THAT(model.GetArrayMap().size(), 1);

  auto& concatenated_array = (*model.GetArrayMap().begin()).second;
  EXPECT_THAT(concatenated_array->GetBuffer<toco::ArrayDataType::kFloat>().data,
              ElementsAreArray(ArrayFloatNear(
                  {0.,  1.,  10., 11., 20., 21., 30., 31., 2.,  3.,  12.,
                   13., 22., 23., 32., 33., 4.,  5.,  14., 15., 24., 25.,
                   34., 35., 6.,  7.,  16., 17., 26., 27., 36., 37.})));
}

}  // namespace toco
