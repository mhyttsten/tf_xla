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
class MHTracer_DTPStensorflowPScompilerPSxlaPSindex_util_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSindex_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSindex_util_testDTcc() {
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

#include "tensorflow/compiler/xla/index_util.h"

#include <initializer_list>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

void SetMinorToMajorLayout(Shape* shape, std::vector<int64_t> dimensions) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSindex_util_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/xla/index_util_test.cc", "SetMinorToMajorLayout");

  shape->mutable_layout()->clear_minor_to_major();
  for (auto dimension : dimensions) {
    shape->mutable_layout()->add_minor_to_major(dimension);
  }
}

TEST(IndexUtilTest, VectorIndexing) {
  // Vectors are trivially laid out and the linear index should always be the
  // same as the "multidimensional" index.
  Shape vector_shape = ShapeUtil::MakeShape(F32, {100});
  EXPECT_EQ(42,
            IndexUtil::MultidimensionalIndexToLinearIndex(vector_shape, {42}));
  std::vector<int64_t> multi_index =
      IndexUtil::LinearIndexToMultidimensionalIndex(vector_shape, 42);
  EXPECT_EQ(1, multi_index.size());
  EXPECT_EQ(42, multi_index[0]);
}

TEST(IndexUtilTest, MatrixIndexingRowMajor) {
  // Set layout to [0, 1]. That is, row major.
  Shape matrix_shape_01 = ShapeUtil::MakeShape(F32, {10, 20});
  SetMinorToMajorLayout(&matrix_shape_01, {0, 1});

  // If index is {a, b} then linear index should be: a + b * 10
  EXPECT_EQ(0, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_01,
                                                             {0, 0}));
  EXPECT_EQ(199, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_01,
                                                               {9, 19}));
  EXPECT_EQ(53, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_01,
                                                              {3, 5}));
  EXPECT_EQ(std::vector<int64_t>({3, 5}),
            IndexUtil::LinearIndexToMultidimensionalIndex(matrix_shape_01, 53));
}

TEST(IndexUtilTest, MatrixIndexingColumnMajor) {
  // Set layout to [1, 0]. That is, column major.
  Shape matrix_shape_10 = ShapeUtil::MakeShape(F32, {10, 20});
  SetMinorToMajorLayout(&matrix_shape_10, {1, 0});

  // If index is {a, b} then linear index should be: a * 20 + b
  EXPECT_EQ(0, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_10,
                                                             {0, 0}));
  EXPECT_EQ(199, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_10,
                                                               {9, 19}));
  EXPECT_EQ(65, IndexUtil::MultidimensionalIndexToLinearIndex(matrix_shape_10,
                                                              {3, 5}));
  EXPECT_EQ(std::vector<int64_t>({3, 5}),
            IndexUtil::LinearIndexToMultidimensionalIndex(matrix_shape_10, 65));
}

TEST(IndexUtilTest, ThreeDArrayIndexing210) {
  // Set layout to [2, 1, 0]. That is, column major.
  Shape shape_210 = ShapeUtil::MakeShape(F32, {10, 20, 30});
  SetMinorToMajorLayout(&shape_210, {2, 1, 0});

  // If index is {a, b, c} then linear index should be:
  // a * 20 * 30 + b * 30 + c
  EXPECT_EQ(1957, IndexUtil::MultidimensionalIndexToLinearIndex(shape_210,
                                                                {3, 5, 7}));
  EXPECT_EQ(5277, IndexUtil::MultidimensionalIndexToLinearIndex(shape_210,
                                                                {8, 15, 27}));
}

TEST(IndexUtilTest, ThreeDArrayIndexing120) {
  // Set layout to [1, 2, 0]
  Shape shape_120 = ShapeUtil::MakeShape(F32, {10, 20, 30});
  SetMinorToMajorLayout(&shape_120, {1, 2, 0});

  // If index is {a, b, c} then linear index should be:
  // a * 20 * 30 + b + c * 20
  EXPECT_EQ(1945, IndexUtil::MultidimensionalIndexToLinearIndex(shape_120,
                                                                {3, 5, 7}));
  EXPECT_EQ(5355, IndexUtil::MultidimensionalIndexToLinearIndex(shape_120,
                                                                {8, 15, 27}));
}

TEST(IndexUtilTest, FourDArrayIndexing3210) {
  // Set layout to [3, 2, 1,0]. That is, column major.
  Shape shape_3210 = ShapeUtil::MakeShape(F32, {10, 20, 30, 40});
  SetMinorToMajorLayout(&shape_3210, {3, 2, 1, 0});

  // If index is {a, b, c, d} then linear index should be:
  // a * 20 * 30 * 40 + b * 30 * 40 + c * 40 + d
  EXPECT_EQ(78289, IndexUtil::MultidimensionalIndexToLinearIndex(shape_3210,
                                                                 {3, 5, 7, 9}));
  EXPECT_EQ(211113, IndexUtil::MultidimensionalIndexToLinearIndex(
                        shape_3210, {8, 15, 27, 33}));
}

TEST(IndexUtilTest, LinearToMultiToLinear) {
  // Verify that converting a linear index to a multidimensional index and back
  // always returns the same value for different crazy shapes.  Shape has
  // 1440000000 elements. Inputs are randomly-ish selected.
  std::vector<int64_t> linear_indexes = {0,        1439999999, 1145567336,
                                         43883404, 617295214,  1117613654};

  std::vector<std::vector<int64_t>> minor_to_major_orders;
  minor_to_major_orders.push_back({6, 5, 4, 3, 2, 1, 0});
  minor_to_major_orders.push_back({0, 1, 2, 3, 4, 5, 6});
  minor_to_major_orders.push_back({4, 5, 1, 2, 6, 0, 3});

  for (auto minor_to_major_order : minor_to_major_orders) {
    Shape shape = ShapeUtil::MakeShape(F32, {10, 20, 30, 40, 30, 20, 10});
    SetMinorToMajorLayout(&shape, minor_to_major_order);
    for (auto linear_index : linear_indexes) {
      std::vector<int64_t> multi_index =
          IndexUtil::LinearIndexToMultidimensionalIndex(shape, linear_index);
      EXPECT_EQ(linear_index, IndexUtil::MultidimensionalIndexToLinearIndex(
                                  shape, multi_index));
    }
  }
}

TEST(IndexUtilTest, BumpIndices2x2) {
  auto shape = ShapeUtil::MakeShape(S32, {2, 2});
  std::vector<int64_t> indices = {0, 0};
  EXPECT_TRUE(IndexUtil::BumpIndices(shape, absl::MakeSpan(indices)));
  EXPECT_THAT(indices, ::testing::ElementsAre(0, 1));
  EXPECT_TRUE(IndexUtil::BumpIndices(shape, absl::MakeSpan(indices)));
  EXPECT_THAT(indices, ::testing::ElementsAre(1, 0));
  EXPECT_TRUE(IndexUtil::BumpIndices(shape, absl::MakeSpan(indices)));
  EXPECT_THAT(indices, ::testing::ElementsAre(1, 1));
  EXPECT_FALSE(IndexUtil::BumpIndices(shape, absl::MakeSpan(indices)));
}

}  // namespace
}  // namespace xla
