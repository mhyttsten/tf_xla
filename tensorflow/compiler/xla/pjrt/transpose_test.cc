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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/transpose.h"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/numeric/int128.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace xla {

class TestTransposePlan : public TransposePlan {
 public:
  using TransposePlan::CoalesceDimensions;
  using TransposePlan::RemoveTrivialDimensions;
};

TEST(TransposeTest, RemoveTrivialDimensions) {
  absl::InlinedVector<int64_t, 4> dims = {4, 5, 1, 3, 1, 2, 5};
  absl::InlinedVector<int64_t, 4> perm = {0, 2, 1, 4, 3, 6, 5};
  absl::InlinedVector<int64_t, 4> lda = {2, 5, 7, 100, 3, 0, 1};
  absl::InlinedVector<int64_t, 4> lda_tile = {1, 1, 1, 1, 1, 1, 1};
  absl::InlinedVector<int64_t, 4> input_tiling = {1, 1, 1, 1, 1, 1, 1};
  absl::InlinedVector<int64_t, 4> output_tiling = {1, 1, 1, 1, 1, 1, 1};
  TestTransposePlan::RemoveTrivialDimensions(dims, perm, lda, lda_tile,
                                             input_tiling, output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(4, 5, 3, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(0, 1, 2, 4, 3));

  dims = {4, 5, 3, 2, 5};
  perm = {4, 3, 2, 1, 0};
  lda = {2, 5, 100, 0, 1};
  lda_tile = {1, 1, 1, 1, 1};
  input_tiling = {1, 1, 1, 1, 1};
  output_tiling = {1, 1, 1, 1, 1};
  TestTransposePlan::RemoveTrivialDimensions(dims, perm, lda, lda_tile,
                                             input_tiling, output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(4, 5, 3, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(4, 3, 2, 1, 0));
}

TEST(TransposeTest, CoalesceDimensions) {
  absl::InlinedVector<int64_t, 4> dims = {4, 5, 1, 3, 1, 2, 5};
  absl::InlinedVector<int64_t, 4> perm = {0, 2, 1, 4, 3, 6, 5};
  absl::InlinedVector<int64_t, 4> lda = {50, 30, 30, 10, 10, 5, 1};
  absl::InlinedVector<int64_t, 4> lda_tile = {1, 1, 1, 1, 1, 1, 1};
  absl::InlinedVector<int64_t, 4> input_tiling = {1, 1, 1, 1, 1, 1, 1};
  absl::InlinedVector<int64_t, 4> output_tiling = {1, 1, 1, 1, 1, 1, 1};
  TestTransposePlan::CoalesceDimensions(dims, perm, lda, lda_tile, input_tiling,
                                        output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(4, 5, 1, 3, 1, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(0, 2, 1, 4, 3, 6, 5));
  EXPECT_THAT(lda, testing::ElementsAre(50, 30, 30, 10, 10, 5, 1));

  dims = {4, 5, 3, 2, 5};
  perm = {4, 1, 2, 3, 0};
  lda = {150, 30, 10, 5, 1};
  lda_tile = {1, 1, 1, 1, 1};
  input_tiling = {1, 1, 1, 1, 1};
  output_tiling = {1, 1, 1, 1, 1};
  TestTransposePlan::CoalesceDimensions(dims, perm, lda, lda_tile, input_tiling,
                                        output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(4, 30, 5));
  EXPECT_THAT(perm, testing::ElementsAre(2, 1, 0));
  EXPECT_THAT(lda, testing::ElementsAre(150, 5, 1));

  dims = {4, 5, 3, 2, 5};
  perm = {0, 1, 2, 3, 4};
  lda = {150, 30, 10, 5, 1};
  lda_tile = {1, 1, 1, 1, 1};
  input_tiling = {1, 1, 1, 1, 1};
  output_tiling = {1, 1, 1, 1, 1};
  TestTransposePlan::CoalesceDimensions(dims, perm, lda, lda_tile, input_tiling,
                                        output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(600));
  EXPECT_THAT(perm, testing::ElementsAre(0));
  EXPECT_THAT(lda, testing::ElementsAre(1));

  dims = {4, 5, 3, 2, 5};
  perm = {4, 1, 2, 3, 0};
  lda = {150, 30, 10, 7, 1};  // Non-standard stridings prevent coalescing.
  lda_tile = {1, 1, 1, 1, 1};
  input_tiling = {1, 1, 1, 1, 1};
  output_tiling = {1, 1, 1, 1, 1};
  TestTransposePlan::CoalesceDimensions(dims, perm, lda, lda_tile, input_tiling,
                                        output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(4, 15, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(3, 1, 2, 0));
  EXPECT_THAT(lda, testing::ElementsAre(150, 10, 7, 1));
}

TEST(TransposeTest, InvalidTilings) {
  auto plan =
      TransposePlan::Create(sizeof(float), {3, 4, 5}, {0, 1, 2},
                            /*input_layout=*/TransposePlan::Tiling{{8, 128}},
                            /*output_tiling=*/TransposePlan::Tiling{{4}});
  EXPECT_EQ(plan.status().code(), tensorflow::error::UNIMPLEMENTED);
  EXPECT_THAT(
      plan.status().error_message(),
      testing::HasSubstr(
          "Only one of the input and output may have a non-trivial tiling"));
}

// Computes the size in elements of a tiled array.
int64_t SizeOfTiledArray(absl::Span<int64_t const> shape,
                         absl::Span<int64_t const> tiling) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_0(mht_0_v, 301, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "SizeOfTiledArray");

  int64_t size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i >= shape.size() - tiling.size()) {
      size *= RoundUpTo(shape[i], tiling[i - (shape.size() - tiling.size())]);
    } else {
      size *= shape[i];
    }
  }
  return size;
}

// Advances 'indices' in the lexicographical order of the multidimensional
// array with `shape`. Returns false if the end of the array has been reached.
bool BumpIndices(absl::Span<int64_t const> shape, absl::Span<int64_t> indices) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_1(mht_1_v, 318, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "BumpIndices");

  CHECK_EQ(shape.size(), indices.size());
  for (int dimno = indices.size() - 1; dimno >= 0; --dimno) {
    if (indices[dimno] + 1 < shape[dimno]) {
      indices[dimno]++;
      // Whenever an index of a dimension is increased, it means that all
      // following dimensions have maxed out, so they must go to 0.
      std::fill(indices.begin() + dimno + 1, indices.end(), 0);
      return true;
    }
  }
  return false;
}

// Converts a multidimensional index `indices` into an array with `shape` and
// tiling `tiling` into a linear offset into a buffer.
int64_t IndexToLinearIndex(absl::Span<int64_t const> shape,
                           absl::Span<int64_t const> tiling,
                           absl::Span<int64_t const> indices) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_2(mht_2_v, 339, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "IndexToLinearIndex");

  CHECK_LE(tiling.size(), shape.size());
  CHECK_EQ(shape.size(), indices.size());
  int64_t stride = 1;
  int64_t offset = 0;

  auto index_it = indices.rbegin();
  auto tile_it = tiling.rbegin();
  for (; tile_it != tiling.rend(); ++index_it, ++tile_it) {
    offset += (*index_it % *tile_it) * stride;
    stride *= *tile_it;
  }
  index_it = indices.rbegin();
  tile_it = tiling.rbegin();
  auto shape_it = shape.rbegin();
  for (; tile_it != tiling.rend(); ++index_it, ++shape_it, ++tile_it) {
    offset += (*index_it / *tile_it) * stride;
    stride *= CeilOfRatio(*shape_it, *tile_it);
  }
  for (; shape_it != shape.rend(); ++index_it, ++shape_it) {
    offset += *index_it * stride;
    stride *= *shape_it;
  }
  return offset;
}

// Slow reference code that converts an array from an untiled layout into a
// tiled layout.
template <typename T>
std::vector<T> TileArray(const Array<T>& in, absl::Span<int64_t const> tiling) {
  std::vector<T> out(SizeOfTiledArray(in.dimensions(), tiling), -1);
  if (in.num_elements() == 0) {
    return out;
  }
  std::vector<int64_t> indices(in.num_dimensions(), 0);
  do {
    int64_t i = IndexToLinearIndex(in.dimensions(), tiling, indices);
    out.at(i) = in(indices);
  } while (BumpIndices(in.dimensions(), absl::MakeSpan(indices)));
  return out;
}

// Reference implementation: transpose using Eigen.
template <typename T, int NDIMS>
void TransposeUsingEigenNd(const T* input, T* output,
                           absl::Span<int64_t const> dims,
                           absl::Span<int64_t const> dims_out,
                           absl::Span<int64_t const> permutation) {
  typedef Eigen::TensorMap<
      Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Aligned>
      Tensor;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Aligned>
      ConstTensor;

  Eigen::array<int, NDIMS> p;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> dims_eigen;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> dims_out_eigen;
  for (int i = 0; i < NDIMS; ++i) {
    p[i] = permutation[i];
    dims_eigen[i] = dims[i];
    dims_out_eigen[i] = dims_out[i];
  }
  auto x = ConstTensor(input, dims_eigen);
  auto y = Tensor(output, dims_out_eigen);
  y = x.shuffle(p);
}

template <typename T>
void TransposeUsingEigen(const T* input, T* output,
                         absl::Span<int64_t const> dims,
                         absl::Span<int64_t const> dims_out,
                         absl::Span<int64_t const> permutation) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_3(mht_3_v, 416, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "TransposeUsingEigen");

  switch (dims.size()) {
    case 0:
      return;
    case 1:
      TransposeUsingEigenNd<T, 1>(input, output, dims, dims_out, permutation);
      return;
    case 2:
      TransposeUsingEigenNd<T, 2>(input, output, dims, dims_out, permutation);
      return;
    case 3:
      TransposeUsingEigenNd<T, 3>(input, output, dims, dims_out, permutation);
      return;
    case 4:
      TransposeUsingEigenNd<T, 4>(input, output, dims, dims_out, permutation);
      return;
    default:
      LOG(FATAL) << "Unimplemented Eigen transpose rank";
  }
}

struct TransposeTestCase {
  TransposeTestCase(std::vector<int64_t> dims, std::vector<int64_t> permutation,
                    std::vector<int64_t> input_tiling = {},
                    std::vector<int64_t> output_tiling = {})
      : dims(std::move(dims)),
        permutation(std::move(permutation)),
        input_tiling(std::move(input_tiling)),
        output_tiling(std::move(output_tiling)) {}

  std::vector<int64_t> dims;
  std::vector<int64_t> permutation;
  std::vector<int64_t> input_tiling;
  std::vector<int64_t> output_tiling;

  std::string ToString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_4(mht_4_v, 454, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "ToString");

    return absl::StrFormat(
        "[%s],perm=[%s],tiling=[%s]/[%s]", absl::StrJoin(dims, ","),
        absl::StrJoin(permutation, ","), absl::StrJoin(input_tiling, ","),
        absl::StrJoin(output_tiling, ","));
  }
};

std::ostream& operator<<(std::ostream& os, const TransposeTestCase& test) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_5(mht_5_v, 465, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "operator<<");

  os << test.ToString();
  return os;
}

std::vector<TransposeTestCase> GetTransposeTestCases() {
  std::vector<TransposeTestCase> cases = {
      TransposeTestCase(/*dims=*/{1}, /*permutation=*/{0}),
      TransposeTestCase(/*dims=*/{4}, /*permutation=*/{0}),
      TransposeTestCase(/*dims=*/{27}, /*permutation=*/{0}),
      TransposeTestCase(/*dims=*/{1, 1}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{1, 1}, /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{2, 2}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{4, 4}, /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{4, 4}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{4, 4}, /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{8, 8}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{8, 8}, /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{16, 16}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{16, 16}, /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{11, 15}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{11, 15}, /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{11, 15, 13}, /*permutation=*/{0, 1, 2}),
      TransposeTestCase(/*dims=*/{11, 15, 13}, /*permutation=*/{0, 2, 1}),
      TransposeTestCase(/*dims=*/{11, 15, 13}, /*permutation=*/{1, 2, 0}),
      TransposeTestCase(/*dims=*/{11, 15, 13}, /*permutation=*/{1, 0, 2}),
      TransposeTestCase(/*dims=*/{11, 15, 13}, /*permutation=*/{2, 0, 1}),
      TransposeTestCase(/*dims=*/{64, 64, 64}, /*permutation=*/{2, 1, 0}),
      TransposeTestCase(/*dims=*/{256, 256, 256}, /*permutation=*/{2, 1, 0}),
      TransposeTestCase(/*dims=*/{4, 8, 16, 32}, /*permutation=*/{3, 1, 0, 2}),
      TransposeTestCase(/*dims=*/{64, 224, 224, 3},
                        /*permutation=*/{3, 1, 2, 0}),

      TransposeTestCase(/*dims=*/{3}, /*permutation=*/{0},
                        /*input_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{3}, /*permutation=*/{0},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{2, 4, 6}, /*permutation=*/{0, 1, 2},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{2, 3}),
      TransposeTestCase(/*dims=*/{4}, /*permutation=*/{0},
                        /*input_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{5}, /*permutation=*/{0},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{8}, /*permutation=*/{0},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{8}, /*permutation=*/{0},
                        /*input_tiling=*/{3},
                        /*output_tiling=*/{}),
      TransposeTestCase(/*dims=*/{29}, /*permutation=*/{0},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{12, 7}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{4}),
      TransposeTestCase(/*dims=*/{12, 7}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{}, /*output_tiling=*/{5}),
      TransposeTestCase(/*dims=*/{12, 7}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{2, 4}),
      TransposeTestCase(/*dims=*/{12, 7}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{}, /*output_tiling=*/{5, 2}),
      TransposeTestCase(/*dims=*/{128, 224, 224, 3},
                        /*permutation=*/{3, 1, 2, 0},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{8, 128}),
  };
  return cases;
}

class TransposeTest : public ::testing::TestWithParam<TransposeTestCase> {
 protected:
  template <typename T>
  void TestTranspose(int parallelism) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_6(mht_6_v, 542, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "TestTranspose");

    const TransposeTestCase test = GetParam();
    tensorflow::thread::ThreadPool threadpool(tensorflow::Env::Default(),
                                              "Transpose", parallelism);
    std::vector<int64_t> output_dims = Permute(test.dims, test.permutation);
    TF_ASSERT_OK_AND_ASSIGN(
        auto plan, TransposePlan::Create(
                       sizeof(T), test.dims, test.permutation,
                       TransposePlan::Tiling{test.input_tiling},
                       TransposePlan::Tiling{test.output_tiling},
                       TransposePlan::Transformation::kNone, parallelism));
    VLOG(1) << plan->ToString();
    xla::Array<T> untiled_input(test.dims);
    untiled_input.FillIota(0);
    xla::Array<T> expected_untiled_output(output_dims);
    TransposeUsingEigen(untiled_input.data(), expected_untiled_output.data(),
                        test.dims, output_dims, test.permutation);

    auto tiled_input = TileArray(untiled_input, test.input_tiling);
    auto expected_tiled_output =
        TileArray(expected_untiled_output, test.output_tiling);

    std::vector<T> output(
        SizeOfTiledArray(plan->OutputDims(), test.output_tiling), -1);
    plan->Execute(
        tiled_input.data(), output.data(),
        [&](std::function<void()> fn) { threadpool.Schedule(std::move(fn)); });

    EXPECT_EQ(expected_tiled_output, output);
  }
};

TEST_P(TransposeTest, TransposeInt8) { TestTranspose<int8_t>(1); }
TEST_P(TransposeTest, TransposeInt16) { TestTranspose<int16_t>(1); }
TEST_P(TransposeTest, TransposeInt32) { TestTranspose<int32_t>(1); }
TEST_P(TransposeTest, TransposeInt64) { TestTranspose<int64_t>(1); }
TEST_P(TransposeTest, TransposeInt128) { TestTranspose<absl::int128>(1); }

TEST_P(TransposeTest, ParallelTransposeInt8) { TestTranspose<int8_t>(16); }
TEST_P(TransposeTest, ParallelTransposeInt32) { TestTranspose<int32_t>(16); }

INSTANTIATE_TEST_SUITE_P(TransposeTestInstance, TransposeTest,
                         ::testing::ValuesIn(GetTransposeTestCases()));

TEST(TransposeTest, NegativeStrides1D) {
  int64_t n = 10;
  std::vector<int32_t> input(n);
  std::vector<int32_t> output(n);
  std::vector<int32_t> expected(n);
  absl::c_iota(input, int32_t{7});
  std::iota(expected.rbegin(), expected.rend(), 7);
  TF_ASSERT_OK_AND_ASSIGN(
      auto plan, TransposePlan::Create(
                     sizeof(int32_t), {n}, /*permutation=*/{0},
                     TransposePlan::Striding{{-int64_t{sizeof(int32_t)}}}));
  plan->Execute(input.data() + (n - 1), output.data());
  EXPECT_EQ(expected, output);
}

TEST(TransposeTest, NegativeStrides2D) {
  xla::Array<int16_t> input = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
  };
  xla::Array<int16_t> expected = {
      {4, 8, 12},
      {3, 7, 11},
      {2, 6, 10},
      {1, 5, 9},
  };
  xla::Array<int16_t> output({4, 3});
  TF_ASSERT_OK_AND_ASSIGN(
      auto plan, TransposePlan::Create(
                     sizeof(int16_t), {3, 4}, /*permutation=*/{1, 0},
                     TransposePlan::Striding{
                         {4 * sizeof(int16_t), -int64_t{sizeof(int16_t)}}}));
  plan->Execute(input.data() + 3, output.data());
  EXPECT_EQ(expected, output);
}

static std::vector<TransposeTestCase> BenchmarkCases() {
  return std::vector<TransposeTestCase>{
      TransposeTestCase(/*dims=*/{256, 256},
                        /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{512, 512},
                        /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{1024, 1024},
                        /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{256, 256, 256},
                        /*permutation=*/{0, 2, 1}),
      TransposeTestCase(/*dims=*/{256, 256, 256},
                        /*permutation=*/{1, 0, 2}),
      TransposeTestCase(/*dims=*/{256, 256, 256},
                        /*permutation=*/{1, 2, 0}),
      TransposeTestCase(/*dims=*/{256, 256, 256},
                        /*permutation=*/{2, 0, 1}),
      TransposeTestCase(/*dims=*/{256, 256, 256},
                        /*permutation=*/{2, 1, 0}),
      TransposeTestCase(/*dims=*/{512, 512, 512},
                        /*permutation=*/{0, 2, 1}),
      TransposeTestCase(/*dims=*/{512, 512, 512},
                        /*permutation=*/{1, 0, 2}),
      TransposeTestCase(/*dims=*/{512, 512, 512},
                        /*permutation=*/{1, 2, 0}),
      TransposeTestCase(/*dims=*/{512, 512, 512},
                        /*permutation=*/{2, 0, 1}),
      TransposeTestCase(/*dims=*/{512, 512, 512},
                        /*permutation=*/{2, 1, 0}),
      TransposeTestCase(/*dims=*/{64, 224, 224, 3},
                        /*permutation=*/{1, 2, 3, 0}),
      TransposeTestCase(/*dims=*/{256, 64, 64, 3},
                        /*permutation=*/{1, 3, 2, 0}),
  };
}

template <typename T>
void BM_Eigen(const TransposeTestCase& bm, int parallelism,
              ::testing::benchmark::State& state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_7(mht_7_v, 663, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "BM_Eigen");

  CHECK_EQ(parallelism, 1);
  Array<T> input(bm.dims);
  input.FillIota(0);
  std::vector<int64_t> output_dims = Permute(bm.dims, bm.permutation);
  Array<T> output(output_dims);
  for (auto s : state) {
    TransposeUsingEigen(input.data(), output.data(), bm.dims, output_dims,
                        bm.permutation);
    tensorflow::testing::DoNotOptimize(output);
  }
}
static void BM_Eigen_uint8(const TransposeTestCase& bm, int parallelism,
                           ::testing::benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_8(mht_8_v, 679, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "BM_Eigen_uint8");

  BM_Eigen<uint8_t>(std::move(bm), parallelism, state);
}
static void BM_Eigen_float(const TransposeTestCase& bm, int parallelism,
                           ::testing::benchmark::State& state) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_9(mht_9_v, 686, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "BM_Eigen_float");

  BM_Eigen<float>(bm, parallelism, state);
}

template <typename T>
void BM_Transpose(const TransposeTestCase& bm, int parallelism,
                  ::testing::benchmark::State& state) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_10(mht_10_v, 695, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "BM_Transpose");

  TF_ASSERT_OK_AND_ASSIGN(
      auto plan,
      TransposePlan::Create(sizeof(T), bm.dims, bm.permutation,
                            TransposePlan::Tiling{}, TransposePlan::Tiling{},
                            TransposePlan::Transformation::kNone, parallelism));
  Array<T> input(bm.dims);
  input.FillIota(0);
  std::vector<int64_t> output_dims = Permute(bm.dims, bm.permutation);
  Array<T> output(output_dims);
  tensorflow::thread::ThreadPool threadpool(tensorflow::Env::Default(),
                                            "Transpose", parallelism);
  for (auto s : state) {
    plan->Execute(input.data(), output.data(), [&](std::function<void()> fn) {
      threadpool.Schedule(std::move(fn));
    });
    tensorflow::testing::DoNotOptimize(output);
  }
}
static void BM_Transpose_uint8(const TransposeTestCase& bm, int parallelism,
                               ::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_11(mht_11_v, 718, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "BM_Transpose_uint8");

  BM_Transpose<uint8_t>(bm, parallelism, state);
}
static void BM_Transpose_float(const TransposeTestCase& bm, int parallelism,
                               ::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_12(mht_12_v, 725, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "BM_Transpose_float");

  BM_Transpose<float>(bm, parallelism, state);
}

static void* benchmarks = []() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_testDTcc mht_13(mht_13_v, 732, "", "./tensorflow/compiler/xla/pjrt/transpose_test.cc", "lambda");

  using BenchmarkFn =
      void (*)(const TransposeTestCase&, int, testing::benchmark::State&);
  std::vector<std::tuple<std::string, BenchmarkFn, std::vector<int>>> variants =
      {
          {"BM_Eigen_uint8", BM_Eigen_uint8, {1}},
          {"BM_Transpose_uint8", BM_Transpose_uint8, {1, 4, 8}},  //
          {"BM_Eigen_float", BM_Eigen_float, {1}},
          {"BM_Transpose_float", BM_Transpose_float, {1, 4, 8}},  //
  };
  auto benchmark_cases = BenchmarkCases();
  for (const auto& benchmark_case : benchmark_cases) {
    for (const auto& variant : variants) {
      for (int num_threads : std::get<2>(variant)) {
        std::string name =
            absl::StrCat(std::get<0>(variant), "_threads_", num_threads, "_",
                         absl::StrJoin(benchmark_case.dims, "_"), "_perm_",
                         absl::StrJoin(benchmark_case.permutation, "_"));

        TransposeTestCase testcase = benchmark_case;
        BenchmarkFn fn = std::get<1>(variant);
        benchmark::RegisterBenchmark(
            name.c_str(), [fn, num_threads, testcase](benchmark::State& state) {
              fn(testcase, num_threads, state);
            });
      }
    }
  }
  return nullptr;
}();

TEST(TransposePlanCache, Basics) {
  TransposePlanCache cache(2);
  TF_ASSERT_OK_AND_ASSIGN(
      auto p1, cache.GetOrCreate(/*elem_size_in_bytes=*/4, /*dims=*/{1, 2, 3},
                                 /*permutation=*/{2, 1, 0}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto p1a, cache.GetOrCreate(/*elem_size_in_bytes=*/4, /*dims=*/{1, 2, 3},
                                  /*permutation=*/{2, 1, 0}));
  EXPECT_TRUE(p1.get() == p1a.get());
  TF_ASSERT_OK_AND_ASSIGN(
      auto p2, cache.GetOrCreate(/*elem_size_in_bytes=*/4, /*dims=*/{1, 2, 3},
                                 /*permutation=*/{1, 2, 0}));
  EXPECT_TRUE(p1.get() != p2.get());
  TF_ASSERT_OK_AND_ASSIGN(
      auto p3, cache.GetOrCreate(/*elem_size_in_bytes=*/4, /*dims=*/{1, 2, 3},
                                 /*permutation=*/{0, 1, 2}));
  EXPECT_TRUE(p3.get() != p1.get());
  TF_ASSERT_OK_AND_ASSIGN(
      auto p1b, cache.GetOrCreate(/*elem_size_in_bytes=*/4, /*dims=*/{1, 2, 3},
                                  /*permutation=*/{2, 1, 0}));
  EXPECT_TRUE(p1.get() != p1b.get());
}

}  // namespace xla
