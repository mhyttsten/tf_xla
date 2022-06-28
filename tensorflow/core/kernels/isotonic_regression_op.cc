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
class MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc() {
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
#include <cmath>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/threadpool.h"

namespace {

using ::int64_t;
using tensorflow::int32;

// The # of ops estimated for the isotonic regression solver is the size of the
// array multiplied by this constant. This is used by the thread pool executor
// when deciding how many threads to use.
constexpr int kCostMultiplier = 100;

// In separable chain-constrained problems, i.e., those of the form
//
//  min_{y_1 >= y_2 >= ... >= y_n} \sum_{i=1}^n h_i(y_i)
//
// for any set of convex functions h_i, of particular importance are contiguous
// segments of coordinates, which this class represents. The interval is assumed
// to be half-closed and equal to [col_start(), col_limit()).
class Segment {
 public:
  // Creates the [col_index, col_index+1).
  explicit Segment(int col_index)
      : col_start_(col_index), col_limit_(col_index + 1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/isotonic_regression_op.cc", "Segment");
}

  // Returns the number of points in the segment.
  int num_points() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/isotonic_regression_op.cc", "num_points");
 return col_limit_ - col_start_; }

  // Merge another segment into this one.
  void merge_with(const Segment& other) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/kernels/isotonic_regression_op.cc", "merge_with");

    col_start_ = std::min(col_start_, other.col_start());
    col_limit_ = std::max(col_limit_, other.col_limit());
  }

  int col_start() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc mht_3(mht_3_v, 233, "", "./tensorflow/core/kernels/isotonic_regression_op.cc", "col_start");
 return col_start_; }

  int col_limit() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc mht_4(mht_4_v, 238, "", "./tensorflow/core/kernels/isotonic_regression_op.cc", "col_limit");
 return col_limit_; }

 private:
  int col_start_;
  int col_limit_;
};

// If we can solve for each segment {j, j+1, ..., j+m} the interval problem
//
//  argmin_y \sum_{i=j}^{j+m} h_i(y),
//
// we can use such an oracle to solve the general problem. The following class
// implements such an oracle for the case when h_i is the squared (l2) loss,
// or formally h_i(y) = (y - x_i)^2, where x_i is the i-th input.
//
// TODO(josipd): We know how and can extend this to other functions if needed.
template <typename T>
class L2PavaSegment : public Segment {
 public:
  L2PavaSegment(T y, int col_index)
      : Segment(col_index), y_sum_(y), minimum_(y) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc mht_5(mht_5_v, 261, "", "./tensorflow/core/kernels/isotonic_regression_op.cc", "L2PavaSegment");
}

  void merge_with(const L2PavaSegment& other) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc mht_6(mht_6_v, 266, "", "./tensorflow/core/kernels/isotonic_regression_op.cc", "merge_with");

    Segment::merge_with(other);
    y_sum_ += other.y_sum_;
    minimum_ = y_sum_ / static_cast<T>(num_points());
  }

  T minimum() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc mht_7(mht_7_v, 275, "", "./tensorflow/core/kernels/isotonic_regression_op.cc", "minimum");
 return minimum_; }

 private:
  T y_sum_;    // The sum of the inputs within the segment.
  T minimum_;  // The minimum, cached to avoid expensive divisions.
};

// Solve one of the problems in the batch (the row_index'th one) using the
// pool-adjacent violators algorithm (PAVA).
//
// The PAVA algorithm goes back to
//
// Nonmetric Multidimensional Scaling: A numerical method
// Kruskal, J. B. (1964), Psychometrika (1964)
//
// For a more recent analysis, please refer to
//
// Active set algorithms for isotonic regression; a unifying framework
// Best, Michael J., and Nilotpal Chakravarti
// Mathematical Programming 47.1-3 (1990)
//
// Intuitively, the algorithm splits the inputs into blocks (starting from
// singleton ones), and then whenever there are two consecutive blocks whose
// minima violate the inequality constraint, they are merged. The solution is
// then block-wise constant, each block equal to the corresponding minimum.
//
// The tensors should be two dimensional, and the segment objects should
// support the minimum() and merge_with() methods.
template <typename SegmentType, typename FloatTensor, typename IntTensor>
void solve_pava(const std::function<SegmentType(int, int)>& make_segment,
                FloatTensor* solution, IntTensor* segments, int row_index) {
  const size_t n = solution->dimensions()[1];
  std::vector<SegmentType> pools;
  pools.reserve(n);

  for (size_t col_index = 0; col_index < n; ++col_index) {
    pools.push_back(make_segment(row_index, col_index));

    // While the last two pools are decreasing, merge them.
    while (pools.size() > 1 &&
           pools.rbegin()->minimum() > (pools.rbegin() + 1)->minimum()) {
      (pools.rbegin() + 1)->merge_with(*pools.rbegin());
      pools.pop_back();
    }
  }

  int segment_id = 0;
  for (const auto& pool : pools) {
    const auto pool_minimum = pool.minimum();
    // The matrices are row major, so we can scan the memory linearly.
    auto* solution_ptr = &(*solution)(row_index, pool.col_start());
    auto* segments_ptr = &(*segments)(row_index, pool.col_start());
    for (int i = pool.col_start(); i < pool.col_limit(); ++i) {
      *solution_ptr++ = pool_minimum;
      *segments_ptr++ = segment_id;
    }
    ++segment_id;
  }
}

// Solve a batch of problems using the pool-adjacent violators algorithm.
// The problems are solved in parallel using tensorflow's thread pool.
template <typename SegmentType, typename FloatTensor, typename IntTensor>
void solve_pava_batch(const std::function<SegmentType(int, int)>& make_segment,
                      FloatTensor* solution, IntTensor* segments,
                      tensorflow::OpKernelContext* context) {
  const int batch_size = solution->dimensions()[0];
  const int problem_size = solution->dimensions()[1];

  auto thread_pool =
      context->device()->tensorflow_cpu_worker_threads()->workers;

  thread_pool->ParallelFor(
      batch_size, kCostMultiplier * problem_size,
      [&make_segment, &solution, &segments](int64_t row_start,
                                            int64_t row_limit) {
        // Casting to int is safe, as we do boundary checks in `Compute`.
        for (int row_index = static_cast<int>(row_start);
             row_index < static_cast<int>(row_limit); ++row_index) {
          solve_pava(make_segment, solution, segments, row_index);
        }
      });
}

}  // namespace

template <typename Tin, typename Tout>
class IsotonicRegressionOp : public tensorflow::OpKernel {
 public:
  explicit IsotonicRegressionOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc mht_8(mht_8_v, 368, "", "./tensorflow/core/kernels/isotonic_regression_op.cc", "IsotonicRegressionOp");
}

  void Compute(tensorflow::OpKernelContext* context) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc mht_9(mht_9_v, 373, "", "./tensorflow/core/kernels/isotonic_regression_op.cc", "Compute");

    // Grab the input tensor.
    const tensorflow::Tensor& input_tensor = context->input(0);
    const auto input = input_tensor.flat_inner_dims<Tin, 2>();
    int int_max = std::numeric_limits<int32>::max();
    OP_REQUIRES(context,
                tensorflow::FastBoundsCheck(input.dimensions()[0], int_max) &&
                    tensorflow::FastBoundsCheck(input.dimensions()[1], int_max),
                tensorflow::errors::InvalidArgument("Tensor too large"));

    // Create the output tensor holding the minimizers.
    const auto shape = input_tensor.shape();
    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, shape, &output_tensor));
    auto output = output_tensor->flat_inner_dims<Tout, 2>();

    // Create the output tensor holidng the segment memberships.
    tensorflow::Tensor* segments_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, shape, &segments_tensor));
    auto segments = segments_tensor->flat_inner_dims<int>();

    auto make_l2_segment = [&input](int row_index, int col_index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSisotonic_regression_opDTcc mht_10(mht_10_v, 399, "", "./tensorflow/core/kernels/isotonic_regression_op.cc", "lambda");

      return L2PavaSegment<Tout>(input(row_index, col_index), col_index);
    };
    solve_pava_batch<L2PavaSegment<Tout>>(make_l2_segment, &output, &segments,
                                          context);
  }
};

#define REGISTER_CPU_KERNEL(Tin, Tout)                               \
  REGISTER_KERNEL_BUILDER(Name("IsotonicRegression")                 \
                              .Device(tensorflow::DEVICE_CPU)        \
                              .TypeConstraint<Tin>("T")              \
                              .TypeConstraint<Tout>("output_dtype"), \
                          IsotonicRegressionOp<Tin, Tout>);

// Float types have the same input and output.
#define REGISTER_CPU_SAME_KERNEL(T) REGISTER_CPU_KERNEL(T, T)
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SAME_KERNEL);

// 8 and 16 bit integers get converted to 32 bit floats.
#define REGISTER_CPU_KERNEL_FLOAT(Tin) REGISTER_CPU_KERNEL(Tin, float)
TF_CALL_int16(REGISTER_CPU_KERNEL_FLOAT);
TF_CALL_int8(REGISTER_CPU_KERNEL_FLOAT);

// 32 and 64 bit integers get converted to 64 bit floats.
#define REGISTER_CPU_KERNEL_DOUBLE(Tin) REGISTER_CPU_KERNEL(Tin, double)
TF_CALL_int64(REGISTER_CPU_KERNEL_DOUBLE);
TF_CALL_int32(REGISTER_CPU_KERNEL_DOUBLE);
