/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_BCAST_H_
#define TENSORFLOW_CORE_UTIL_BCAST_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSbcastDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSbcastDTh() {
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


#include <algorithm>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Returns the mapping from the output batch indices to the corresponding
// input's batch indices, given the input's "reshape" and "bcast" shapes as
// returned by the BCastList helper class. The i'th element denotes the
// (flattened) batch index of the input that must be used to compute the i'th
// batch output.
//
inline void ComputeBatchIndices(const int64_t output_batch_size,
                                const gtl::InlinedVector<int64_t, 4>& reshape,
                                const gtl::InlinedVector<int64_t, 4>& bcast,
                                std::vector<int64_t>* out_indices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_0(mht_0_v, 206, "", "./tensorflow/core/util/bcast.h", "ComputeBatchIndices");

  // Populates the mapping in out_indices. This algorithm is identical to
  // the following steps:
  //  - Reshape {0, 1, ..., input_batch_size - 1} to the input shape.
  //  - Broadcast to the output shape.
  //  - Reshape back to a flat 1D vector.
  out_indices->resize(output_batch_size);
  int64_t num_output_elements = 1;
  int64_t num_input_elements = 1;
  for (int64_t i = reshape.size() - 1; i >= 0; --i) {
    // Replicate the already populated mapping an additional (dim - 1) times.
    // If we are broadcasting, just copy the existing mapping.
    // Otherwise, add another dimension from the input shape.
    const int64_t dim = std::max(reshape[i], bcast[i]);
    const int64_t incr = bcast[i] > 1 ? 0 : num_input_elements;
    for (int64_t k = 0; k < (dim - 1) * num_output_elements; ++k) {
      (*out_indices)[num_output_elements + k] = (*out_indices)[k] + incr;
    }
    num_output_elements *= dim;
    num_input_elements *= reshape[i];
  }
}

template <int N>
class BCastList {
 public:
  // A vector of int64 representing the shape of tensor. The 0-th
  // element is the outer-most dimension and the last element is the
  // inner-most dimension. Note that we do not use TensorShape since
  // it's more convenient to manipulate Vec directly for this module.
  typedef gtl::InlinedVector<int64_t, 4> Vec;

  // Constructs all helper shapes, following the aforementioned rules.
  //
  // If "fewer_dims_optimization" is set to true (the default), the
  // implementation tries to reduce intermediate dimensions needed to be more
  // efficient.  This is transparent to the caller.
  //
  // If false, all intermediate shapes (except for grad_{x,y}_reduce_idx()) have
  // the same number of dimensions as the larger of the two inputs.
  //
  // If return_flattened_batch_indices is true, the implementation will compute
  // for each output member of the flattened output, which batch indices of
  // each input correspond to it. This is disabled by default.
  explicit BCastList(const Vec (&x)[N],
                     const bool fewer_dims_optimization = true,
                     const bool return_flattened_batch_indices = false);
  ~BCastList() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_1(mht_1_v, 256, "", "./tensorflow/core/util/bcast.h", "~BCastList");
}

  // Returns true iff two operands are compatible according to the
  // broadcasting rule.
  bool IsValid() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_2(mht_2_v, 263, "", "./tensorflow/core/util/bcast.h", "IsValid");
 return valid_; }
  bool IsBroadcastingRequired() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_3(mht_3_v, 267, "", "./tensorflow/core/util/bcast.h", "IsBroadcastingRequired");
 return broadcasting_required_; }

  // If and only if IsValid(), the following fields can be used in
  // implementing a broadcasted binary tensor operation according to
  // the broadcasting rule.
  const Vec& reshape(int i) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_4(mht_4_v, 275, "", "./tensorflow/core/util/bcast.h", "reshape");
 return reshape_[i]; }
  const Vec& bcast(int i) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_5(mht_5_v, 279, "", "./tensorflow/core/util/bcast.h", "bcast");
 return bcast_[i]; }
  const Vec& result_shape() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_6(mht_6_v, 283, "", "./tensorflow/core/util/bcast.h", "result_shape");
 return result_; }
  const Vec& output_shape() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_7(mht_7_v, 287, "", "./tensorflow/core/util/bcast.h", "output_shape");
 return output_; }
  const Vec& grad_reduce_idx(int i) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_8(mht_8_v, 291, "", "./tensorflow/core/util/bcast.h", "grad_reduce_idx");
 return grad_reduce_idx_[i]; }
  const int64_t output_batch_size() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_9(mht_9_v, 295, "", "./tensorflow/core/util/bcast.h", "output_batch_size");
 return output_batch_size_; }

  // Returns the mapping from the flattened output batch indices to x's
  // flattened batch indices. The result is a vector of length
  // output_batch_size(). To compute the i'th batch output, a binary matmul-like
  // operation should use the `x_batch_indices()[i]`th batch index of `x`.
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64_t>& batch_indices(int i) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_10(mht_10_v, 306, "", "./tensorflow/core/util/bcast.h", "batch_indices");

    return batch_indices_[i];
  }

 protected:
  bool valid_ = true;
  bool broadcasting_required_ = true;
  Vec reshape_[N];
  Vec bcast_[N];
  Vec result_;
  Vec output_;
  Vec grad_reduce_idx_[N];

  int64_t output_batch_size_;
  std::vector<int64_t> batch_indices_[N];

  static void Reverse(Vec* shape) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_11(mht_11_v, 325, "", "./tensorflow/core/util/bcast.h", "Reverse");

    std::reverse(shape->begin(), shape->end());
  }

  TF_DISALLOW_COPY_AND_ASSIGN(BCastList);
};

template <int N>
BCastList<N>::BCastList(const BCastList::Vec (&x)[N],
                        const bool fewer_dims_optimization,
                        const bool return_flattened_batch_indices) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_12(mht_12_v, 338, "", "./tensorflow/core/util/bcast.h", "BCastList<N>::BCastList");

  typedef BCastList::Vec Vec;

  // Safely multiplies dimensions taking into account symbolic shapes.
  auto mul_dims = [](int64_t dim1, int64_t dim2) -> int64 {
    return dim1 != 0 && dim2 != 0 && (dim1 < 0 || dim2 < 0) ? -1 : dim1 * dim2;
  };

  bool all_equal = true;
  size_t largest_rank = 0;
  output_batch_size_ = 1;
  for (int i = 0; i < N; ++i) {
    if (x[i] != x[0]) {
      all_equal = false;
    }
    if (x[i].size() > largest_rank) {
      largest_rank = x[i].size();
    }
  }
  if (all_equal) {
    broadcasting_required_ = false;
  }
  if (all_equal && TF_PREDICT_TRUE(fewer_dims_optimization)) {
    // Fast path for common case of identical shapes.
    int64_t elements = 1;
    const int rank = x[0].size();
    output_.resize(rank);
    for (int i = 0; i < rank; i++) {
      const int64_t dim = x[0][i];
      elements = mul_dims(elements, dim);
      output_[i] = dim;
    }
    result_.push_back(elements);
    output_batch_size_ = elements;
    for (int i = 0; i < N; ++i) {
      reshape_[i].push_back(elements);
      bcast_[i].push_back(1);
    }
    // grad_reduce_ is left as empty
    return;
  }

  // Reverse all the shapes for convenience
  // After the reverse, 0-th is the inner-most dimension.
  Vec copy[N];
  for (int i = 0; i < N; ++i) {
    copy[i] = x[i];
    Reverse(&copy[i]);
  }

  // 1-extend and align all vectors.
  for (int i = 0; i < N; ++i) {
    if (copy[i].size() < largest_rank) {
      copy[i].resize(largest_rank, 1);
    }
  }
  // Going through each dimension starting from the inner-most
  // dimension, compares dimension of x and y. They are compatible if
  // they are equal or either is 1.

  // indices of j-th component of each input.
  bool prev_is_one[N];
  bool current_is_one[N];
  for (int i = 0; i < N; ++i) {
    prev_is_one[i] = false;
    current_is_one[i] = false;
  }
  Vec output;
  bool output_dim_set = false;
  int output_dim = -1;
  bool none_is_one = true;
  bool set_one = false;
  for (int j = 0; j < largest_rank; ++j) {
    output_dim = -1;
    output_dim_set = false;
    none_is_one = true;
    // Find which indices are 1.
    for (int i = 0; i < N; ++i) {
      // Keep track of which indices are 1.
      if (copy[i][j] == 1) {
        current_is_one[i] = true;
        none_is_one = false;
      } else {
        current_is_one[i] = false;
        if (!output_dim_set || copy[i][j] == output_dim) {
          output_dim = copy[i][j];
          output_dim_set = true;
        } else {
          valid_ = false;
          return;
        }
      }
    }
    output_.push_back(output_dim_set ? output_dim : 1);
    output_batch_size_ = mul_dims(output_batch_size_, output_.back());
    // All dimensions are 1.
    if (!output_dim_set) {
      if (!TF_PREDICT_TRUE(fewer_dims_optimization)) {
        for (int i = 0; i < N; ++i) {
          bcast_[i].push_back(1);
          reshape_[i].push_back(1);
        }
        result_.push_back(1);
      }
      for (int i = 0; i < N; ++i) {
        grad_reduce_idx_[i].push_back(largest_rank - 1 - j);
      }
      // This will skip updating the previous state to the current one. We'll
      // explain why this is safe below.
      // Consider the previous state P, current state C and the next state N.
      // In the case where N also is all ones (N == C), we'll do the same
      // optimization here (push back one dimensions if we need to), which is
      // safe and is expected.
      //
      // When N != C, we'll continue as usual. However, we might trigger the
      // next block if N == P (because we didn't update the previous state).
      // We trigger the next block if `fewer_dims_optimization` is true.
      // This means that we did not modify and broadcast / reshapes in this
      // block (we skipped updating, since the one dimensions can be ignored).
      // In essence, we only need to check whether the previous non-one state is
      // equal to the current non-one state.

      continue;
    } else if (TF_PREDICT_TRUE(fewer_dims_optimization) &&
               std::equal(current_is_one, current_is_one + N, prev_is_one) &&
               set_one) {
      // It is a run of the same broadcasting case as last time.
      // We can reshape the input so that fewer dimensions
      // are involved in the intermediate computation.
      result_.back() = mul_dims(result_.back(), output_dim);
      for (int i = 0; i < N; ++i) {
        reshape_[i].back() = mul_dims(reshape_[i].back(), copy[i][j]);
        bcast_[i].back() =
            mul_dims(bcast_[i].back(), current_is_one[i] ? output_dim : 1);
        if (current_is_one[i] && !none_is_one) {
          grad_reduce_idx_[i].push_back(largest_rank - 1 - j);
        }
      }
    } else {
      result_.push_back(output_dim);
      for (int i = 0; i < N; ++i) {
        reshape_[i].push_back(copy[i][j]);
        bcast_[i].push_back(current_is_one[i] ? output_dim : 1);
        if (current_is_one[i] && !none_is_one) {
          grad_reduce_idx_[i].push_back(largest_rank - 1 - j);
        }
      }
    }
    set_one = true;
    for (int i = 0; i < N; ++i) {
      prev_is_one[i] = current_is_one[i];
    }
  }
  if (result_.empty()) {
    result_.push_back(1);
    for (int i = 0; i < N; ++i) {
      reshape_[i].push_back(1);
      bcast_[i].push_back(1);
    }
  }
  // Do something about batches.
  for (int i = 0; i < N; ++i) {
    Reverse(&reshape_[i]);
    Reverse(&bcast_[i]);
    Reverse(&grad_reduce_idx_[i]);
  }
  Reverse(&result_);
  Reverse(&output_);
  // Only compute batch indices when we need broadcasting, and we aren't doing
  // needless work (when the output size is 0 or the
  // return_flattened_batch_indices isn't enabled).
  if (return_flattened_batch_indices && broadcasting_required_ &&
      output_batch_size_ > 0) {
    for (int i = 0; i < N; ++i) {
      ComputeBatchIndices(output_batch_size_, reshape_[i], bcast_[i],
                          &batch_indices_[i]);
    }
  }
}

// BCast is a helper for broadcasting binary tensor operation.
// TensorFlow's broadcasting rule follows that of numpy (See
// http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
//
// The rule has the following properties:
//
//   1. suffix matching: the rule starts with the right-most
//      dimension, and works towards the left-most dimension. Since
//      TensorFlow is row-major, the right-most dimension (the last
//      element in the shape of a tensor) is the inner-most, a.k.a.
//      the fastest changing, dimension.
//
//   2. Two dimensions are compatible for broadcasting if both are the
//      same or either is 1.
//
// BCast takes the shape of two tensors and computes a few vectors of
// int32 that are useful for the caller to reshape the tensors, apply
// the right broadcasts to them, compute the broadcasted operation,
// and possibly the gradients. In a nutshell, the caller is expected
// to compute the broadcasted operation as following:
//
//   BCast b(x.shape(), y.shape());
//   output = x.reshape(b.x_reshape()).broadcast(b.x_bcast())
//            _op_
//            y.reshape(b.y_reshape()).broadcast(b.y_bcast())
//
// For the gradient computation,
//   grad_x = sum(grad * backprop_x(x, y), grad_x_reduce_idx)
//            .reshape(x.shape())
//   grad_y = sum(grad * backprop_y(x, y), grad_y_reduce_idx)
//            .reshape(y.shape())
// backprop_x and backprop_y are functionals of the binary function "op",
// e.g.,
//   for +, backprop_x(x, y) = backprop_y(x, y) = 1;
//   for *, backprop_x(x, y) =  y, backprop_y(x, y) = x;
//   for /, backprop_x(x, y) = 1/y, backprop_y(x, y) = -x/y^2;
//
// The multiplication in the grad * backprop_x itself is also
// broadcasting following the same rule.
class BCast : public BCastList<2> {
 public:
  // Constructs all helper shapes, following the aforementioned rules.
  //
  // If "fewer_dims_optimization" is set to true (the default), the
  // implementation tries to reduce intermediate dimensions needed to be more
  // efficient.  This is transparent to the caller.
  //
  // If false, all intermediate shapes (except for grad_{x,y}_reduce_idx()) have
  // the same number of dimensions as the larger of the two inputs.
  typedef gtl::InlinedVector<int64_t, 4> Vec;

  BCast(const Vec& x, const Vec& y, const bool fewer_dims_optimization = true,
        const bool return_flattened_batch_indices = false)
      : BCastList<2>({x, y}, fewer_dims_optimization,
                     return_flattened_batch_indices) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_13(mht_13_v, 575, "", "./tensorflow/core/util/bcast.h", "BCast");
}

  ~BCast() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_14(mht_14_v, 580, "", "./tensorflow/core/util/bcast.h", "~BCast");
}

  // If and only if IsValid(), the following fields can be used in
  // implementing a broadcasted binary tensor operation according to
  // the broadcasting rule.
  const Vec& x_reshape() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_15(mht_15_v, 588, "", "./tensorflow/core/util/bcast.h", "x_reshape");
 return reshape_[0]; }
  const Vec& x_bcast() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_16(mht_16_v, 592, "", "./tensorflow/core/util/bcast.h", "x_bcast");
 return bcast_[0]; }
  const Vec& y_reshape() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_17(mht_17_v, 596, "", "./tensorflow/core/util/bcast.h", "y_reshape");
 return reshape_[1]; }
  const Vec& y_bcast() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_18(mht_18_v, 600, "", "./tensorflow/core/util/bcast.h", "y_bcast");
 return bcast_[1]; }
  const Vec& result_shape() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_19(mht_19_v, 604, "", "./tensorflow/core/util/bcast.h", "result_shape");
 return result_; }
  const Vec& output_shape() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_20(mht_20_v, 608, "", "./tensorflow/core/util/bcast.h", "output_shape");
 return output_; }
  const Vec& grad_x_reduce_idx() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_21(mht_21_v, 612, "", "./tensorflow/core/util/bcast.h", "grad_x_reduce_idx");
 return grad_reduce_idx_[0]; }
  const Vec& grad_y_reduce_idx() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_22(mht_22_v, 616, "", "./tensorflow/core/util/bcast.h", "grad_y_reduce_idx");
 return grad_reduce_idx_[1]; }

  // Returns the mapping from the flattened output batch indices to x's
  // flattened batch indices. The result is a vector of length
  // output_batch_size(). To compute the i'th batch output, a binary matmul-like
  // operation should use the `x_batch_indices()[i]`th batch index of `x`.
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64_t>& x_batch_indices() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_23(mht_23_v, 627, "", "./tensorflow/core/util/bcast.h", "x_batch_indices");

    return batch_indices_[0];
  }
  // Returns the mapping from the flattened output batch indices to y's
  // flattened batch indices. Similar to x_batch_indices().
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64_t>& y_batch_indices() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSutilPSbcastDTh mht_24(mht_24_v, 637, "", "./tensorflow/core/util/bcast.h", "y_batch_indices");

    return batch_indices_[1];
  }

  template <typename IndexType, int NDIMS>
  static Eigen::array<IndexType, NDIMS> ToIndexArrayType(
      const BCast::Vec& vec) {
    CHECK_EQ(vec.size(), NDIMS);
    Eigen::array<IndexType, NDIMS> ret;
    for (int i = 0; i < NDIMS; ++i) ret[i] = vec[i];
    return ret;
  }

  template <int NDIMS>
  static Eigen::array<Eigen::DenseIndex, NDIMS> ToIndexArray(
      const BCast::Vec& vec) {
    return ToIndexArrayType<Eigen::DenseIndex, NDIMS>(vec);
  }

  // Static helpers.
  static Vec FromShape(const TensorShape& shape);
  static TensorShape ToShape(const Vec& vec);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BCast);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_BCAST_H_
