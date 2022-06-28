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
class MHTracer_DTPStensorflowPScorePSkernelsPStopk_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStopk_opDTcc() {
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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/topk_op.h"

#include <algorithm>
#include <numeric>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class TopK : public OpKernel {
 public:
  explicit TopK(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_opDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/topk_op.cc", "TopK");

    OP_REQUIRES_OK(context, context->GetAttr("sorted", &sorted_));
    if (num_inputs() < 2) {  // k is an attr (TopK).
      OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
    } else {  // k is an input (TopKV2), so we won't know it until Compute.
      k_ = -1;
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_opDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/kernels/topk_op.cc", "Compute");

    int k = k_;
    if (num_inputs() >= 2) {
      const auto& k_in = context->input(1);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(k_in.shape()),
                  errors::InvalidArgument("k must be scalar, got shape ",
                                          k_in.shape().DebugString()));
      k = k_in.scalar<int32>()();
    }
    OP_REQUIRES(context, k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", k));
    const auto& input_in = context->input(0);
    OP_REQUIRES(context, input_in.dims() >= 1,
                errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input_in.shape().DebugString()));
    OP_REQUIRES(context, input_in.dim_size(input_in.dims() - 1) >= k,
                errors::InvalidArgument(
                    "input must have at least k columns. Had ",
                    input_in.dim_size(input_in.dims() - 1), ", needed ", k));

    const auto& input = input_in.flat_inner_dims<T>();

    const int64_t num_rows = input.dimension(0);  // generally batch_size
    const int64_t num_cols = input.dimension(1);
    OP_REQUIRES(
        context, num_rows <= std::numeric_limits<int32>::max(),
        errors::InvalidArgument(
            "First dimension of flattened input must be <= INT_MAX, got ",
            num_rows));
    OP_REQUIRES(
        context, num_cols <= std::numeric_limits<int32>::max(),
        errors::InvalidArgument(
            "Second dimension of flattened input must be <= INT_MAX, got ",
            num_cols));

    TensorShape output_shape = input_in.shape();
    output_shape.set_dim(input_in.dims() - 1, k);
    Tensor* values_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &values_out));
    Tensor* indices_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &indices_out));

    // Nothing to do for top-nothing or over nothing.
    if (k == 0 || num_rows == 0) return;

    auto values = values_out->flat_inner_dims<T>();
    auto indices = indices_out->flat_inner_dims<int32>();
    Status s = functor::TopKFunctor<Device, T>::Compute(
        context, sorted_, k, input, num_rows, num_cols, values, indices);
    OP_REQUIRES_OK(context, s);
  }

 private:
  int k_;
  bool sorted_;
};

namespace functor {

template <typename T>
struct TopKFunctor<CPUDevice, T> {
  static EIGEN_ALWAYS_INLINE Status Compute(
      OpKernelContext* context, bool sorted, int k,
      const typename TTypes<T, 2>::ConstTensor& input, const int64_t num_rows,
      const int64_t num_cols, typename TTypes<T, 2>::Tensor values,
      typename TTypes<int, 2>::Tensor indices) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_opDTcc mht_2(mht_2_v, 293, "", "./tensorflow/core/kernels/topk_op.cc", "Compute");

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    // Special case for k == 1.
    if (k == 1) {
      typename Eigen::IndexList<Eigen::type2index<1>> reduce_on_cols;
      typename Eigen::IndexList<int, Eigen::type2index<1>> rows_by_one;
      rows_by_one.set(0, num_rows);

      values.device(d) =
          input.maximum(/*dims=*/reduce_on_cols).eval().reshape(rows_by_one);
      // Get the indices of the maximum values.
      for (int r = 0; r < num_rows; ++r) {
        indices(r, 0) = 0;
        for (int c = 0; c < num_cols; ++c) {
          if (values(r, 0) == input(r, c)) {
            indices(r, 0) = c;
            break;
          }
        }
        values(r, 0) = input(r, indices(r, 0));
      }

      return Status::OK();
    }

    auto SortIndices = [&](int64_t start_batch, int64_t limit_batch) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_opDTcc mht_3(mht_3_v, 322, "", "./tensorflow/core/kernels/topk_op.cc", "lambda");

      for (int32_t b = start_batch; b < limit_batch; ++b) {
        const T* input_data = &input(b, 0);
        const auto stable_comp = [input_data](const int32_t a,
                                              const int32_t b) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_opDTcc mht_4(mht_4_v, 329, "", "./tensorflow/core/kernels/topk_op.cc", "lambda");

          if (input_data[b] < input_data[a]) {
            return true;
          } else if (input_data[b] > input_data[a]) {
            return false;
          } else {
            return a < b;
          }
        };
        const auto comp = [input_data](const int32_t a, const int32_t b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_opDTcc mht_5(mht_5_v, 341, "", "./tensorflow/core/kernels/topk_op.cc", "lambda");

          return input_data[b] < input_data[a];
        };
        // TODO(ebrevdo): For large k < num_cols, instead of using
        // TopN, it may be faster to create a temporary vector of
        // values 0..num_cols - 1 and then use std::partial_sort_copy
        // of this into indices. Choosing the appropriate minimum k or
        // ratio of k/num_cols will require some experimentation.
        if (k == num_cols) {
          auto* begin = &indices(b, 0);
          auto* end = &indices(b, k);
          // Set the initial array of indices 0 ... k - 1.
          std::iota(begin, end, 0);
          // We want an in-place sort, but we can cheat because we're sorting
          // indices that started out sorted.  First, do a std::sort, which
          // is notably faster than std::stable_sort.
          std::sort(begin, end, comp);
          // Then, for runs of adjacent elements that were equal, sort the
          // indices in those runs in increasing order.
          for (auto* run_begin = begin; run_begin != end;) {
            auto* run_end = run_begin + 1;
            if (run_end == end) break;
            if (input_data[*run_begin] == input_data[*run_end]) {
              while (++run_end != end) {
                if (input_data[*run_begin] != input_data[*run_end]) break;
              }
              std::sort(run_begin, run_end);
            }
            run_begin = run_end;
          }
        } else {
          // Use the TopN heap object to sort.
          gtl::TopN<int32, decltype(stable_comp)> filter(k, stable_comp);
          filter.reserve(num_cols);
          for (int32_t c = 0; c < num_cols; ++c) {
            filter.push(c);
          }

          int32_t i = 0;
          if (sorted) {
            std::unique_ptr<std::vector<int32>> top_k(filter.Extract());
            for (auto top_k_it = top_k->begin(); top_k_it != top_k->end();
                 ++top_k_it, ++i) {
              indices(b, i) = *top_k_it;
            }
          } else {
            for (auto top_k_it = filter.unsorted_begin();
                 top_k_it != filter.unsorted_end(); ++top_k_it, ++i) {
              indices(b, i) = *top_k_it;
            }
          }
        }
        // Now that the indices are sorted, copy the values over in
        // sorted order.
        std::transform(
            &indices(b, 0), &indices(b, k), &values(b, 0),
            [b, &input](const int32_t loc) { return input(b, loc); });
      }  // for (int32 b = ...
    };

    // Guesstimate of cost; 4*N*log(K) where N == num_cols.
    // If K == N, assume the cost is N*log(K + 1).
    const double cmp_cost = 3 * Eigen::TensorOpCost::AddCost<int32>() +
                            Eigen::TensorOpCost::AddCost<T>();
    const double base_cost =
        cmp_cost *
        static_cast<double>(num_cols *
                            Eigen::numext::log2(static_cast<float>(k + 1)));
    const double sort_cost = (k == num_cols) ? base_cost : 4 * base_cost;
    const double copy_cost = 2 * k * Eigen::TensorOpCost::AddCost<T>();
    const double total_cost = sort_cost + copy_cost;
    const int64_t final_cost = (total_cost >= static_cast<double>(kint64max))
                                   ? kint64max
                                   : static_cast<int64_t>(total_cost);
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, num_rows,
          final_cost, SortIndices);

    return Status::OK();
  }
};

}  // namespace functor

#define REGISTER_KERNELS_NAME(name, type)                       \
  REGISTER_KERNEL_BUILDER(                                      \
      Name(#name).Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TopK<CPUDevice, type>)

#define REGISTER_KERNELS(type)       \
  REGISTER_KERNELS_NAME(TopK, type); \
  REGISTER_KERNELS_NAME(TopKV2, type)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS_NAME
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  Status TopKFunctor<GPUDevice, T>::Compute(                                   \
      OpKernelContext* context, bool sorted, int k,                            \
      const typename TTypes<T, 2>::ConstTensor& input, const int64_t num_rows, \
      const int64_t num_cols, typename TTypes<T, 2>::Tensor values,            \
      typename TTypes<int, 2>::Tensor indices);                                \
  extern template struct functor::TopKFunctor<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
TF_CALL_INTEGRAL_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

}  // namespace functor

#define REGISTER_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TopK").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      TopK<GPUDevice, type>)                                     \
  REGISTER_KERNEL_BUILDER(Name("TopKV2")                         \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("k"),                  \
                          TopK<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS);
TF_CALL_INTEGRAL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // end GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow
