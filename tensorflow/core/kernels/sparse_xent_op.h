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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_XENT_OP_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_XENT_OP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTh() {
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

// Functor definition for SparseXentOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace sparse_xent_helpers {

template <typename T>
typename TTypes<const T, 1>::Tensor32Bit To32BitConst(
    typename TTypes<T>::Vec in) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTh mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/sparse_xent_op.h", "To32BitConst");

  return To32Bit(typename TTypes<T>::ConstVec(in.data(), in.dimensions()));
}

template <typename T>
typename TTypes<const T, 2>::Tensor32Bit To32BitConst(
    typename TTypes<T>::Matrix in) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTh mht_1(mht_1_v, 211, "", "./tensorflow/core/kernels/sparse_xent_op.h", "To32BitConst");

  return To32Bit(typename TTypes<T>::ConstMatrix(in.data(), in.dimensions()));
}

}  // namespace sparse_xent_helpers

namespace generator {

// Generator for calculation of the sparse Xent loss.
// This generator takes the logits, the sum of the exponentiated
// logits, and the label indices.  For each minibatch entry, ignoring
// the batch index b, it calculates:
//
//   loss[j] = (log(sum_exp_logits) - logits[j]) * 1{ j == label }
//
// for j = 0 .. num_classes.  This value must be summed over all j for
// the final loss.
template <typename T, typename Index>
class SparseXentLossGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE SparseXentLossGenerator(
      typename TTypes<const T, 2>::Tensor32Bit logits,
      typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits,
      typename TTypes<const Index, 1>::Tensor32Bit labels,
      const Index max_depth)
      : logits_(logits),
        sum_exp_logits_(sum_exp_logits),
        labels_(labels),
        max_depth_(max_depth) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTh mht_2(mht_2_v, 242, "", "./tensorflow/core/kernels/sparse_xent_op.h", "SparseXentLossGenerator");
}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<int, 2>& coords) const {
    const int batch = coords[0];
    const int depth = coords[1];
    const Index label = tensorflow::internal::SubtleMustCopy(labels_(batch));
    if (!FastBoundsCheck(label, max_depth_)) {
      return Eigen::NumTraits<T>::quiet_NaN();
    }
    return TF_PREDICT_FALSE(label == depth)
               ? (Eigen::numext::log(sum_exp_logits_(batch)) - logits_(coords))
               : T(0.0);
  };

 private:
  typename TTypes<const T, 2>::Tensor32Bit logits_;
  typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits_;
  typename TTypes<const Index, 1>::Tensor32Bit labels_;
  const Index max_depth_;
};

// Generator for calculation of the sparse Xent gradient.
// This generator takes the exponentiated logits, their sums, and the label
// indices. For each minibatch entry, ignoring the batch index b, it calculates:
//
//   exp_logits[j] / sum_exp_logits - 1{ j == label }
//
// for j = 0 .. num_classes.
template <typename T, typename Index>
class SparseXentGradGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE SparseXentGradGenerator(
      typename TTypes<const T, 2>::Tensor32Bit exp_logits,
      typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits,
      typename TTypes<const Index, 1>::Tensor32Bit labels,
      const Index max_depth)
      : exp_logits_(exp_logits),
        sum_exp_logits_(sum_exp_logits),
        labels_(labels),
        max_depth_(max_depth) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTh mht_3(mht_3_v, 285, "", "./tensorflow/core/kernels/sparse_xent_op.h", "SparseXentGradGenerator");
}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<int, 2>& coords) const {
    const int batch = coords[0];
    const int depth = coords[1];
    const Index label = tensorflow::internal::SubtleMustCopy(labels_(batch));
    if (!FastBoundsCheck(label, max_depth_)) {
      return Eigen::NumTraits<T>::quiet_NaN();
    }
    T subtract = TF_PREDICT_FALSE(depth == label) ? T(1.0) : T(0.0);
    return exp_logits_(coords) / sum_exp_logits_(batch) - subtract;
  };

 private:
  typename TTypes<const T, 2>::Tensor32Bit exp_logits_;
  typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits_;
  typename TTypes<const Index, 1>::Tensor32Bit labels_;
  const Index max_depth_;
};

}  // namespace generator

namespace functor {

template <typename Device, typename T>
struct RowMaxReduction {
  // Computes the maximum across the rows of logits
  //
  // logits: batch_size, num_classes.
  // maximum: temporary tensor, dims: batch_size, 1
  static inline void Compute(OpKernelContext* ctx,
                             typename TTypes<T>::ConstMatrix logits,
                             typename TTypes<T>::Vec maximum) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTh mht_4(mht_4_v, 321, "", "./tensorflow/core/kernels/sparse_xent_op.h", "Compute");

    Eigen::IndexList<Eigen::type2index<1> > along_row;
    Device d = ctx->eigen_device<Device>();
    To32Bit(maximum).device(d) = To32Bit(logits).maximum(along_row);
  }
};

// Functor used by SparseXentOp to do the computations.
template <typename Device, typename T, typename Index>
struct SparseXentFunctor {
  // Computes Cross Entropy loss and backprop.
  //
  // logits: batch_size, num_classes.
  // labels: num_classes.
  // scratch: temporary tensor, dims: batch_size, 1
  // loss: output tensor for the loss, dims: batch_size.
  // backprop: output tensor for the backprop, dims: batch_size, num_classes.
  void operator()(OpKernelContext* ctx, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::Vec scratch, typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop);
};

// Eigen code implementing SparseXentFunctor::operator().
// This code works for both CPU and GPU and is used by the functor
// specializations for both device types.
template <typename Device, typename T, typename Index>
struct SparseXentEigenImpl {
  static void Compute(OpKernelContext* ctx,
                      typename TTypes<T>::ConstMatrix logits,
                      typename TTypes<Index>::ConstVec labels,
                      typename TTypes<T>::Vec scratch,
                      typename TTypes<T>::Vec loss,
                      typename TTypes<T>::Matrix backprop) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTh mht_5(mht_5_v, 357, "", "./tensorflow/core/kernels/sparse_xent_op.h", "Compute");

    // NOTE(touts): This duplicates some of the computations in softmax_op
    // because we need the intermediate (logits -max(logits)) values to
    // avoid a log(exp()) in the computation of the loss.

    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);

// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
    Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
    Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
    batch_by_one.set(0, batch_size);
    Eigen::IndexList<int> batch_only;
    batch_only.set(0, batch_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
    one_by_class.set(1, num_classes);

    // scratch = max_logits along classes.
    RowMaxReduction<Device, T>::Compute(ctx, logits, scratch);

    Device d = ctx->eigen_device<Device>();
    // backprop = logits - max_logits.
    To32Bit(backprop).device(d) =
        To32Bit(logits) -
        To32Bit(scratch).reshape(batch_by_one).broadcast(one_by_class);

    // scratch = sum(exp(logits - max_logits)) along classes.
    To32Bit(scratch).device(d) = To32Bit(backprop).exp().sum(along_class);

    //  sum(-labels *
    //     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
    //  along classes
    generator::SparseXentLossGenerator<T, Index> sparse_xent_loss_gen(
        sparse_xent_helpers::To32BitConst<T>(backprop),
        sparse_xent_helpers::To32BitConst<T>(scratch), To32Bit(labels),
        backprop.dimension(1) /* max_depth */);
    To32Bit(loss).device(d) =
        To32Bit(backprop).generate(sparse_xent_loss_gen).sum(along_class);

    // backprop: prob - labels, where
    //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
    To32Bit(backprop).device(d) = To32Bit(backprop).exp();
    generator::SparseXentGradGenerator<T, Index> sparse_xent_grad_gen(
        sparse_xent_helpers::To32BitConst<T>(backprop),
        sparse_xent_helpers::To32BitConst<T>(scratch), To32Bit(labels),
        backprop.dimension(1) /* max_depth */);
    To32Bit(backprop).device(d) =
        To32Bit(backprop).generate(sparse_xent_grad_gen);
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_XENT_OP_H_
