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

// This is an internal header file intended to only be included as the
// front-matter in the implementation files of various reduction ops.  It
// is a header file because we split the various reduction ops into their
// own compilation units to get more parallelism in compilation.

#ifndef TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_COMMON_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh() {
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


#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/reduction_ops.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
struct Constants {
  // Derive Index type. int (32-bit) or long (64-bit) depending on the
  // compile-time configuration. "float" here is not relevant.
  // TODO(zhifengc): Moves the definition to TTypes.
  typedef TTypes<float>::Tensor::Index Index;
  Eigen::array<Index, 1> kZero;
  Eigen::array<Index, 1> kOne;
  Eigen::array<Index, 2> kZeroTwo;

  Constants() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh mht_0(mht_0_v, 224, "", "./tensorflow/core/kernels/reduction_ops_common.h", "Constants");

    kZero[0] = 0;
    kOne[0] = 1;
    kZeroTwo[0] = 0;
    kZeroTwo[1] = 2;
  }
};

struct ConstantsBase {
  const Eigen::IndexList<Eigen::type2index<0>> kZero;
  const Eigen::IndexList<Eigen::type2index<1>> kOne;
  const Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<2>> kZeroTwo;
};
template <>
struct Constants<CPUDevice> : ConstantsBase {};

class ReductionHelper {
 public:
  ReductionHelper() : reduce_first_axis_(false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh mht_1(mht_1_v, 245, "", "./tensorflow/core/kernels/reduction_ops_common.h", "ReductionHelper");
}

  Status Simplify(const Tensor& data, const Tensor& axis, const bool keep_dims);

  // We need to do roughly:
  //   tmp_out = allocate(out_reshape())
  //   tmp_out.reshape(out_reshape) = data.reshape(data_reshape).reduce(axes)
  //   out = tmp_out.reshape(out_shape)

  // The reduction result must be allocated with this shape.
  TensorShape out_reshape() const;

  // The final output shape must be allocated with this shape.
  TensorShape out_shape() const;

  // The reduction is on a reshaped tensor of this rank.
  int ndims() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh mht_2(mht_2_v, 264, "", "./tensorflow/core/kernels/reduction_ops_common.h", "ndims");
 return data_reshape_.size(); }

  // True if need to reduce the 0-th dimension.
  bool reduce_first_axis() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh mht_3(mht_3_v, 270, "", "./tensorflow/core/kernels/reduction_ops_common.h", "reduce_first_axis");
 return reduce_first_axis_; }

  // The output is reshaped.
  template <typename T, int N>
  typename TTypes<T, N>::Tensor out(Tensor* out) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh mht_4(mht_4_v, 277, "", "./tensorflow/core/kernels/reduction_ops_common.h", "out");

    return out->shaped<T, N>(out_reshape_);
  }

  // The input is reshaped.
  template <typename T, int N>
  typename TTypes<T, N>::ConstTensor in(const Tensor& data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh mht_5(mht_5_v, 286, "", "./tensorflow/core/kernels/reduction_ops_common.h", "in");

    return data.shaped<T, N>(data_reshape_);
  }

  // Shape of shuffled input
  TensorShape data_reshape() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh mht_6(mht_6_v, 294, "", "./tensorflow/core/kernels/reduction_ops_common.h", "data_reshape");

    TensorShape shape;
    for (auto s : data_reshape_) shape.AddDim(s);
    return shape;
  }

  // Shape with all reduction dimensions at the end
  TensorShape shuffled_shape();

  // Permutation of reduced dims needed to put reduction dimensions at the end
  gtl::InlinedVector<int32, 8> permutation();

 private:
  bool reduce_first_axis_;  // True if need to reduce the 0-th dimension.
  gtl::InlinedVector<int64_t, 4>
      data_reshape_;                          // Reshape data before reduction.
  gtl::InlinedVector<int64_t, 4> out_shape_;  // The final output shape.
  gtl::InlinedVector<int64_t, 4> out_reshape_;  // Reshape output for reduction.
};

// For operations where the output is a reduction function along some
// dimensions of the input.
template <typename Device, class T, typename Tperm, typename Reducer>
class ReductionOp : public OpKernel {
 public:
  explicit ReductionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh mht_7(mht_7_v, 322, "", "./tensorflow/core/kernels/reduction_ops_common.h", "ReductionOp");

    const DataType dt = DataTypeToEnum<T>::v();
    const DataType pt = DataTypeToEnum<Tperm>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({dt, pt}, {dt}));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh mht_8(mht_8_v, 333, "", "./tensorflow/core/kernels/reduction_ops_common.h", "Compute");

    const Tensor& data = ctx->input(0);
    const Tensor& axes = ctx->input(1);
    VLOG(1) << "data shape: " << data.shape().DebugString();
    VLOG(1) << "axes      : " << axes.SummarizeValue(10);

    ReductionHelper helper;
    OP_REQUIRES_OK(ctx, helper.Simplify(data, axes, keep_dims_));
    CHECK_GE(helper.ndims(), 0);

    bool is_scalar_identity = functor::ReducerTraits<Reducer>::IsScalarIdentity;
    bool is_trivial = helper.ndims() == 0 ||
                      (helper.ndims() == 1 && !helper.reduce_first_axis());
    if (is_scalar_identity && is_trivial) {
      Tensor out;
      // Special case. Reduces nothing and does not alter the input values.
      if (!out.CopyFrom(data, helper.out_shape())) {
        ctx->SetStatus(errors::Internal("Error during reduction copy."));
      }
      ctx->set_output(0, out);
      return;
    }

    // We must allocate temp tensors using the same alloc attr as
    // output(0) because it is returned as output(0) in the end.
    const AllocatorAttributes alloc_attr = ctx->output_alloc_attr(0);

    Tensor tmp_out;
    typedef functor::ReduceFunctor<Device, Reducer> Functor;
    Constants<Device> constants;
    const Device& d = ctx->eigen_device<Device>();
    Reducer reducer;

    if (data.NumElements() > 0 && is_trivial && !is_scalar_identity) {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(ctx->expected_output_dtype(0),
                                             TensorShape({data.NumElements()}),
                                             &tmp_out, alloc_attr));
      Functor::Reduce(ctx, tmp_out.flat<T>(),
                      data.shaped<T, 2>({1, data.NumElements()}),
                      constants.kZero, reducer);
    } else {
      // A temporary tensor whose size matches the size of the reduced
      // output.
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(ctx->expected_output_dtype(0),
                                  helper.out_reshape(), &tmp_out, alloc_attr));

      if (tmp_out.NumElements() == 0) {
        // Nothing to do, fall through to final reshaping.
      } else if (data.NumElements() == 0) {
        // Degenerate reduction where the input is empty but the output is
        // nonempty (thus tmp_out.NumElements() > 0), and we must fill the
        // output with identity elements.  Example: tf.reduce_sum(tf.zeros((0,
        // 3)), [0]). Eigen sometimes crashes in this case, so we do it
        // manually.
        Functor::FillIdentity(d, tmp_out.flat<T>(), reducer);
      } else if ((helper.ndims() == 1) && helper.reduce_first_axis()) {
        // Reduce to a scalar.
        Functor::Reduce(ctx, helper.out<T, 0>(&tmp_out), helper.in<T, 1>(data),
                        constants.kZero, reducer);
      } else if ((helper.ndims() == 2) && helper.reduce_first_axis()) {
        // Can be viewed as a reduction of a matrix along 1st dimension.
        Functor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
                        constants.kZero, reducer);
      } else if ((helper.ndims() == 2) && !helper.reduce_first_axis()) {
        // Can be viewed as a reduction of a matrix along 2nd dimension.
        Functor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
                        constants.kOne, reducer);
      } else if ((helper.ndims() == 3) && helper.reduce_first_axis()) {
        // Can be viewed as a reduction of a 3D tensor along 1st and 3rd
        // dimensions.
        Functor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 3>(data),
                        constants.kZeroTwo, reducer);
      } else if ((helper.ndims() == 3) && !helper.reduce_first_axis()) {
        // Can be viewed as a reduction of a 3D tensor along 2nd dimension.
        Functor::Reduce(ctx, helper.out<T, 2>(&tmp_out), helper.in<T, 3>(data),
                        constants.kOne, reducer);
      } else {
        // If we don't hit one of the cases above, transpose the data so that
        // all reduced dimensions are last and reuse the 2-D -> 1-D case.
        Tensor data_reshaped;
        OP_REQUIRES(ctx, data_reshaped.CopyFrom(data, helper.data_reshape()),
                    errors::Internal("Error during reduction copy."));
        Tensor shuffled;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                               helper.shuffled_shape(),
                                               &shuffled, alloc_attr));
        OP_REQUIRES_OK(ctx, DoTranspose(d, data_reshaped, helper.permutation(),
                                        &shuffled));
        const int64_t unreduced = tmp_out.NumElements();
        const int64_t reduced = shuffled.NumElements() / unreduced;
        const Tensor& const_shuffled = shuffled;
        Functor::Reduce(ctx, tmp_out.flat<T>(),
                        const_shuffled.shaped<T, 2>({unreduced, reduced}),
                        constants.kOne, reducer);
      }
    }

    // Set the real output using the contents of the reduction but the
    // real expected output shape.  The number of elements should
    // match between the two shapes.
    Tensor out;
    OP_REQUIRES(ctx, out.CopyFrom(tmp_out, helper.out_shape()),
                errors::Internal("Error during reduction copy."));
    ctx->set_output(0, out);
  }

 private:
  // True if the number of dimensions should be maintained.
  bool keep_dims_;
};

namespace functor {

template <typename Device, typename Reducer>
struct ReduceFunctorBase {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer) {
    const Device& d = ctx->eigen_device<Device>();
    ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes, Reducer> reducer_impl;
    reducer_impl(d, out, in, reduction_axes, reducer);
  }

  template <typename OUT_T>
  static void FillIdentity(const Device& d, OUT_T out, const Reducer& reducer) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTh mht_9(mht_9_v, 462, "", "./tensorflow/core/kernels/reduction_ops_common.h", "FillIdentity");

    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <typename Reducer>
struct ReduceFunctor<CPUDevice, Reducer>
    : ReduceFunctorBase<CPUDevice, Reducer> {};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_REDUCTION_OPS_COMMON_H_
