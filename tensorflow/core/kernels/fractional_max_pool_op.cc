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
class MHTracer_DTPStensorflowPScorePSkernelsPSfractional_max_pool_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfractional_max_pool_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfractional_max_pool_opDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#define EIGEN_USE_THREADS

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "tensorflow/core/kernels/fractional_pool_common.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
class FractionalMaxPoolOp : public OpKernel {
 public:
  explicit FractionalMaxPoolOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfractional_max_pool_opDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/fractional_max_pool_op.cc", "FractionalMaxPoolOp");

    OP_REQUIRES_OK(context, context->GetAttr("pooling_ratio", &pooling_ratio_));
    OP_REQUIRES_OK(context, context->GetAttr("pseudo_random", &pseudo_random_));
    OP_REQUIRES_OK(context, context->GetAttr("overlapping", &overlapping_));

    OP_REQUIRES(context, pooling_ratio_.size() == 4,
                errors::InvalidArgument("pooling_ratio field must "
                                        "specify 4 dimensions"));

    OP_REQUIRES(
        context, pooling_ratio_[0] == 1 || pooling_ratio_[3] == 1,
        errors::Unimplemented("Fractional max pooling is not yet "
                              "supported on the batch nor channel dimension."));

    OP_REQUIRES_OK(context, context->GetAttr("deterministic", &deterministic_));
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(context, context->GetAttr("seed2", &seed2_));
    if (deterministic_) {
      // If both seeds are not set when deterministic_ is true, force set seeds.
      if ((seed_ == 0) && (seed2_ == 0)) {
        seed_ = random::New64();
        seed2_ = random::New64();
      }
    } else {
      OP_REQUIRES(
          context, (seed_ == 0) && (seed2_ == 0),
          errors::InvalidArgument(
              "Both seed and seed2 should be 0 if deterministic is false."));
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfractional_max_pool_opDTcc mht_1(mht_1_v, 242, "", "./tensorflow/core/kernels/fractional_max_pool_op.cc", "Compute");

    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        ConstEigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        EigenMatrixMap;

    constexpr int tensor_in_and_out_dims = 4;

    const Tensor& tensor_in = context->input(0);
    OP_REQUIRES(context, tensor_in.dims() == tensor_in_and_out_dims,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));

    std::vector<int> input_size(tensor_in_and_out_dims);
    std::vector<int> output_size(tensor_in_and_out_dims);
    for (int i = 0; i < tensor_in_and_out_dims; ++i) {
      input_size[i] = tensor_in.dim_size(i);

      OP_REQUIRES(
          context, input_size[i] >= pooling_ratio_[i],
          errors::InvalidArgument("Pooling ratio is higher than input "
                                  "dimension size for dimension ",
                                  i, ". Input dim size: ", input_size[i],
                                  " pooling ratio: ", pooling_ratio_[i]));
    }
    // Output size.
    for (int i = 0; i < tensor_in_and_out_dims; ++i) {
      // This must match the same logic in the shape function in
      // core/ops/nn_ops.cc.
      output_size[i] =
          static_cast<int>(std::floor(input_size[i] / pooling_ratio_[i]));
      DCHECK_GT(output_size[i], 0);
    }

    // Generate pooling sequence.
    std::vector<int64_t> height_cum_seq;
    std::vector<int64_t> width_cum_seq;
    GuardedPhiloxRandom generator;
    generator.Init(seed_, seed2_);
    height_cum_seq = GeneratePoolingSequence(input_size[1], output_size[1],
                                             &generator, pseudo_random_);
    width_cum_seq = GeneratePoolingSequence(input_size[2], output_size[2],
                                            &generator, pseudo_random_);

    // Prepare output.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0,
                                TensorShape({output_size[0], output_size[1],
                                             output_size[2], output_size[3]}),
                                &output_tensor));
    Tensor* output_height_seq_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            1, TensorShape({static_cast<int64_t>(height_cum_seq.size())}),
            &output_height_seq_tensor));
    Tensor* output_width_seq_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            2, TensorShape({static_cast<int64_t>(width_cum_seq.size())}),
            &output_width_seq_tensor));

    ConstEigenMatrixMap in_mat(tensor_in.flat<T>().data(), input_size[3],
                               input_size[2] * input_size[1] * input_size[0]);

    EigenMatrixMap out_mat(output_tensor->flat<T>().data(), output_size[3],
                           output_size[2] * output_size[1] * output_size[0]);

    // Initializes the output tensor with MIN<T>.
    output_tensor->flat<T>().setConstant(Eigen::NumTraits<T>::lowest());

    auto output_height_seq_flat = output_height_seq_tensor->flat<int64_t>();
    auto output_width_seq_flat = output_width_seq_tensor->flat<int64_t>();

    // Set output tensors.
    for (int i = 0; i < height_cum_seq.size(); ++i) {
      output_height_seq_flat(i) = height_cum_seq[i];
    }

    for (int i = 0; i < width_cum_seq.size(); ++i) {
      output_width_seq_flat(i) = width_cum_seq[i];
    }

    // For both input and output,
    // 0: batch
    // 1: height / row
    // 2: width / col
    // 3: depth / channel
    const int64_t height_max = input_size[1] - 1;
    const int64_t width_max = input_size[2] - 1;
    for (int64_t b = 0; b < input_size[0]; ++b) {
      // height sequence.
      for (int64_t hs = 0; hs < height_cum_seq.size() - 1; ++hs) {
        // height start and end.
        const int64_t height_start = height_cum_seq[hs];
        int64_t height_end =
            overlapping_ ? height_cum_seq[hs + 1] : height_cum_seq[hs + 1] - 1;
        height_end = std::min(height_end, height_max);

        // width sequence.
        for (int64_t ws = 0; ws < width_cum_seq.size() - 1; ++ws) {
          const int64_t out_offset =
              (b * output_size[1] + hs) * output_size[2] + ws;
          // width start and end.
          const int64_t width_start = width_cum_seq[ws];
          int64_t width_end =
              overlapping_ ? width_cum_seq[ws + 1] : width_cum_seq[ws + 1] - 1;
          width_end = std::min(width_end, width_max);
          for (int64_t h = height_start; h <= height_end; ++h) {
            for (int64_t w = width_start; w <= width_end; ++w) {
              const int64_t in_offset =
                  (b * input_size[1] + h) * input_size[2] + w;
              out_mat.col(out_offset) =
                  out_mat.col(out_offset).cwiseMax(in_mat.col(in_offset));
            }
          }
        }
      }
    }
  }

 private:
  bool deterministic_;
  int64_t seed_;
  int64_t seed2_;
  std::vector<float> pooling_ratio_;
  bool pseudo_random_;
  bool overlapping_;
};

#define REGISTER_FRACTIONALMAXPOOL(type)                                      \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("FractionalMaxPool").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      FractionalMaxPoolOp<type>)

REGISTER_FRACTIONALMAXPOOL(int32);
REGISTER_FRACTIONALMAXPOOL(int64_t);
REGISTER_FRACTIONALMAXPOOL(float);
REGISTER_FRACTIONALMAXPOOL(double);

#undef REGISTER_FRACTIONALMAXPOOL

static const int kInvalidMaxPoolingIndex = -1;

template <class T>
class FractionalMaxPoolGradOp : public OpKernel {
 public:
  explicit FractionalMaxPoolGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfractional_max_pool_opDTcc mht_2(mht_2_v, 394, "", "./tensorflow/core/kernels/fractional_max_pool_op.cc", "FractionalMaxPoolGradOp");

    OP_REQUIRES_OK(context, context->GetAttr("overlapping", &overlapping_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfractional_max_pool_opDTcc mht_3(mht_3_v, 401, "", "./tensorflow/core/kernels/fractional_max_pool_op.cc", "Compute");

    // There are two steps when calculating gradient for FractionalMaxPool.
    // 1) Walk through the process of calculating fractional pooling given
    //    pooling region; however, in the process, keep track of where the max
    //    element comes from. (arg_max)
    // 2) Populate the value of out_backprop to where arg_max indicates. If
    //    we support overlapping, it is likely to have multiple out_backprop[i]
    //    propagates back to the same arg_max value.
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        ConstEigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        EigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic>>
        EigenIndexMatrixMap;

    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_backprop = context->input(2);
    const Tensor& height_seq_tensor = context->input(3);
    const Tensor& width_seq_tensor = context->input(4);

    // Just to make it similar to FractionalMaxPoolOp.
    constexpr int tensor_in_and_out_dims = 4;
    OP_REQUIRES(
        context, tensor_in.dims() == tensor_in_and_out_dims,
        errors::InvalidArgument("orig_input should be a tensor of rank 4, got ",
                                tensor_in.DebugString()));
    OP_REQUIRES(context, tensor_in.NumElements() > 0,
                errors::InvalidArgument("orig_input must not be empty, got ",
                                        tensor_in.DebugString()));
    OP_REQUIRES(context, tensor_out.dims() == tensor_in_and_out_dims,
                errors::InvalidArgument(
                    "orig_output should be a tensor of rank 4, got ",
                    tensor_out.DebugString()));
    OP_REQUIRES(context, tensor_out.NumElements() > 0,
                errors::InvalidArgument("orig_output must not be empty, got ",
                                        tensor_out.DebugString()));
    std::vector<int64_t> input_size(tensor_in_and_out_dims);
    std::vector<int64_t> output_size(tensor_in_and_out_dims);
    for (int i = 0; i < tensor_in_and_out_dims; ++i) {
      input_size[i] = tensor_in.dim_size(i);
    }
    for (int i = 0; i < tensor_in_and_out_dims; ++i) {
      output_size[i] = tensor_out.dim_size(i);
    }

    // ---------
    // Step 1
    // ---------
    Tensor tensor_out_dup;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_temp(
                                {1}, DataTypeToEnum<T>::v(), tensor_out.shape(),
                                &tensor_out_dup));
    Tensor tensor_out_arg_max;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int64_t>::v(),
                                                   tensor_out.shape(),
                                                   &tensor_out_arg_max));
    // Find arg_max for each tensor_out
    ConstEigenMatrixMap tensor_in_mat(
        tensor_in.flat<T>().data(), input_size[3],
        input_size[2] * input_size[1] * input_size[0]);
    EigenMatrixMap tensor_out_dup_mat(
        tensor_out_dup.flat<T>().data(), output_size[3],
        output_size[2] * output_size[1] * output_size[0]);
    EigenIndexMatrixMap tensor_out_arg_max_mat(
        tensor_out_arg_max.flat<int64_t>().data(), output_size[3],
        output_size[2] * output_size[1] * output_size[0]);

    tensor_out_arg_max.flat<int64_t>().setConstant(kInvalidMaxPoolingIndex);
    // Initializes the duplicate output tensor with MIN<T>.
    tensor_out_dup.flat<T>().setConstant(Eigen::NumTraits<T>::lowest());

    auto height_seq_tensor_flat = height_seq_tensor.flat<int64_t>();
    auto width_seq_tensor_flat = width_seq_tensor.flat<int64_t>();

    // Now walk through the process of fractional max pooling again.
    // For both input and output,
    // 0: batch
    // 1: height / row
    // 2: width / col
    // 3: depth / channel
    const int64_t height_max = input_size[1] - 1;
    const int64_t width_max = input_size[2] - 1;
    for (int64_t b = 0; b < input_size[0]; ++b) {
      // height sequence.
      for (int64_t hs = 0; hs < height_seq_tensor.dim_size(0) - 1; ++hs) {
        // height start and end.
        const int64_t height_start = height_seq_tensor_flat(hs);
        int64_t height_end = overlapping_ ? height_seq_tensor_flat(hs + 1)
                                          : height_seq_tensor_flat(hs + 1) - 1;
        height_end = std::min(height_end, height_max);

        // width sequence.
        for (int64_t ws = 0; ws < width_seq_tensor.dim_size(0) - 1; ++ws) {
          const int64_t out_index =
              (b * output_size[1] + hs) * output_size[2] + ws;
          // width start and end.
          const int64_t width_start = width_seq_tensor_flat(ws);
          int64_t width_end = overlapping_ ? width_seq_tensor_flat(ws + 1)
                                           : width_seq_tensor_flat(ws + 1) - 1;
          width_end = std::min(width_end, width_max);
          for (int64_t h = height_start; h <= height_end; ++h) {
            for (int64_t w = width_start; w <= width_end; ++w) {
              const int64_t in_index =
                  (b * input_size[1] + h) * input_size[2] + w;
              // Walk through each channel (depth).
              for (int64_t d = 0; d < input_size[3]; ++d) {
                const T& input_ref = tensor_in_mat.coeffRef(d, in_index);
                T& output_ref = tensor_out_dup_mat.coeffRef(d, out_index);
                int64_t& out_arg_max_ref =
                    tensor_out_arg_max_mat.coeffRef(d, out_index);
                if (output_ref < input_ref ||
                    out_arg_max_ref == kInvalidMaxPoolingIndex) {
                  output_ref = input_ref;
                  int input_offset = in_index * input_size[3] + d;
                  out_arg_max_ref = input_offset;
                }
              }
            }
          }
        }
      }
    }

    // Check tensor_out_dup is the same as tensor_out.
    ConstEigenMatrixMap tensor_out_mat(
        tensor_out.flat<T>().data(), output_size[3],
        output_size[2] * output_size[1] * output_size[0]);
    const int64_t num_reshaped_cols =
        output_size[2] * output_size[1] * output_size[0];
    for (int64_t i = 0; i < num_reshaped_cols; ++i) {
      for (int64_t j = 0; j < output_size[3]; ++j) {
        DCHECK_EQ(tensor_out_dup_mat(j, i), tensor_out_mat(j, i));
      }
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, tensor_in.shape(), &output));
    output->flat<T>().setZero();

    auto out_backprop_flat = out_backprop.flat<T>();
    auto input_backprop_flat = output->flat<T>();
    auto out_arg_max_flat = tensor_out_arg_max.flat<int64_t>();
    int num_total_outputs = out_backprop_flat.size();
    int num_total_inputs = input_backprop_flat.size();

    for (int index = 0; index < num_total_outputs; ++index) {
      int input_backprop_index = out_arg_max_flat(index);
      // According to maxpooling_op.cc, the performance impact below is small.
      CHECK(input_backprop_index >= 0 &&
            input_backprop_index < num_total_inputs)
          << "Invalid input backprop index: " << input_backprop_index << ", "
          << num_total_inputs;
      input_backprop_flat(input_backprop_index) += out_backprop_flat(index);
    }
  }

 private:
  bool overlapping_;
};

#define REGISTER_FRACTIONALMAXPOOLGRAD(type)              \
  REGISTER_KERNEL_BUILDER(Name("FractionalMaxPoolGrad")   \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          FractionalMaxPoolGradOp<type>)

REGISTER_FRACTIONALMAXPOOLGRAD(int32);
REGISTER_FRACTIONALMAXPOOLGRAD(int64_t);
REGISTER_FRACTIONALMAXPOOLGRAD(float);
REGISTER_FRACTIONALMAXPOOLGRAD(double);

#undef REGISTER_FRACTIONALMAXPOOLGRAD
}  // namespace tensorflow
