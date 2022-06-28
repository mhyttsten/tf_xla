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
class MHTracer_DTPStensorflowPScorePSkernelsPSedit_distance_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSedit_distance_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSedit_distance_opDTcc() {
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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include <limits>

#include <vector>
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/edit_distance.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

namespace {

Status ValidateShapes(OpKernelContext* ctx, const Tensor& hypothesis_indices,
                      const Tensor& hypothesis_values,
                      const Tensor& hypothesis_shape,
                      const Tensor& truth_indices, const Tensor& truth_values,
                      const Tensor& truth_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSedit_distance_opDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/edit_distance_op.cc", "ValidateShapes");

  if (!TensorShapeUtils::IsMatrix(hypothesis_indices.shape()))
    return errors::InvalidArgument(
        "hypothesis_indices should be a matrix, but got shape: ",
        hypothesis_indices.shape().DebugString());
  if (!TensorShapeUtils::IsMatrix(truth_indices.shape()))
    return errors::InvalidArgument(
        "truth_indices should be a matrix, but got shape: ",
        truth_indices.shape().DebugString());
  if (!TensorShapeUtils::IsVector(hypothesis_values.shape()))
    return errors::InvalidArgument(
        "hypothesis_values should be a vector, but got shape: ",
        hypothesis_values.shape().DebugString());
  if (!TensorShapeUtils::IsVector(truth_values.shape()))
    return errors::InvalidArgument(
        "truth_values should be a vector, but got shape: ",
        truth_values.shape().DebugString());
  if (!TensorShapeUtils::IsVector(hypothesis_shape.shape()))
    return errors::InvalidArgument(
        "hypothesis_shape should be a vector, but got shape: ",
        hypothesis_shape.shape().DebugString());
  if (!TensorShapeUtils::IsVector(truth_shape.shape()))
    return errors::InvalidArgument(
        "truth_shape should be a vector, but got shape: ",
        truth_shape.shape().DebugString());
  if (hypothesis_values.NumElements() != hypothesis_indices.dim_size(0))
    return errors::InvalidArgument(
        "Expected hypothesis_values.NumElements == "
        "#rows(hypothesis_indices), their shapes are: ",
        hypothesis_values.shape().DebugString(), " and ",
        hypothesis_indices.shape().DebugString());
  if (hypothesis_shape.NumElements() != hypothesis_indices.dim_size(1))
    return errors::InvalidArgument(
        "Expected hypothesis_shape.NumElements == "
        "#cols(hypothesis_indices), their shapes are: ",
        hypothesis_shape.shape().DebugString(), " and ",
        hypothesis_indices.shape().DebugString());
  if (truth_shape.NumElements() < 2)
    return errors::InvalidArgument(
        "Input SparseTensors must have rank at least 2, but truth_shape "
        "rank is: ",
        truth_shape.NumElements());
  if (truth_values.NumElements() != truth_indices.dim_size(0))
    return errors::InvalidArgument(
        "Expected truth_values.NumElements == "
        "#rows(truth_indices), their shapes are: ",
        truth_values.shape().DebugString(), " and ",
        truth_indices.shape().DebugString());
  if (truth_shape.NumElements() != truth_indices.dim_size(1))
    return errors::InvalidArgument(
        "Expected truth_shape.NumElements == "
        "#cols(truth_indices), their shapes are: ",
        truth_shape.shape().DebugString(), " and ",
        truth_indices.shape().DebugString());
  if (truth_shape.NumElements() != hypothesis_shape.NumElements())
    return errors::InvalidArgument(
        "Expected truth and hypothesis to have matching ranks, but "
        "their shapes are: ",
        truth_shape.shape().DebugString(), " and ",
        hypothesis_shape.shape().DebugString());

  return Status::OK();
}

}  // namespace

template <typename T>
class EditDistanceOp : public OpKernel {
 public:
  explicit EditDistanceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSedit_distance_opDTcc mht_1(mht_1_v, 283, "", "./tensorflow/core/kernels/edit_distance_op.cc", "EditDistanceOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("normalize", &normalize_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSedit_distance_opDTcc mht_2(mht_2_v, 290, "", "./tensorflow/core/kernels/edit_distance_op.cc", "Compute");

    const Tensor* hypothesis_indices;
    const Tensor* hypothesis_values;
    const Tensor* hypothesis_shape;
    const Tensor* truth_indices;
    const Tensor* truth_values;
    const Tensor* truth_shape;
    OP_REQUIRES_OK(ctx, ctx->input("hypothesis_indices", &hypothesis_indices));
    OP_REQUIRES_OK(ctx, ctx->input("hypothesis_values", &hypothesis_values));
    OP_REQUIRES_OK(ctx, ctx->input("hypothesis_shape", &hypothesis_shape));
    OP_REQUIRES_OK(ctx, ctx->input("truth_indices", &truth_indices));
    OP_REQUIRES_OK(ctx, ctx->input("truth_values", &truth_values));
    OP_REQUIRES_OK(ctx, ctx->input("truth_shape", &truth_shape));

    OP_REQUIRES_OK(
        ctx, ValidateShapes(ctx, *hypothesis_indices, *hypothesis_values,
                            *hypothesis_shape, *truth_indices, *truth_values,
                            *truth_shape));

    TensorShape hypothesis_st_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeUtils::MakeShape(
                       hypothesis_shape->vec<int64_t>().data(),
                       hypothesis_shape->NumElements(), &hypothesis_st_shape));
    TensorShape truth_st_shape;
    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                            truth_shape->vec<int64_t>().data(),
                            truth_shape->NumElements(), &truth_st_shape));

    // Assume indices are sorted in row-major order.
    std::vector<int64_t> sorted_order(truth_st_shape.dims());
    std::iota(sorted_order.begin(), sorted_order.end(), 0);

    sparse::SparseTensor hypothesis;
    OP_REQUIRES_OK(ctx, sparse::SparseTensor::Create(
                            *hypothesis_indices, *hypothesis_values,
                            hypothesis_st_shape, sorted_order, &hypothesis));

    sparse::SparseTensor truth;
    OP_REQUIRES_OK(ctx, sparse::SparseTensor::Create(
                            *truth_indices, *truth_values, truth_st_shape,
                            sorted_order, &truth));

    // Group dims 0, 1, ..., RANK - 1.  The very last dim is assumed
    // to store the variable length sequences.
    std::vector<int64_t> group_dims(truth_st_shape.dims() - 1);
    std::iota(group_dims.begin(), group_dims.end(), 0);

    TensorShape output_shape;
    for (int d = 0; d < static_cast<int>(group_dims.size()); ++d) {
      output_shape.AddDim(std::max(hypothesis_st_shape.dim_size(d),
                                   truth_st_shape.dim_size(d)));
    }
    const auto output_elements = output_shape.num_elements();
    OP_REQUIRES(
        ctx, output_elements > 0,
        errors::InvalidArgument("Got output shape ", output_shape.DebugString(),
                                " which has 0 elements"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", output_shape, &output));
    auto output_t = output->flat<float>();
    output_t.setZero();

    std::vector<int64_t> output_strides(output_shape.dims());
    output_strides[output_shape.dims() - 1] = 1;
    for (int d = output_shape.dims() - 2; d >= 0; --d) {
      output_strides[d] = output_strides[d + 1] * output_shape.dim_size(d + 1);
    }

    auto hypothesis_grouper = hypothesis.group(group_dims);
    auto truth_grouper = truth.group(group_dims);

    auto hypothesis_iter = hypothesis_grouper.begin();
    auto truth_iter = truth_grouper.begin();

    auto cmp = std::equal_to<T>();

    while (hypothesis_iter != hypothesis_grouper.end() &&
           truth_iter != truth_grouper.end()) {
      sparse::Group truth_i = *truth_iter;
      sparse::Group hypothesis_j = *hypothesis_iter;
      std::vector<int64_t> g_truth = truth_i.group();
      std::vector<int64_t> g_hypothesis = hypothesis_j.group();
      auto truth_seq = truth_i.values<T>();
      auto hypothesis_seq = hypothesis_j.values<T>();

      if (g_truth == g_hypothesis) {
        auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                      output_strides.begin(), int64_t{0});
        OP_REQUIRES(
            ctx, 0 <= loc && loc < output_elements,
            errors::Internal("Got an inner product ", loc,
                             " which would require writing to outside of "
                             "the buffer for the output tensor (max elements ",
                             output_elements, ")"));
        output_t(loc) =
            gtl::LevenshteinDistance<T>(truth_seq, hypothesis_seq, cmp);
        if (normalize_) output_t(loc) /= truth_seq.size();

        ++hypothesis_iter;
        ++truth_iter;
      } else if (g_truth > g_hypothesis) {  // zero-length truth
        auto loc = std::inner_product(g_hypothesis.begin(), g_hypothesis.end(),
                                      output_strides.begin(), int64_t{0});
        OP_REQUIRES(
            ctx, 0 <= loc && loc < output_elements,
            errors::Internal("Got an inner product ", loc,
                             " which would require writing to outside of "
                             "the buffer for the output tensor (max elements ",
                             output_elements, ")"));
        output_t(loc) = hypothesis_seq.size();
        if (normalize_ && output_t(loc) != 0.0f) {
          output_t(loc) = std::numeric_limits<float>::infinity();
        }
        ++hypothesis_iter;
      } else {  // zero-length hypothesis
        auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                      output_strides.begin(), int64_t{0});
        OP_REQUIRES(
            ctx, 0 <= loc && loc < output_elements,
            errors::Internal("Got an inner product ", loc,
                             " which would require writing to outside of "
                             "the buffer for the output tensor (max elements ",
                             output_elements, ")"));
        output_t(loc) = (normalize_) ? 1.0 : truth_seq.size();
        ++truth_iter;
      }
    }
    while (hypothesis_iter != hypothesis_grouper.end()) {  // zero-length truths
      sparse::Group hypothesis_j = *hypothesis_iter;
      std::vector<int64_t> g_hypothesis = hypothesis_j.group();
      auto hypothesis_seq = hypothesis_j.values<T>();
      auto loc = std::inner_product(g_hypothesis.begin(), g_hypothesis.end(),
                                    output_strides.begin(), int64_t{0});
      OP_REQUIRES(
          ctx, 0 <= loc && loc < output_elements,
          errors::Internal("Got an inner product ", loc,
                           " which would require writing to outside of the "
                           "buffer for the output tensor (max elements ",
                           output_elements, ")"));
      output_t(loc) = hypothesis_seq.size();
      if (normalize_ && output_t(loc) != 0.0f) {
        output_t(loc) = std::numeric_limits<float>::infinity();
      }
      ++hypothesis_iter;
    }
    while (truth_iter != truth_grouper.end()) {  // missing hypotheses
      sparse::Group truth_i = *truth_iter;
      std::vector<int64_t> g_truth = truth_i.group();
      auto truth_seq = truth_i.values<T>();
      auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                    output_strides.begin(), int64_t{0});
      OP_REQUIRES(
          ctx, 0 <= loc && loc < output_elements,
          errors::Internal("Got an inner product ", loc,
                           " which would require writing to outside of the "
                           "buffer for the output tensor (max elements ",
                           output_elements, ")"));
      output_t(loc) = (normalize_) ? 1.0 : truth_seq.size();
      ++truth_iter;
    }
  }

 private:
  bool normalize_;

  TF_DISALLOW_COPY_AND_ASSIGN(EditDistanceOp);
};

#define REGISTER_CPU_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("EditDistance").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      EditDistanceOp<T>);

TF_CALL_POD_STRING_TYPES(REGISTER_CPU_KERNEL);

#undef REGISTER_CPU_KERNEL

}  // end namespace tensorflow
