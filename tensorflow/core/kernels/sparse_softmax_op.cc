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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_softmax_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_softmax_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_softmax_opDTcc() {
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

// See docs in ../ops/sparse_ops.cc.

#define EIGEN_USE_THREADS

#include <numeric>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

using tensorflow::gtl::ArraySlice;
using tensorflow::sparse::SparseTensor;

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename T>
class SparseSoftmaxOp : public OpKernel {
 public:
  explicit SparseSoftmaxOp(OpKernelConstruction *context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_softmax_opDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/sparse_softmax_op.cc", "SparseSoftmaxOp");
}

  void Compute(OpKernelContext *context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_softmax_opDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/kernels/sparse_softmax_op.cc", "Compute");

    const Tensor *indices_t, *values_t, *shape_t;
    OP_REQUIRES_OK(context, context->input("sp_indices", &indices_t));
    OP_REQUIRES_OK(context, context->input("sp_values", &values_t));
    OP_REQUIRES_OK(context, context->input("sp_shape", &shape_t));

    // Validations.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(indices_t->shape()),
                errors::InvalidArgument(
                    "Input sp_indices should be a matrix but received shape: ",
                    indices_t->shape().DebugString()));
    OP_REQUIRES(context,
                TensorShapeUtils::IsVector(values_t->shape()) &&
                    TensorShapeUtils::IsVector(shape_t->shape()),
                errors::InvalidArgument(
                    "Inputs sp_values and sp_shape should be vectors "
                    "but received shapes: ",
                    values_t->shape().DebugString(), " and ",
                    shape_t->shape().DebugString()));
    OP_REQUIRES(context, shape_t->NumElements() >= 2,
                errors::InvalidArgument(
                    "Input should have rank >= 2, but received shape: ",
                    shape_t->SummarizeValue(3)));
    TensorShape shape;
    OP_REQUIRES_OK(context, TensorShape::BuildTensorShape(
                                shape_t->flat<int64_t>(), &shape));

    const int64_t nnz = indices_t->dim_size(0);
    const int rank = static_cast<int>(indices_t->dim_size(1));
    SparseTensor st;
    OP_REQUIRES_OK(
        context, SparseTensor::Create(tensor::DeepCopy(*indices_t),
                                      tensor::DeepCopy(*values_t), shape, &st));

    Tensor *output_values = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({nnz}),
                                                     &output_values));
    typename TTypes<T>::Flat output_flat = output_values->flat<T>();

    Tensor tmp_t;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   TensorShape({}), &tmp_t));
    typename TTypes<T>::Scalar tmp_scalar = tmp_t.scalar<T>();

    gtl::InlinedVector<int64_t, 4> dims(rank);
    std::iota(dims.begin(), dims.end(), 0);
    // { 0, ..., rank-1 }.
    const ArraySlice<int64_t> kReorderDims(dims);
    // All but the last dim -- the class dimension to be max-reduced along.
    const ArraySlice<int64_t> kGroupByDims = kReorderDims.subspan(0, rank - 1);
    st.Reorder<T>(kReorderDims);
    int count = 0;

    // The SparseTensor has logical shape [..., b, c], where the
    // innermost size-"c" dimension is the class dimension to be max-reduced.
    // Therefore we group by the first (rank - 1) dimensions.
    const Device &device = context->eigen_device<Device>();
    for (const auto &g : st.group(kGroupByDims)) {
      const auto group_vals = g.values<T>();
      const int group_size = group_vals.size();

      // Shifts by max, exponentiates, then renormalizes.
      tmp_scalar.device(context->eigen_device<Device>()) = group_vals.maximum();
      const T group_max = tmp_scalar();

      Eigen::Tensor<T, 1, Eigen::RowMajor> tmp(group_size);
      tmp.device(device) = (group_vals - tmp.constant(group_max)).exp();

      tmp_scalar.device(device) = tmp.sum().inverse();
      tmp.device(device) = tmp * tmp.constant(tmp_scalar());

      // Assigns back to output[count, count + group_size).
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> output_part(
          output_flat.data() + count, group_size);
      output_part.device(device) = tmp;

      count += group_size;
    }
  }
};

#define REGISTER_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("SparseSoftmax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseSoftmaxOp<CPUDevice, T>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

}  // namespace tensorflow
