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
class MHTracer_DTPStensorflowPScorePSkernelsPSragged_range_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_range_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSragged_range_opDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

using errors::InvalidArgument;

template <typename T, typename SPLITS_TYPE>
class RaggedRangeOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_range_opDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/ragged_range_op.cc", "Compute");

    const Tensor& starts_in = context->input(0);
    const Tensor& limits_in = context->input(1);
    const Tensor& deltas_in = context->input(2);

    // Check input tensor shapes.
    OP_REQUIRES(context, starts_in.shape().dims() <= 1,
                InvalidArgument("starts must be a scalar or vector"));
    OP_REQUIRES(context, limits_in.shape().dims() <= 1,
                InvalidArgument("limits must be a scalar or vector"));
    OP_REQUIRES(context, deltas_in.shape().dims() <= 1,
                InvalidArgument("deltas must be a scalar or vector"));

    // Determine which tensors we need to broadcast.
    bool broadcast_starts = starts_in.shape().dims() == 0;
    bool broadcast_limits = limits_in.shape().dims() == 0;
    bool broadcast_deltas = deltas_in.shape().dims() == 0;

    // nrows (number of output rows) is the size of the non-broadcast inputs,
    // or 1 if all inputs are scalars.
    std::vector<int> in_sizes;
    if (!broadcast_starts) in_sizes.push_back(starts_in.shape().dim_size(0));
    if (!broadcast_limits) in_sizes.push_back(limits_in.shape().dim_size(0));
    if (!broadcast_deltas) in_sizes.push_back(deltas_in.shape().dim_size(0));
    for (int i = 1; i < in_sizes.size(); ++i) {
      OP_REQUIRES(context, in_sizes[i] == in_sizes[i - 1],
                  InvalidArgument("starts, limits, and deltas must have the "
                                  "same shape"));
    }
    SPLITS_TYPE nrows = in_sizes.empty() ? 1 : in_sizes[0];

    const auto& starts = starts_in.flat<T>();
    const auto& limits = limits_in.flat<T>();
    const auto& deltas = deltas_in.flat<T>();

    // Construct the rt_nested_splits tensor.
    Tensor* rt_nested_splits_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({nrows + 1}),
                                            &rt_nested_splits_out));
    auto rt_nested_splits = rt_nested_splits_out->flat<SPLITS_TYPE>();
    rt_nested_splits(0) = 0;
    for (int row = 0; row < nrows; ++row) {
      T start = broadcast_starts ? starts(0) : starts(row);
      T limit = broadcast_limits ? limits(0) : limits(row);
      T delta = broadcast_deltas ? deltas(0) : deltas(row);
      OP_REQUIRES(context, delta != 0, InvalidArgument("Requires delta != 0"));
      rt_nested_splits(row + 1) =
          rt_nested_splits(row) + RangeSize(start, limit, delta);
    }
    SPLITS_TYPE nvals = rt_nested_splits(nrows);

    // Construct the rt_dense_values tensor.
    Tensor* rt_dense_values_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({nvals}),
                                                     &rt_dense_values_out));
    auto rt_dense_values = rt_dense_values_out->flat<T>();
    int value_index = 0;
    for (int row = 0; row < nrows; ++row) {
      SPLITS_TYPE row_size = rt_nested_splits(row + 1) - rt_nested_splits(row);
      T value = broadcast_starts ? starts(0) : starts(row);
      T delta = broadcast_deltas ? deltas(0) : deltas(row);
      for (SPLITS_TYPE i = 0; i < row_size; ++i) {
        rt_dense_values(value_index++) = T(value);
        value += delta;
      }
    }
  }

 private:
  // Returns the number of elements in the specified range.
  SPLITS_TYPE RangeSize(T start, T limit, T delta) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_range_opDTcc mht_1(mht_1_v, 277, "", "./tensorflow/core/kernels/ragged_range_op.cc", "RangeSize");

    if (((delta > 0) && (limit < start)) || ((delta < 0) && (limit > start))) {
      return 0;
    }
    // The following is copied from tensorflow::RangeOp::Compute().
    return (std::is_integral<T>::value
                ? ((std::abs(limit - start) + std::abs(delta) - 1) /
                   std::abs(delta))
                : std::ceil(std::abs((limit - start) / delta)));
  }
};

#define REGISTER_CPU_KERNEL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("RaggedRange")                      \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<TYPE>("T")           \
                              .TypeConstraint<int32>("Tsplits"),   \
                          RaggedRangeOp<TYPE, int32>);             \
  REGISTER_KERNEL_BUILDER(Name("RaggedRange")                      \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<TYPE>("T")           \
                              .TypeConstraint<int64_t>("Tsplits"), \
                          RaggedRangeOp<TYPE, int64>);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
TF_CALL_int32(REGISTER_CPU_KERNEL);
TF_CALL_int64(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

}  // namespace tensorflow
