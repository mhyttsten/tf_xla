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
class MHTracer_DTPStensorflowPScorePSkernelsPSunravel_index_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSunravel_index_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSunravel_index_opDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

namespace {
template <typename T>
struct mod_op {
  const T operator()(const T& a, const T& b) const { return a % b; }
};
}  // namespace

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Tidx>
class UnravelIndexOp : public OpKernel {
 public:
  explicit UnravelIndexOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), dtidx_(DataTypeToEnum<Tidx>::v()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunravel_index_opDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/unravel_index_op.cc", "UnravelIndexOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunravel_index_opDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/kernels/unravel_index_op.cc", "Compute");

    const Tensor& indices_tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(indices_tensor.shape()) ||
                    TensorShapeUtils::IsScalar(indices_tensor.shape()),
                errors::InvalidArgument(
                    "The indices can only be scalar or vector, got \"",
                    indices_tensor.shape().DebugString(), "\""));

    const Tensor& dims_tensor = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(dims_tensor.shape()),
        errors::InvalidArgument("The indices can only be 1-D, got \"",
                                dims_tensor.shape().DebugString(), "\""));

    auto dims = dims_tensor.vec<Tidx>();
    // Make sure dims does not contain a zero
    double prod = 1;
    uint64_t limit;
    if (dtidx_ == DataType::DT_INT64) {
      limit = kint64max;
    } else {
      limit = kint32max;
    }

    for (int i = 0; i < dims.size(); i++) {
      OP_REQUIRES(
          ctx, dims(i) != 0,
          errors::InvalidArgument("Input dims cannot contain a dim of zero, "
                                  "but dims contains zero at index ",
                                  i));
      OP_REQUIRES(ctx, dims(i) > 0,
                  errors::InvalidArgument(
                      "Input dims cannot be negative. Got dim = ", dims(i),
                      " at index ", i));
      // Check interger overflow
      OP_REQUIRES(
          ctx, prod <= limit / dims(i),
          errors::InvalidArgument("Input dims product is causing integer "
                                  "overflow: (",
                                  dims, ")"));
      prod = (prod * dims(i));
    }

    // Check to make sure indices is not out of boundary
    Eigen::Tensor<Tidx, 0, Eigen::RowMajor> dims_prod_eigen = dims.prod();
    Tidx dims_prod = dims_prod_eigen();
    const Tidx* indices = indices_tensor.flat<Tidx>().data();
    int64_t size = indices_tensor.NumElements();
    bool check = std::all_of(indices, indices + size,
                             [&](Tidx index) { return index < dims_prod; });
    OP_REQUIRES(ctx, check,
                errors::InvalidArgument("index is out of bound as with dims"));

    Eigen::array<bool, 1> reverse({true});

    Tensor strides_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<Tidx>::value,
                                      TensorShape({dims_tensor.NumElements()}),
                                      &strides_tensor));

    auto strides = strides_tensor.vec<Tidx>();
    strides = dims.reverse(reverse)
                  .scan(0, Eigen::internal::ProdReducer<Tidx>(), false)
                  .reverse(reverse);

    Tensor strides_shifted_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<Tidx>::value,
                                      TensorShape({dims_tensor.NumElements()}),
                                      &strides_shifted_tensor));

    auto strides_shifted = strides_shifted_tensor.vec<Tidx>();
    strides_shifted = dims.reverse(reverse)
                          .scan(0, Eigen::internal::ProdReducer<Tidx>(), true)
                          .reverse(reverse);

    Tensor* output_tensor = nullptr;
    if (TensorShapeUtils::IsScalar(indices_tensor.shape())) {
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0, TensorShape({dims_tensor.NumElements()}),
                                    &output_tensor));

      auto output = output_tensor->vec<Tidx>();

      output = output.constant(indices_tensor.scalar<Tidx>()());
      output = output.binaryExpr(strides, mod_op<Tidx>()) / strides_shifted;
    } else {
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0,
                                    TensorShape({dims_tensor.NumElements(),
                                                 indices_tensor.NumElements()}),
                                    &output_tensor));

      auto output = output_tensor->matrix<Tidx>();

      Eigen::array<Eigen::Index, 2> reshape{
          {static_cast<Eigen::Index>(dims_tensor.NumElements()), 1}};
      Eigen::array<Eigen::Index, 2> bcast(
          {1, static_cast<Eigen::Index>(indices_tensor.NumElements())});
      Eigen::array<Eigen::Index, 2> indices_reshape{
          {1, static_cast<Eigen::Index>(indices_tensor.NumElements())}};
      Eigen::array<Eigen::Index, 2> indices_bcast(
          {static_cast<Eigen::Index>(dims_tensor.NumElements()), 1});

      output = indices_tensor.vec<Tidx>()
                   .reshape(indices_reshape)
                   .broadcast(indices_bcast);
      output = output.binaryExpr(strides.reshape(reshape).broadcast(bcast),
                                 mod_op<Tidx>()) /
               strides_shifted.reshape(reshape).broadcast(bcast);
    }
  }
  const DataType dtidx_;
};

#define REGISTER_KERNEL(type)                                               \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("UnravelIndex").Device(DEVICE_CPU).TypeConstraint<type>("Tidx"), \
      UnravelIndexOp<type>);
TF_CALL_int32(REGISTER_KERNEL) TF_CALL_int64(REGISTER_KERNEL)
#undef REGISTER_KERNEL

}  // namespace tensorflow
