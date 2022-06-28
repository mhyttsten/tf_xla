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
class MHTracer_DTPStensorflowPScorePSkernelsPSnth_element_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSnth_element_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSnth_element_opDTcc() {
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
#include "tensorflow/core/kernels/nth_element_op.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class NthElementOp : public OpKernel {
 public:
  explicit NthElementOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnth_element_opDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/nth_element_op.cc", "NthElementOp");

    OP_REQUIRES_OK(context, context->GetAttr("reverse", &reverse_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnth_element_opDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/kernels/nth_element_op.cc", "Compute");

    // The second args is N, which must be a positive scalar.
    const auto& n_in = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(n_in.shape()),
        errors::InvalidArgument("N must be scalar but has rank ", n_in.dims()));
    int n = n_in.scalar<int32>()();
    OP_REQUIRES(context, n >= 0,
                errors::InvalidArgument("n must be non-negative but is ", n));

    // The first args is input tensor, which must have 1 dimension at least.
    const Tensor& input_in = context->input(0);
    const int num_dims = input_in.dims();
    OP_REQUIRES(context, num_dims >= 1,
                errors::InvalidArgument(
                    "Input must be at least rank 1 but is rank ", num_dims));
    // The last dimension of input tensor must be greater than N.
    OP_REQUIRES(
        context, input_in.dim_size(num_dims - 1) > n,
        errors::InvalidArgument("Input must have last dimension > n = ", n));

    // std::nth_element only support the nth-smallest selection.
    if (reverse_) {
      n = input_in.dim_size(num_dims - 1) - n - 1;
    }

    // Assume input_shape is [d1,d2,...dk], and output_shape is [d1,d2...dk-1].
    TensorShape out_shape;
    for (int i = 0; i < num_dims - 1; ++i) {
      out_shape.AddDim(input_in.dim_size(i));
    }
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output_tensor));

    functor::NthElementFunctor<Device, T> nthElementFunc;
    nthElementFunc(context, input_in, *output_tensor, n, reverse_);
  }

 private:
  bool reverse_;
};

namespace functor {

template <typename T>
struct NthElementFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_tensor,
                  Tensor& output_tensor, int n, bool reverse) {
    const T* input = input_tensor.flat<T>().data();
    T* output = output_tensor.flat<T>().data();

    // Assume input_shape is [d1,d2,...dk], and output_shape is [d1,d2...dk-1],
    // then num_rows = d1*d2...dk-1, last_dim = dk.
    const int num_rows = output_tensor.NumElements();
    const int last_dim = input_tensor.dim_size(input_tensor.dims() - 1);

    // Allocate each row to different shard.
    auto SubNthElement = [&, input, output, last_dim, n](int64_t start,
                                                         int64_t limit) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnth_element_opDTcc mht_2(mht_2_v, 274, "", "./tensorflow/core/kernels/nth_element_op.cc", "lambda");

      // std::nth_element would rearrange the array, so we need a new buffer.
      std::vector<T> buf(last_dim);

      for (int b = start; b < limit; ++b) {
        // Copy from one row of elements to buffer
        const T* input_start = input + b * last_dim;
        const T* input_end = input + (b + 1) * last_dim;
        std::copy(input_start, input_end, buf.begin());

        std::nth_element(buf.begin(), buf.begin() + n, buf.end());
        // The element placed in the nth position is exactly the element that
        // would occur in this position if the range was fully sorted.
        output[b] = buf[n];
      }
    };

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    // The average time complexity of partition-based nth_element (BFPRT) is
    // O(n), although the worst time complexity could be O(n^2). Here, 20 is a
    // empirical factor of cost_per_unit.
    Shard(worker_threads.num_threads, worker_threads.workers, num_rows,
          20 * last_dim, SubNthElement);
  }
};

}  // namespace functor

#define REGISTER_NTHOP(T)                                           \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("NthElement").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      NthElementOp<CPUDevice, T>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_NTHOP);
#undef REGISTER_NTHOP

}  // end namespace tensorflow
