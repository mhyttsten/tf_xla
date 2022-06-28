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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_cwise_ops_commonDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_cwise_ops_commonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_cwise_ops_commonDTcc() {
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

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef INTEL_MKL

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS
#include <iostream>
#include <vector>

#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename Functor>
class MklBinaryOp : public BinaryOp<Device, Functor> {
 public:
  explicit MklBinaryOp(OpKernelConstruction* context)
      : BinaryOp<Device, Functor>(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_cwise_ops_commonDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/mkl/mkl_cwise_ops_common.cc", "MklBinaryOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_cwise_ops_commonDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/kernels/mkl/mkl_cwise_ops_common.cc", "Compute");

    auto in0 = context->input(0);
    auto in1 = context->input(1);
    VLOG(1) << "Shapes (start mklbinaryop compute): "
            << in0.shape().DebugString() << " _and_ "
            << in1.shape().DebugString();

    // Call the TensorFlow BinaryOp Compute method
    BinaryOp<Device, Functor>::Compute(context);

    auto out = context->mutable_output(0);
    VLOG(1) << "Shapes (output): " << out->shape().DebugString();

    // Pass input shape through to output shape
    ForwardMklMetaDataInToOut(context, 0, 0);

    out = context->mutable_output(0);
    VLOG(1) << "Shapes (output): " << out->shape().DebugString();
  }
};

//---------- Registration macros for various element-wise ops -----------
// We will need to redefine "REGISTER" to include the mkl_op_registry flag
#pragma push_macro("REGISTER")
#undef REGISTER
#define REGISTER(OP, D, N, F, T)                               \
  REGISTER_KERNEL_BUILDER(                                     \
      Name(N)                                                  \
          .Device(DEVICE_##D)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      OP<D##Device, F<T>>);

REGISTER6(MklBinaryOp, CPU, "_MklAdd", functor::add, float, Eigen::half, double,
          int32, int64, bfloat16);
REGISTER6(MklBinaryOp, CPU, "_MklAddV2", functor::add, float, Eigen::half,
          double, int32, int64, bfloat16);
REGISTER8(MklBinaryOp, CPU, "_MklSub", functor::sub, float, Eigen::half, double,
          int32, int64, complex64, complex128, bfloat16);
REGISTER6(MklBinaryOp, CPU, "_MklMul", functor::mul, float, Eigen::half, double,
          uint8, int32, bfloat16);
REGISTER6(MklBinaryOp, CPU, "_MklMaximum", functor::maximum, float, Eigen::half,
          double, int32, int64, bfloat16);
REGISTER6(MklBinaryOp, CPU, "_MklSquaredDifference",
          functor::squared_difference, float, Eigen::half, double, int32, int64,
          bfloat16);

#undef REGISTER
#pragma pop_macro("REGISTER")
//-----------------------------------------------------------------------

}  // end namespace tensorflow

#endif  // INTEL_MKL
