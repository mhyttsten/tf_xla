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
class MHTracer_DTPStensorflowPScorePSkernelsPSlistdiff_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlistdiff_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlistdiff_opDTcc() {
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

#include <string>
#include <unordered_set>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template <typename T, typename Tidx>
class ListDiffOp : public OpKernel {
 public:
  explicit ListDiffOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlistdiff_opDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/kernels/listdiff_op.cc", "ListDiffOp");

    const DataType dt = DataTypeToEnum<T>::v();
    const DataType dtidx = DataTypeToEnum<Tidx>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt, dtidx}));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlistdiff_opDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/kernels/listdiff_op.cc", "Compute");

    const Tensor& x = context->input(0);
    const Tensor& y = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(x.shape()),
                errors::InvalidArgument("x should be a 1D vector."));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(y.shape()),
                errors::InvalidArgument("y should be a 1D vector."));

    const auto Tx = x.vec<T>();
    const size_t x_size = Tx.size();
    const auto Ty = y.vec<T>();
    const size_t y_size = Ty.size();

    OP_REQUIRES(context, x_size < std::numeric_limits<int32>::max(),
                errors::InvalidArgument("x too large for int32 indexing"));

    std::unordered_set<T> y_set;
    y_set.reserve(y_size);
    for (size_t i = 0; i < y_size; ++i) {
      y_set.insert(Ty(i));
    }

    // Compute the size of the output.

    int64_t out_size = 0;
    for (size_t i = 0; i < x_size; ++i) {
      if (y_set.count(Tx(i)) == 0) {
        ++out_size;
      }
    }

    // Allocate and populate outputs.
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {out_size}, &out));
    auto Tout = out->vec<T>();

    Tensor* indices = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {out_size}, &indices));
    auto Tindices = indices->vec<Tidx>();

    for (Tidx i = 0, p = 0; i < static_cast<Tidx>(x_size); ++i) {
      if (y_set.count(Tx(i)) == 0) {
        OP_REQUIRES(context, p < out_size,
                    errors::InvalidArgument(
                        "Tried to set output index ", p,
                        " when output Tensor only had ", out_size,
                        " elements. Check that your "
                        "input tensors are not being concurrently mutated."));
        Tout(p) = Tx(i);
        Tindices(p) = i;
        p++;
      }
    }
  }
};

#define REGISTER_LISTDIFF(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("ListDiff")                         \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("out_idx"),   \
                          ListDiffOp<type, int32>)                 \
  REGISTER_KERNEL_BUILDER(Name("ListDiff")                         \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64_t>("out_idx"), \
                          ListDiffOp<type, int64>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_LISTDIFF);
REGISTER_LISTDIFF(tstring);
#undef REGISTER_LISTDIFF

}  // namespace tensorflow
