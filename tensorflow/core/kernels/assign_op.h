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

#ifndef TENSORFLOW_CORE_KERNELS_ASSIGN_OP_H_
#define TENSORFLOW_CORE_KERNELS_ASSIGN_OP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSassign_opDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSassign_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSassign_opDTh() {
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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

// TODO(jeff): Get rid of use_exclusive_lock_ option

// Computes *input[0] = input[1]
class AssignOp : public OpKernel {
 public:
  explicit AssignOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSassign_opDTh mht_0(mht_0_v, 201, "", "./tensorflow/core/kernels/assign_op.h", "AssignOp");

    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("validate_shape", &validate_shape_));
    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
    if (!context
             ->GetAttr("_grappler_relax_allocator_constraints",
                       &relax_constraints_)
             .ok()) {
      relax_constraints_ = false;
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSassign_opDTh mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/assign_op.h", "Compute");

    const Tensor& rhs = context->input(1);

    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);

    // Prevent copying uninitialized data, to solve harder to debug undefined
    // behaviors that cannot be traced back to the original tensor.
    OP_REQUIRES(
        context, rhs.IsInitialized(),
        errors::Internal("Right hand side of AssignOp is not initialized"));

    // We can't always know how this value will be used downstream, so make
    // conservative assumptions in specifying constraints on the memory
    // allocation attributes, unless the Grappler graph analysis determined that
    // it was safe not to.
    AllocatorAttributes attr;
    if (!relax_constraints_) {
      attr.set_gpu_compatible(true);
      attr.set_nic_compatible(true);
    }

    {
      mutex_lock l(*context->input_ref_mutex(0));
      const Tensor& old_lhs = context->mutable_input(0, /* lock_held */ true);
      const bool same_shape = old_lhs.shape().IsSameSize(rhs.shape());
      if (validate_shape_) {
        OP_REQUIRES(context, same_shape,
                    errors::InvalidArgument(
                        "Assign requires shapes of both tensors to match. "
                        "lhs shape= ",
                        old_lhs.shape().DebugString(),
                        " rhs shape= ", rhs.shape().DebugString()));
      }

      // In the code below we try to minimize the amount of memory allocation
      // and copying by trying the following two shortcuts:
      // 1. If the lhs is initialized and has the same number of elements as
      //    the rhs we can avoid a memory allocation.
      // 2. If we can reuse the rhs buffer we avoid both a memory allocation
      //    and copying.

      // 1. Try to copy into an existing buffer.
      if (old_lhs.IsInitialized() &&
          old_lhs.shape().num_elements() == rhs.shape().num_elements()) {
        // The existing lhs tensor has already been initialized and the right
        // hand side can fit in the underlying buffer.
        Tensor reshaped_old_lhs;
        if (same_shape) {
          reshaped_old_lhs = old_lhs;
        } else {
          CHECK(reshaped_old_lhs.CopyFrom(old_lhs, rhs.shape()));
          context->replace_ref_input(0, reshaped_old_lhs,
                                     /* lock_held */ true);
        }
        if (use_exclusive_lock_) {
          Copy(context, &reshaped_old_lhs, rhs);
          return;
        }
      } else {
        // 2. Try to reuse the rhs.
        std::unique_ptr<Tensor> input_alias = context->forward_input(
            1, OpKernelContext::Params::kNoReservation /*output_index*/,
            rhs.dtype(), rhs.shape(), DEVICE_MEMORY, attr);
        if (input_alias != nullptr) {
          // Update the ref to point to the new buffer.
          context->replace_ref_input(0, *input_alias, /* lock_held */ true);
          return;
        }

        // Otherwise, create a new tensor whose shape matches the
        // right hand side, hand off to lhs and copy the rhs into it.
        Tensor copy_tensor;
        OP_REQUIRES_OK(context,
                       context->allocate_temp(old_lhs.dtype(), rhs.shape(),
                                              &copy_tensor, attr));
        // We track memory of variables in variable ops instead of in this
        // assign op.
        context->clear_recorded_memory();
        context->replace_ref_input(0, copy_tensor, /* lock_held */ true);
        if (use_exclusive_lock_) {
          Copy(context, &copy_tensor, rhs);
          return;
        }
      }
    }

    // The tensor has already been initialized and the right hand side
    // matches the left hand side's shape. We have been told to do the
    // copy outside the lock.
    Tensor old_unlocked_lhs = context->mutable_input(0, /* lock_held */ false);
    Copy(context, &old_unlocked_lhs, rhs);
  }

  virtual void Copy(OpKernelContext* context, Tensor* lhs,
                    const Tensor& rhs) = 0;

  bool use_exclusive_lock_;
  bool validate_shape_;
  bool relax_constraints_;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_ASSIGN_OP_H_
