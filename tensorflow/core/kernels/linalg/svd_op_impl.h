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

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_SVD_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_SVD_OP_IMPL_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSsvd_op_implDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSsvd_op_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSsvd_op_implDTh() {
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


// See docs in ../ops/linalg_ops.cc.
//
// This header file is used by the individual svd_*op*.cc files for registering
// individual kernels. A separate file is used for each instantiated kernel to
// improve compilation times.
#include <algorithm>

#include "third_party/eigen3/Eigen/SVD"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <class Scalar>
class SvdOp : public LinearAlgebraOp<Scalar> {
 public:
  typedef LinearAlgebraOp<Scalar> Base;

  explicit SvdOp(OpKernelConstruction* context) : Base(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSsvd_op_implDTh mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/linalg/svd_op_impl.h", "SvdOp");

    OP_REQUIRES_OK(context, context->GetAttr("compute_uv", &compute_uv_));
    OP_REQUIRES_OK(context, context->GetAttr("full_matrices", &full_matrices_));
  }

  using TensorShapes = typename Base::TensorShapes;

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSsvd_op_implDTh mht_1(mht_1_v, 224, "", "./tensorflow/core/kernels/linalg/svd_op_impl.h", "ValidateInputMatrixShapes");

    Base::ValidateSingleMatrix(context, input_matrix_shapes);
  }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSsvd_op_implDTh mht_2(mht_2_v, 232, "", "./tensorflow/core/kernels/linalg/svd_op_impl.h", "GetOutputMatrixShapes");

    int64_t m = input_matrix_shapes[0].dim_size(0);
    int64_t n = input_matrix_shapes[0].dim_size(1);
    int64_t min_size = std::min(m, n);
    if (compute_uv_) {
      return TensorShapes({TensorShape({min_size}),
                           TensorShape({m, full_matrices_ ? m : min_size}),
                           TensorShape({n, full_matrices_ ? n : min_size})});
    } else {
      return TensorShapes({TensorShape({min_size})});
    }
  }

  // TODO(rmlarsen): This should depend on compute_uv. See b/30409375.
  int64_t GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSsvd_op_implDTh mht_3(mht_3_v, 249, "", "./tensorflow/core/kernels/linalg/svd_op_impl.h", "GetCostPerUnit");

    double m = static_cast<double>(input_matrix_shapes[0].dim_size(0));
    double n = static_cast<double>(input_matrix_shapes[0].dim_size(1));
    double cost = 12 * std::max(m, n) * std::min(m, n) * std::min(m, n);
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64_t>(cost);
  }

  using Matrix = typename Base::Matrix;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSsvd_op_implDTh mht_4(mht_4_v, 266, "", "./tensorflow/core/kernels/linalg/svd_op_impl.h", "ComputeMatrix");

    int64_t n = inputs[0].cols();
    int64_t m = inputs[0].rows();
    const bool empty = (m == 0 || n == 0);
    int options = 0;  // Don't compute singular vectors;
    if (compute_uv_) {
      options = full_matrices_ ? Eigen::ComputeFullU | Eigen::ComputeFullV
                               : Eigen::ComputeThinU | Eigen::ComputeThinV;
    }

    if (empty) {
      // For an empty matrix where only one dimension is zero, we still set
      // U or V to the unit matrix for the dimension that is non-zero.
      if (compute_uv_ && full_matrices_) {
        if (m > 0) {
          outputs->at(1) = Matrix::Identity(m, m);
        } else {
          outputs->at(2) = Matrix::Identity(n, n);
        }
      }
      return;
    }

    Eigen::BDCSVD<Matrix> svd(inputs[0], options);
    if (svd.info() != Eigen::Success) {
      LOG(ERROR) << "Eigen::BDCSVD failed with error code " << svd.info();
      outputs->at(0).fill(std::numeric_limits<Scalar>::quiet_NaN());
      if (compute_uv_) {
        outputs->at(1).fill(std::numeric_limits<Scalar>::quiet_NaN());
        outputs->at(2).fill(std::numeric_limits<Scalar>::quiet_NaN());
      }
    } else {
      outputs->at(0) = svd.singularValues().template cast<Scalar>();
      if (compute_uv_) {
        outputs->at(1) = svd.matrixU();
        outputs->at(2) = svd.matrixV();
      }
    }
  }

 private:
  bool compute_uv_;
  bool full_matrices_;

  TF_DISALLOW_COPY_AND_ASSIGN(SvdOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_SVD_OP_IMPL_H_
