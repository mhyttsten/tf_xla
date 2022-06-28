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
class MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_eigenDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_eigenDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_eigenDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TFLITE_WITH_RUY

#include "tensorflow/lite/kernels/cpu_backend_gemm_eigen.h"

// See b/131835803: in TFLite code, because eigen_spatial_convolutions.h does
// #define Eigen EigenForTFLite, it is difficult to have any #include of Eigen
// headers in a header file, as that results in name classes (compilation
// errors) depending on the order in which these headers are #included.
// So we have moved the #include of Eigen here, in a .cc file, where we have
// control over the header #include sequence.
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

// This function is out-of-line in a .cc file because of the issue
// noted above in the comment on the #include for Eigen/Core.
void GemmImplUsingEigen::Run(
    const MatrixParams<float>& lhs_params, const float* lhs_data,
    const MatrixParams<float>& rhs_params, const float* rhs_data,
    const MatrixParams<float>& dst_params, float* dst_data,
    const GemmParams<float, float>& params, CpuBackendContext* /* context */) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_eigenDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/kernels/cpu_backend_gemm_eigen.cc", "GemmImplUsingEigen::Run");

  // This code assumes specific storage orders, encoded in these Eigen types.
  // These assumptions have been checked by TF_LITE_ASSERT's in the public
  // Gemm entry point already, before the implementation gets to this point.
  using EigenMatrixMapRowMajorConst =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>;
  using EigenMatrixMapColMajorConst =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::ColMajor>>;
  using EigenMatrixMapColMajorMutable = Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;

  EigenMatrixMapRowMajorConst eigen_lhs(lhs_data, lhs_params.rows,
                                        lhs_params.cols);
  EigenMatrixMapColMajorConst eigen_rhs(rhs_data, rhs_params.rows,
                                        rhs_params.cols);
  EigenMatrixMapColMajorMutable eigen_dst(dst_data, dst_params.rows,
                                          dst_params.cols);

  if (rhs_params.cols == 1) {
    eigen_dst.col(0).noalias() = eigen_lhs * eigen_rhs.col(0);
  } else if (lhs_params.rows == 1) {
    eigen_dst.row(0).noalias() = eigen_lhs.row(0) * eigen_rhs;
  } else {
    eigen_dst.noalias() = eigen_lhs * eigen_rhs;
  }

  if (params.bias) {
    BiasAndClamp(params.clamp_min, params.clamp_max, dst_params.rows,
                 params.bias, dst_params.rows * dst_params.cols, dst_data);
  } else {
    eigen_dst = eigen_dst.cwiseMin(params.clamp_max).cwiseMax(params.clamp_min);
  }
}

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // not TFLITE_WITH_RUY
