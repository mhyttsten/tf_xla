/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_BATCH_MATMUL_HELPER_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_BATCH_MATMUL_HELPER_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_batch_matmul_helperDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_batch_matmul_helperDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_batch_matmul_helperDTh() {
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

#ifdef INTEL_MKL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace tensorflow {

struct MklBatchMatMulHelper {
  using dims = dnnl::memory::dims;
  // This method makes the rank (ndims) of input same as the output by creating
  // new axes to the input. For example, if input shape is [a, b, c, d] and
  // output shape is [e, f, g, h, i, j], then the reshaped input would have a
  // shape of [1, 1, a, b, c, d].
  void ExpandInputDimsToOutputShape(const TensorShape& input_shape,
                                    const TensorShape& output_shape,
                                    dims* reshaped_dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_batch_matmul_helperDTh mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/mkl/mkl_batch_matmul_helper.h", "ExpandInputDimsToOutputShape");

    auto ndims_input = input_shape.dims();
    auto ndims_output = output_shape.dims();
    auto dim_offset = ndims_output - ndims_input;
    DCHECK(dim_offset > 0);
    reshaped_dims->clear();
    reshaped_dims->resize(ndims_output, 1);
    auto input_dims = input_shape.dim_sizes();
    for (int dim_idx = 0; dim_idx < ndims_input; ++dim_idx)
      reshaped_dims->at(dim_idx + dim_offset) = input_dims[dim_idx];
  }

  std::unique_ptr<MklMatMulParams> CreateMatMulParams(
      const TensorShape& lhs_shape, const TensorShape& rhs_shape,
      const TensorShape& out_shape, bool& adj_x, bool& adj_y) {
    const auto ndims_lhs = lhs_shape.dims();
    const auto ndims_rhs = rhs_shape.dims();
    const auto ndims_out = out_shape.dims();
    auto lhs_dims = TFShapeToMklDnnDims(lhs_shape);
    auto rhs_dims = TFShapeToMklDnnDims(rhs_shape);
    auto out_dims = TFShapeToMklDnnDims(out_shape);

    // DNNL matmul_primitive requires ranks of inputs and output to be same.
    // Create dnnl::memory::dims for inputs and output of same rank.
    // It is assumed here that MatMulBCast object creates output_batch_shape as
    // a conforming superset of input batch shapes, i.e., ndims_out >=
    // ndims_lhs and ndims_out >= ndims_rhs.
    if (ndims_lhs < ndims_out) {
      ExpandInputDimsToOutputShape(lhs_shape, out_shape, &lhs_dims);
    }
    if (ndims_rhs < ndims_out) {
      ExpandInputDimsToOutputShape(rhs_shape, out_shape, &rhs_dims);
    }
    using dim = dnnl::memory::dim;
    dim m;  // Number of rows in x
    dim k;  // Number of columns in x
    dim n;  // Number of columns in y
    auto lhs_strides = CalculateTFStrides(lhs_dims);
    auto rhs_strides = CalculateTFStrides(rhs_dims);
    auto out_strides = CalculateTFStrides(out_dims);

    if (adj_x) {
      int m_idx = ndims_out - 1;
      int k_idx = ndims_out - 2;
      m = lhs_dims[m_idx];
      k = lhs_dims[k_idx];
      std::swap(lhs_dims[m_idx], lhs_dims[k_idx]);
      lhs_strides[m_idx] = m;
      lhs_strides[k_idx] = 1;
    }

    if (adj_y) {
      int k_idx = ndims_out - 1;
      int n_idx = ndims_out - 2;
      k = rhs_dims[k_idx];
      n = rhs_dims[n_idx];
      std::swap(rhs_dims[k_idx], rhs_dims[n_idx]);
      rhs_strides[k_idx] = k;
      rhs_strides[n_idx] = 1;
    }

    return std::make_unique<MklMatMulParams>(
        lhs_dims, rhs_dims, out_dims, lhs_strides, rhs_strides, out_strides);
  }
};

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_BATCH_MATMUL_HELPER_H_
