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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSwinograd_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSwinograd_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSwinograd_utilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

#include <cmath>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {
// Matrices for Winograd trasformations were computed with the method described
// here https://openreview.net/pdf?id=H1ZaRZVKg
std::vector<float> GetTransposedMatrixForWinograd(int width, int height) {
  const float kDelta = std::sqrt(2.0f) / 2.0f;
  std::vector<float> px(width);

  px[0] = 0.0f;
  const int points_count = (width - 1) / 2;
  for (int i = 0; i < points_count; ++i) {
    px[i * 2 + 1] = kDelta * (i + 1.0f);
    px[i * 2 + 2] = -kDelta * (i + 1.0f);
  }
  px[width - 1] = 1.0f;

  std::vector<float> py(width, 1.0f);
  py[width - 1] = 0.0f;

  std::vector<float> result(height * width);
  for (int y = 0; y < width; ++y) {
    for (int x = 0; x < height; ++x) {
      result[x * width + y] =
          std::pow(px[y], 1.0f * x) * std::pow(py[y], (height - 1.0f) - x);
    }
  }
  return result;
}

std::vector<float> GetInversedMatrixForWinograd(int rank) {
  auto matrix = GetTransposedMatrixForWinograd(rank, rank);
  std::vector<float> inverted(rank * rank, 0.0f);
  for (int i = 0; i < rank; ++i) {
    inverted[i * rank + i] = 1.0f;
  }

  for (int i = 1; i < rank - 1; ++i) {
    float inv_t = 1.0f / matrix[i * rank + i];
    for (int x = i; x < rank; ++x) {
      matrix[i * rank + x] *= inv_t;
    }
    for (int x = 0; x < rank; ++x) {
      inverted[i * rank + x] *= inv_t;
    }

    for (int y = 0; y < rank; ++y) {
      if (y == i) continue;
      float t = matrix[y * rank + i];
      for (int x = i; x < rank; ++x) {
        matrix[y * rank + x] -= t * matrix[i * rank + x];
      }
      for (int x = 0; x < rank; ++x) {
        inverted[y * rank + x] -= t * inverted[i * rank + x];
      }
    }
  }

  return inverted;
}

std::vector<float> Multiply(const std::vector<float>& a_mat,
                            const std::vector<float>& b_mat, int m, int n,
                            int k) {
  std::vector<float> result(m * k);
  for (int y = 0; y < m; ++y) {
    for (int x = 0; x < k; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < n; ++i) {
        sum += a_mat[y * n + i] * b_mat[i * k + x];
      }
      result[y * k + x] = sum;
    }
  }
  return result;
}
}  // namespace

std::vector<float> AtMatrixForWinograd4x4To6x6() {
  return GetTransposedMatrixForWinograd(6, 4);
}

std::vector<float> BtMatrixForWinograd4x4To6x6() {
  return GetInversedMatrixForWinograd(6);
}

void RearrangeWeightsToWinograd4x4To6x6Weights(
    const Tensor<OHWI, DataType::FLOAT32>& src_weights,
    Tensor<OHWI, DataType::FLOAT32>* dst_weights) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSwinograd_utilDTcc mht_0(mht_0_v, 282, "", "./tensorflow/lite/delegates/gpu/common/winograd_util.cc", "RearrangeWeightsToWinograd4x4To6x6Weights");

  OHWI dst_shape;
  dst_shape.o = src_weights.shape.o;
  dst_shape.h = 6;
  dst_shape.w = 6;
  dst_shape.i = src_weights.shape.i;
  dst_weights->shape = dst_shape;
  dst_weights->data.resize(dst_shape.DimensionsProduct());

  auto gt_mat = GetTransposedMatrixForWinograd(6, 3);
  std::vector<float> g_mat(gt_mat.size());
  for (int y = 0; y < 3; ++y) {
    for (int x = 0; x < 6; ++x) {
      g_mat[x * 3 + y] = gt_mat[y * 6 + x];
    }
  }

  for (int d = 0; d < src_weights.shape.o; ++d) {
    for (int s = 0; s < src_weights.shape.i; ++s) {
      std::vector<float> in_vals(9);
      for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
          const int f_index = src_weights.shape.LinearIndex({d, y, x, s});
          in_vals[y * 3 + x] = src_weights.data[f_index];
        }
      }

      auto temp_vals = Multiply(g_mat, in_vals, 6, 3, 3);
      auto out_vals = Multiply(temp_vals, gt_mat, 6, 3, 6);
      for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
          const int f_index = dst_shape.LinearIndex({d, y, x, s});
          dst_weights->data[f_index] = out_vals[y * 6 + x];
        }
      }
    }
  }
}

bool IsSuitableForWinograd4x4To6x6(const Convolution2DAttributes& attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSwinograd_utilDTcc mht_1(mht_1_v, 324, "", "./tensorflow/lite/delegates/gpu/common/winograd_util.cc", "IsSuitableForWinograd4x4To6x6");

  return attr.weights.shape.w == 3 && attr.weights.shape.h == 3 &&
         attr.dilations == HW(1, 1) && attr.strides == HW(1, 1) &&
         attr.groups == 1;
}

}  // namespace gpu
}  // namespace tflite
