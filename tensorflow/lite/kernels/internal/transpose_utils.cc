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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStranspose_utilsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStranspose_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStranspose_utilsDTcc() {
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
#include "tensorflow/lite/kernels/internal/transpose_utils.h"

namespace tflite {
namespace transpose_utils {

bool IsTranspose2DApplicable(const TransposeParams& params,
                             const RuntimeShape& input_shape, int* dim0,
                             int* dim1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStranspose_utilsDTcc mht_0(mht_0_v, 191, "", "./tensorflow/lite/kernels/internal/transpose_utils.cc", "IsTranspose2DApplicable");

  const int dims_cnt = input_shape.DimensionsCount();

  if (dims_cnt == 2) {
    *dim0 = input_shape.Dims(0);
    *dim1 = input_shape.Dims(1);
    return true;
  }

  const int first_perm = params.perm[0];
  for (int i = 1; i < dims_cnt; ++i) {
    int rebased = params.perm[i] - first_perm;
    if (rebased < 0) {
      rebased += dims_cnt;
    }
    if (rebased != i) {
      return false;
    }
  }
  *dim0 = 1;
  *dim1 = 1;
  for (int i = 0; i < dims_cnt; ++i) {
    if (i < first_perm) {
      *dim0 *= input_shape.Dims(i);
    } else {
      *dim1 *= input_shape.Dims(i);
    }
  }
  return true;
}

void RemoveOneSizeDimensions(RuntimeShape* input_shape,
                             RuntimeShape* output_shape,
                             TransposeParams* params) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStranspose_utilsDTcc mht_1(mht_1_v, 227, "", "./tensorflow/lite/kernels/internal/transpose_utils.cc", "RemoveOneSizeDimensions");

  const int dims_cnt = input_shape->DimensionsCount();
  TFLITE_DCHECK_EQ(params->perm_count, dims_cnt);

  bool foundOneSizeDim = false;
  for (int i = 0; i < dims_cnt; ++i) {
    if (input_shape->Dims(i) == 1) {
      foundOneSizeDim = true;
      break;
    }
  }

  // Return here if there is no one size dimension.
  if (!foundOneSizeDim) return;

  // Handle the case where all the dimension size is one.
  if (input_shape->FlatSize() == 1) {
    input_shape->Resize(1);
    input_shape->SetDim(0, 1);
    output_shape->Resize(1);
    output_shape->SetDim(0, 1);
    params->perm_count = 1;
    params->perm[0] = 0;
    return;
  }

  // Resize input shape.
  int new_dims_cnt = 0;
  for (int i = 0; i < dims_cnt; ++i) {
    if (input_shape->Dims(i) == 1) {
      continue;
    }
    input_shape->SetDim(new_dims_cnt, input_shape->Dims(i));
    ++new_dims_cnt;
  }
  input_shape->Resize(new_dims_cnt);

  // Resize output shape and re-calculate the perm parameter.
  TransposeParams new_params;
  new_dims_cnt = 0;
  for (int i = 0; i < dims_cnt; ++i) {
    if (output_shape->Dims(i) == 1) {
      continue;
    }
    new_params.perm[new_dims_cnt] = params->perm[i];
    output_shape->SetDim(new_dims_cnt, output_shape->Dims(i));
    ++new_dims_cnt;
  }
  output_shape->Resize(new_dims_cnt);
  new_params.perm_count = new_dims_cnt;

  for (int i = 0; i < new_dims_cnt; ++i) {
    int min_val_idx = -1;
    for (int j = 0; j < new_dims_cnt; ++j) {
      if (new_params.perm[j] >= i &&
          (min_val_idx == -1 ||
           new_params.perm[min_val_idx] > new_params.perm[j])) {
        min_val_idx = j;
      }
    }
    new_params.perm[min_val_idx] = i;
  }
  *params = new_params;
}

size_t Flatten(const RuntimeShape& input_shape,
               const RuntimeShape& output_shape, const TransposeParams& params,
               RuntimeShape* non_flatten_input_shape,
               RuntimeShape* non_flatten_output_shape,
               TransposeParams* non_flatten_params) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStranspose_utilsDTcc mht_2(mht_2_v, 299, "", "./tensorflow/lite/kernels/internal/transpose_utils.cc", "Flatten");

  // Calculate the total size of non-flatten dimensions.
  int skip_dims_cnt = 0;
  size_t flat_size = input_shape.FlatSize();
  for (int i = 0; i < params.perm_count; ++i) {
    if (params.perm[i] == i) {
      flat_size /= input_shape.Dims(i);
      ++skip_dims_cnt;
    } else {
      break;
    }
  }

  // Shrink the shapes and re-calculate the perm parameter.
  const int new_dims_cnt = params.perm_count - skip_dims_cnt;
  non_flatten_input_shape->Resize(new_dims_cnt);
  non_flatten_output_shape->Resize(new_dims_cnt);
  non_flatten_params->perm_count = new_dims_cnt;

  for (int i = skip_dims_cnt; i < params.perm_count; ++i) {
    non_flatten_input_shape->SetDim(i - skip_dims_cnt, input_shape.Dims(i));
    non_flatten_output_shape->SetDim(i - skip_dims_cnt, output_shape.Dims(i));
    non_flatten_params->perm[i - skip_dims_cnt] = params.perm[i];
  }
  for (int i = 0; i < new_dims_cnt; ++i) {
    int min_val_idx = -1;
    for (int j = 0; j < new_dims_cnt; ++j) {
      if (non_flatten_params->perm[j] >= i &&
          (min_val_idx == -1 || non_flatten_params->perm[min_val_idx] >
                                    non_flatten_params->perm[j])) {
        min_val_idx = j;
      }
    }
    non_flatten_params->perm[min_val_idx] = i;
  }

  return flat_size;
}

}  // namespace transpose_utils

}  // namespace tflite
