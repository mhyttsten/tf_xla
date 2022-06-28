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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_TRANSPOSE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_TRANSPOSE_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePStransposeDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePStransposeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePStransposeDTh() {
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


#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <typename T, int N>
void TransposeImpl(const TransposeParams& params,
                   const RuntimeShape& unextended_input_shape,
                   const T* input_data,
                   const RuntimeShape& unextended_output_shape,
                   T* output_data) {
  const int unextended_input_size = unextended_input_shape.DimensionsCount();
  const int unextended_output_size = unextended_output_shape.DimensionsCount();
  TFLITE_DCHECK_LE(unextended_input_size, N);
  TFLITE_DCHECK_LE(unextended_output_size, N);
  TFLITE_DCHECK_EQ(unextended_output_size, params.perm_count);
  const int input_ext_size = N - unextended_input_size;
  const int output_ext_size = N - unextended_output_size;
  NdArrayDesc<N> input_desc;
  NdArrayDesc<N> output_desc;
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_input_shape),
                 &input_desc);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_output_shape),
                 &output_desc);

  // The perm data is extended to match the output, each index incremented by
  // the amount of front padding of the input shape.
  int extended_perm[N];
  for (int i = 0; i < N; ++i) {
    extended_perm[i] = i < output_ext_size
                           ? i
                           : params.perm[i - output_ext_size] + input_ext_size;
  }

  // Permutes the input shape so we don't need to permute the indexes inside
  // the loop. Check to make sure output_dims is matching input_dims.
  NdArrayDesc<N> perm_input_desc;
  for (int k = 0; k < N; ++k) {
    TFLITE_DCHECK_EQ(input_desc.extents[extended_perm[k]],
                     output_desc.extents[k]);
    perm_input_desc.extents[k] = input_desc.extents[extended_perm[k]];
    perm_input_desc.strides[k] = input_desc.strides[extended_perm[k]];
  }

  // Naive transpose loop (iterate on output index and compute input index).
  auto tranpose_func = [&](int indexes[N]) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePStransposeDTh mht_0(mht_0_v, 234, "", "./tensorflow/lite/kernels/internal/reference/transpose.h", "lambda");

    output_data[SubscriptToIndex(output_desc, indexes)] =
        input_data[SubscriptToIndex(perm_input_desc, indexes)];
  };
  NDOpsHelper<N>(output_desc, tranpose_func);
}

template <typename T, int N = 5>
void Transpose(const TransposeParams& params,
               const RuntimeShape& unextended_input_shape, const T* input_data,
               const RuntimeShape& unextended_output_shape, T* output_data) {
  // Transpose kernel only does rearranging values not numeric evaluations on
  // each cell. It's safe to implement per size of scalar type and this trick
  // keeps the total code size in a reasonable range.
  switch (sizeof(T)) {
    case 1:
      TransposeImpl<int8_t, N>(params, unextended_input_shape,
                               reinterpret_cast<const int8_t*>(input_data),
                               unextended_output_shape,
                               reinterpret_cast<int8_t*>(output_data));
      break;
    case 2:
      TransposeImpl<int16_t, N>(params, unextended_input_shape,
                                reinterpret_cast<const int16_t*>(input_data),
                                unextended_output_shape,
                                reinterpret_cast<int16_t*>(output_data));
      break;

    case 4:
      TransposeImpl<int32_t, N>(params, unextended_input_shape,
                                reinterpret_cast<const int32_t*>(input_data),
                                unextended_output_shape,
                                reinterpret_cast<int32_t*>(output_data));
      break;
    case 8:
      TransposeImpl<int64_t, N>(params, unextended_input_shape,
                                reinterpret_cast<const int64_t*>(input_data),
                                unextended_output_shape,
                                reinterpret_cast<int64_t*>(output_data));
      break;
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_TRANSPOSE_H_
