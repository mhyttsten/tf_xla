/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_PADDING_H_
#define TENSORFLOW_LITE_KERNELS_PADDING_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSpaddingDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSpaddingDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSpaddingDTh() {
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


#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

inline int ComputePadding(int stride, int dilation_rate, int in_size,
                          int filter_size, int out_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpaddingDTh mht_0(mht_0_v, 193, "", "./tensorflow/lite/kernels/padding.h", "ComputePadding");

  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  int padding = ((out_size - 1) * stride + effective_filter_size - in_size) / 2;
  return padding > 0 ? padding : 0;
}

// It's not guaranteed that padding is symmetric. It's important to keep
// offset for algorithms need all paddings.
inline int ComputePaddingWithOffset(int stride, int dilation_rate, int in_size,
                                    int filter_size, int out_size,
                                    int* offset) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpaddingDTh mht_1(mht_1_v, 206, "", "./tensorflow/lite/kernels/padding.h", "ComputePaddingWithOffset");

  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  int total_padding =
      ((out_size - 1) * stride + effective_filter_size - in_size);
  total_padding = total_padding > 0 ? total_padding : 0;
  *offset = total_padding % 2;
  return total_padding / 2;
}

// Matching GetWindowedOutputSize in TensorFlow.
inline int ComputeOutSize(TfLitePadding padding, int image_size,
                          int filter_size, int stride, int dilation_rate = 1) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpaddingDTh mht_2(mht_2_v, 220, "", "./tensorflow/lite/kernels/padding.h", "ComputeOutSize");

  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;

  // TODO(b/186448822): This uses 0 since the function has no other way to
  // report error case
  if (stride == 0) return 0;

  switch (padding) {
    case kTfLitePaddingSame:
      return (image_size + stride - 1) / stride;
    case kTfLitePaddingValid:
      return (image_size + stride - effective_filter_size) / stride;
    default:
      return 0;
  }
}

inline TfLitePaddingValues ComputePaddingHeightWidth(
    int stride_height, int stride_width, int dilation_rate_height,
    int dilation_rate_width, int in_height, int in_width, int filter_height,
    int filter_width, TfLitePadding padding, int* out_height, int* out_width) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpaddingDTh mht_3(mht_3_v, 243, "", "./tensorflow/lite/kernels/padding.h", "ComputePaddingHeightWidth");

  *out_width = ComputeOutSize(padding, in_width, filter_width, stride_width,
                              dilation_rate_width);
  *out_height = ComputeOutSize(padding, in_height, filter_height, stride_height,
                               dilation_rate_height);

  TfLitePaddingValues padding_values;
  int offset = 0;
  padding_values.height =
      ComputePaddingWithOffset(stride_height, dilation_rate_height, in_height,
                               filter_height, *out_height, &offset);
  padding_values.height_offset = offset;
  padding_values.width =
      ComputePaddingWithOffset(stride_width, dilation_rate_width, in_width,
                               filter_width, *out_width, &offset);
  padding_values.width_offset = offset;
  return padding_values;
}

inline Padding3DValues ComputePadding3DValues(
    int stride_height, int stride_width, int stride_depth,
    int dilation_rate_height, int dilation_rate_width, int dilation_rate_depth,
    int in_height, int in_width, int in_depth, int filter_height,
    int filter_width, int filter_depth, TfLitePadding padding, int* out_height,
    int* out_width, int* out_depth) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpaddingDTh mht_4(mht_4_v, 270, "", "./tensorflow/lite/kernels/padding.h", "ComputePadding3DValues");

  *out_width = ComputeOutSize(padding, in_width, filter_width, stride_width,
                              dilation_rate_width);
  *out_height = ComputeOutSize(padding, in_height, filter_height, stride_height,
                               dilation_rate_height);
  *out_depth = ComputeOutSize(padding, in_depth, filter_depth, stride_depth,
                              dilation_rate_depth);

  Padding3DValues padding_values;
  int offset = 0;
  padding_values.depth =
      ComputePaddingWithOffset(stride_depth, dilation_rate_depth, in_depth,
                               filter_depth, *out_depth, &offset);
  padding_values.depth_offset = offset;
  padding_values.height =
      ComputePaddingWithOffset(stride_height, dilation_rate_height, in_height,
                               filter_height, *out_height, &offset);
  padding_values.height_offset = offset;
  padding_values.width =
      ComputePaddingWithOffset(stride_width, dilation_rate_width, in_width,
                               filter_width, *out_width, &offset);
  padding_values.width_offset = offset;
  return padding_values;
}
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_PADDING_H_
