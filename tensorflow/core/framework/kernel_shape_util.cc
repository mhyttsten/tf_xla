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
class MHTracer_DTPStensorflowPScorePSframeworkPSkernel_shape_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_shape_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSkernel_shape_utilDTcc() {
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
#include "tensorflow/core/framework/kernel_shape_util.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
Status GetWindowedOutputSizeVerboseV2(int64_t input_size, int64_t filter_size,
                                      int64_t dilation_rate, int64_t stride,
                                      Padding padding_type,
                                      int64_t* output_size,
                                      int64_t* padding_before,
                                      int64_t* padding_after) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_shape_utilDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/framework/kernel_shape_util.cc", "GetWindowedOutputSizeVerboseV2");

  if (stride <= 0) {
    return errors::InvalidArgument("Stride must be > 0, but got ", stride);
  }
  if (dilation_rate < 1) {
    return errors::InvalidArgument("Dilation rate must be >= 1, but got ",
                                   dilation_rate);
  }

  // See also the parallel implementation in GetWindowedOutputSizeFromDimsV2.
  int64_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding_type) {
    case Padding::VALID:
      *output_size = (input_size - effective_filter_size + stride) / stride;
      *padding_before = *padding_after = 0;
      break;
    case Padding::EXPLICIT:
      *output_size = (input_size + *padding_before + *padding_after -
                      effective_filter_size + stride) /
                     stride;
      break;
    case Padding::SAME:
      *output_size = (input_size + stride - 1) / stride;
      const int64_t padding_needed =
          std::max(int64_t{0}, (*output_size - 1) * stride +
                                   effective_filter_size - input_size);
      // For odd values of total padding, add more padding at the 'right'
      // side of the given dimension.
      *padding_before = padding_needed / 2;
      *padding_after = padding_needed - *padding_before;
      break;
  }
  if (*output_size < 0) {
    return errors::InvalidArgument(
        "Computed output size would be negative: ", *output_size,
        " [input_size: ", input_size,
        ", effective_filter_size: ", effective_filter_size,
        ", stride: ", stride, "]");
  }
  return Status::OK();
}

Status GetWindowedOutputSizeVerbose(int64_t input_size, int64_t filter_size,
                                    int64_t stride, Padding padding_type,
                                    int64_t* output_size,
                                    int64_t* padding_before,
                                    int64_t* padding_after) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_shape_utilDTcc mht_1(mht_1_v, 243, "", "./tensorflow/core/framework/kernel_shape_util.cc", "GetWindowedOutputSizeVerbose");

  return GetWindowedOutputSizeVerboseV2(input_size, filter_size,
                                        /*dilation_rate=*/1, stride,
                                        padding_type, output_size,
                                        padding_before, padding_after);
}

Status GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                             int64_t stride, Padding padding_type,
                             int64_t* output_size, int64_t* padding_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_shape_utilDTcc mht_2(mht_2_v, 255, "", "./tensorflow/core/framework/kernel_shape_util.cc", "GetWindowedOutputSize");

  if (padding_type == Padding::EXPLICIT) {
    return errors::Internal(
        "GetWindowedOutputSize does not handle EXPLICIT padding; call "
        "GetWindowedOutputSizeVerbose instead");
  }
  int64_t padding_after_unused;
  return GetWindowedOutputSizeVerbose(input_size, filter_size, stride,
                                      padding_type, output_size, padding_size,
                                      &padding_after_unused);
}

Status GetWindowedOutputSizeV2(int64_t input_size, int64_t filter_size,
                               int64_t dilation_rate, int64_t stride,
                               Padding padding_type, int64_t* output_size,
                               int64_t* padding_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_shape_utilDTcc mht_3(mht_3_v, 273, "", "./tensorflow/core/framework/kernel_shape_util.cc", "GetWindowedOutputSizeV2");

  if (padding_type == Padding::EXPLICIT) {
    return errors::Internal(
        "GetWindowedOutputSizeV2 does not handle EXPLICIT padding; call "
        "GetWindowedOutputSizeVerboseV2 instead");
  }
  int64_t padding_after_unused;
  return GetWindowedOutputSizeVerboseV2(input_size, filter_size, dilation_rate,
                                        stride, padding_type, output_size,
                                        padding_size, &padding_after_unused);
}

Status Get3dOutputSize(const std::array<int64_t, 3>& input,
                       const std::array<int64_t, 3>& window,
                       const std::array<int64_t, 3>& strides,
                       Padding padding_type, std::array<int64_t, 3>* output_ptr,
                       std::array<int64_t, 3>* padding_ptr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_shape_utilDTcc mht_4(mht_4_v, 292, "", "./tensorflow/core/framework/kernel_shape_util.cc", "Get3dOutputSize");

  for (size_t i = 0; i < input.size(); ++i) {
    TF_RETURN_IF_ERROR(GetWindowedOutputSize(input[i], window[i], strides[i],
                                             padding_type, &(*output_ptr)[i],
                                             &(*padding_ptr)[i]));
  }
  return Status::OK();
}

Status Get3dOutputSizeV2(const std::array<int64_t, 3>& input,
                         const std::array<int64_t, 3>& window,
                         const std::array<int64_t, 3>& dilations,
                         const std::array<int64_t, 3>& strides,
                         Padding padding_type,
                         std::array<int64_t, 3>* output_ptr,
                         std::array<int64_t, 3>* padding_ptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_shape_utilDTcc mht_5(mht_5_v, 310, "", "./tensorflow/core/framework/kernel_shape_util.cc", "Get3dOutputSizeV2");

  for (size_t i = 0; i < input.size(); ++i) {
    TF_RETURN_IF_ERROR(GetWindowedOutputSizeV2(
        input[i], window[i], dilations[i], strides[i], padding_type,
        &(*output_ptr)[i], &(*padding_ptr)[i]));
  }
  return Status::OK();
}
}  // namespace tensorflow
