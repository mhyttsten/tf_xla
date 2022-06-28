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
class MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc() {
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

#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

string GetConvnetDataFormatAttrString() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc mht_0(mht_0_v, 189, "", "./tensorflow/core/util/tensor_format.cc", "GetConvnetDataFormatAttrString");

  return "data_format: { 'NHWC', 'NCHW' } = 'NHWC' ";
}

string GetConvnet3dDataFormatAttrString() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc mht_1(mht_1_v, 196, "", "./tensorflow/core/util/tensor_format.cc", "GetConvnet3dDataFormatAttrString");

  return "data_format: { 'NDHWC', 'NCDHW' } = 'NDHWC' ";
}

string GetConvnetDataFormat2D3DAttrString() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc mht_2(mht_2_v, 203, "", "./tensorflow/core/util/tensor_format.cc", "GetConvnetDataFormat2D3DAttrString");

  return "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ";
}

string GetConvnetFilterFormatAttrString() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc mht_3(mht_3_v, 210, "", "./tensorflow/core/util/tensor_format.cc", "GetConvnetFilterFormatAttrString");

  return "filter_format: { 'HWIO', 'OIHW' } = 'HWIO' ";
}

string GetConvnet3dFilterFormatAttrString() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc mht_4(mht_4_v, 217, "", "./tensorflow/core/util/tensor_format.cc", "GetConvnet3dFilterFormatAttrString");

  return "filter_format: { 'DHWIO', 'OIDHW' } = 'DHWIO' ";
}

string ToString(TensorFormat format) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc mht_5(mht_5_v, 224, "", "./tensorflow/core/util/tensor_format.cc", "ToString");

  switch (format) {
    case FORMAT_NHWC:
      return "NHWC";
    case FORMAT_NCHW:
      return "NCHW";
    case FORMAT_NCHW_VECT_C:
      return "NCHW_VECT_C";
    case FORMAT_NHWC_VECT_W:
      return "NHWC_VECT_W";
    case FORMAT_HWNC:
      return "HWNC";
    case FORMAT_HWCN:
      return "HWCN";
    default:
      LOG(FATAL) << "Invalid Format: " << static_cast<int32>(format);
      return "INVALID_FORMAT";
  }
}

string ToString(FilterTensorFormat format) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc mht_6(mht_6_v, 247, "", "./tensorflow/core/util/tensor_format.cc", "ToString");

  switch (format) {
    case FORMAT_HWIO:
      return "HWIO";
    case FORMAT_OIHW:
      return "OIHW";
    case FORMAT_OHWI:
      return "OHWI";
    case FORMAT_OIHW_VECT_I:
      return "OIHW_VECT_I";
    default:
      LOG(FATAL) << "Invalid Filter Format: " << static_cast<int32>(format);
      return "INVALID_FORMAT";
  }
}

bool FormatFromString(absl::string_view format_str, TensorFormat* format) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("format_str: \"" + std::string(format_str.data(), format_str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc mht_7(mht_7_v, 267, "", "./tensorflow/core/util/tensor_format.cc", "FormatFromString");

  if (format_str == "NHWC" || format_str == "NDHWC") {
    *format = FORMAT_NHWC;
    return true;
  }
  if (format_str == "NCHW" || format_str == "NCDHW") {
    *format = FORMAT_NCHW;
    return true;
  }
  if (format_str == "NCHW_VECT_C") {
    *format = FORMAT_NCHW_VECT_C;
    return true;
  }
  if (format_str == "NHWC_VECT_W") {
    *format = FORMAT_NHWC_VECT_W;
    return true;
  }
  if (format_str == "HWNC") {
    *format = FORMAT_HWNC;
    return true;
  }
  if (format_str == "HWCN") {
    *format = FORMAT_HWCN;
    return true;
  }
  return false;
}

bool FilterFormatFromString(absl::string_view format_str,
                            FilterTensorFormat* format) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("format_str: \"" + std::string(format_str.data(), format_str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_formatDTcc mht_8(mht_8_v, 300, "", "./tensorflow/core/util/tensor_format.cc", "FilterFormatFromString");

  if (format_str == "HWIO" || format_str == "DHWIO") {
    *format = FORMAT_HWIO;
    return true;
  }
  if (format_str == "OIHW" || format_str == "OIDHW") {
    *format = FORMAT_OIHW;
    return true;
  }
  if (format_str == "OIHW_VECT_I") {
    *format = FORMAT_OIHW_VECT_I;
    return true;
  }
  return false;
}

}  // namespace tensorflow
