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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSdata_typeDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSdata_typeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSdata_typeDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/data_type.h"

#include <stddef.h>

#include <string>

#include "absl/strings/str_cat.h"

namespace tflite {
namespace gpu {
namespace {
std::string ToGlslType(const std::string& scalar_type,
                       const std::string& vec_type, int vec_size) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("scalar_type: \"" + scalar_type + "\"");
   mht_0_v.push_back("vec_type: \"" + vec_type + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSdata_typeDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/delegates/gpu/common/data_type.cc", "ToGlslType");

  return vec_size == 1 ? scalar_type : absl::StrCat(vec_type, vec_size);
}

std::string GetGlslPrecisionModifier(DataType data_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSdata_typeDTcc mht_1(mht_1_v, 206, "", "./tensorflow/lite/delegates/gpu/common/data_type.cc", "GetGlslPrecisionModifier");

  switch (data_type) {
    case DataType::UINT8:
    case DataType::INT8:
      return "lowp ";
    case DataType::FLOAT16:
    case DataType::INT16:
    case DataType::UINT16:
      return "mediump ";
    case DataType::FLOAT32:
    case DataType::INT32:
    case DataType::UINT32:
      return "highp ";
    default:
      return "";
  }
}
}  // namespace

size_t SizeOf(DataType data_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSdata_typeDTcc mht_2(mht_2_v, 228, "", "./tensorflow/lite/delegates/gpu/common/data_type.cc", "SizeOf");

  switch (data_type) {
    case DataType::UINT8:
    case DataType::INT8:
      return 1;
    case DataType::FLOAT16:
    case DataType::INT16:
    case DataType::UINT16:
      return 2;
    case DataType::FLOAT32:
    case DataType::INT32:
    case DataType::UINT32:
      return 4;
    case DataType::FLOAT64:
    case DataType::INT64:
    case DataType::UINT64:
      return 8;
    case DataType::UNKNOWN:
      return 0;
  }
  return 0;
}

std::string ToString(DataType data_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSdata_typeDTcc mht_3(mht_3_v, 254, "", "./tensorflow/lite/delegates/gpu/common/data_type.cc", "ToString");

  switch (data_type) {
    case DataType::FLOAT16:
      return "float16";
    case DataType::FLOAT32:
      return "float32";
    case DataType::FLOAT64:
      return "float64";
    case DataType::INT16:
      return "int16";
    case DataType::INT32:
      return "int32";
    case DataType::INT64:
      return "int64";
    case DataType::INT8:
      return "int8";
    case DataType::UINT16:
      return "uint16";
    case DataType::UINT32:
      return "uint32";
    case DataType::UINT64:
      return "uint64";
    case DataType::UINT8:
      return "uint8";
    case DataType::UNKNOWN:
      return "unknown";
  }
  return "undefined";
}

std::string ToCLDataType(DataType data_type, int vec_size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSdata_typeDTcc mht_4(mht_4_v, 287, "", "./tensorflow/lite/delegates/gpu/common/data_type.cc", "ToCLDataType");

  const std::string postfix = vec_size == 1 ? "" : std::to_string(vec_size);
  switch (data_type) {
    case DataType::FLOAT16:
      return "half" + postfix;
    case DataType::FLOAT32:
      return "float" + postfix;
    case DataType::FLOAT64:
      return "double" + postfix;
    case DataType::INT16:
      return "short" + postfix;
    case DataType::INT32:
      return "int" + postfix;
    case DataType::INT64:
      return "long" + postfix;
    case DataType::INT8:
      return "char" + postfix;
    case DataType::UINT16:
      return "ushort" + postfix;
    case DataType::UINT32:
      return "uint" + postfix;
    case DataType::UINT64:
      return "ulong" + postfix;
    case DataType::UINT8:
      return "uchar" + postfix;
    case DataType::UNKNOWN:
      return "unknown";
  }
  return "undefined";
}

std::string ToMetalDataType(DataType data_type, int vec_size) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSdata_typeDTcc mht_5(mht_5_v, 321, "", "./tensorflow/lite/delegates/gpu/common/data_type.cc", "ToMetalDataType");

  const std::string postfix = vec_size == 1 ? "" : std::to_string(vec_size);
  switch (data_type) {
    case DataType::FLOAT16:
      return "half" + postfix;
    case DataType::FLOAT32:
      return "float" + postfix;
    case DataType::FLOAT64:
      return "double" + postfix;
    case DataType::INT16:
      return "short" + postfix;
    case DataType::INT32:
      return "int" + postfix;
    case DataType::INT64:
      return "long" + postfix;
    case DataType::INT8:
      return "char" + postfix;
    case DataType::UINT16:
      return "ushort" + postfix;
    case DataType::UINT32:
      return "uint" + postfix;
    case DataType::UINT64:
      return "ulong" + postfix;
    case DataType::UINT8:
      return "uchar" + postfix;
    case DataType::UNKNOWN:
      return "unknown";
  }
  return "undefined";
}

DataType ToMetalTextureType(DataType data_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSdata_typeDTcc mht_6(mht_6_v, 355, "", "./tensorflow/lite/delegates/gpu/common/data_type.cc", "ToMetalTextureType");

  switch (data_type) {
    case DataType::FLOAT32:
    case DataType::FLOAT16:
    case DataType::INT32:
    case DataType::INT16:
    case DataType::UINT32:
    case DataType::UINT16:
      return data_type;
    case DataType::INT8:
      return DataType::INT16;
    case DataType::UINT8:
      return DataType::UINT16;
    default:
      return DataType::UNKNOWN;
  }
}

std::string ToGlslShaderDataType(DataType data_type, int vec_size,
                                 bool add_precision, bool explicit_fp16) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSdata_typeDTcc mht_7(mht_7_v, 377, "", "./tensorflow/lite/delegates/gpu/common/data_type.cc", "ToGlslShaderDataType");

  const std::string precision_modifier =
      add_precision ? GetGlslPrecisionModifier(data_type) : "";
  switch (data_type) {
    case DataType::FLOAT16:
      if (explicit_fp16) {
        return ToGlslType("float16_t", "f16vec", vec_size);
      } else {
        return precision_modifier + ToGlslType("float", "vec", vec_size);
      }
    case DataType::FLOAT32:
      return precision_modifier + ToGlslType("float", "vec", vec_size);
    case DataType::FLOAT64:
      return precision_modifier + ToGlslType("double", "dvec", vec_size);
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
    case DataType::INT64:
      return precision_modifier + ToGlslType("int", "ivec", vec_size);
    case DataType::UINT8:
    case DataType::UINT16:
    case DataType::UINT32:
    case DataType::UINT64:
      return precision_modifier + ToGlslType("uint", "uvec", vec_size);
    case DataType::UNKNOWN:
      return "unknown";
  }
  return "unknown";
}

}  // namespace gpu
}  // namespace tflite
