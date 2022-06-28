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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/task/util.h"

#include <cfloat>
#include <string>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {
std::string GetGlslConversion(const GpuInfo& gpu_info, DataType src_type,
                              DataType dst_type, int vec_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "GetGlslConversion");

  if (src_type == dst_type) {
    return "";
  }
  bool need_explicit_conversion = true;
  switch (dst_type) {
    case DataType::FLOAT32:
    case DataType::FLOAT16:
      if (gpu_info.IsGlslSupportsExplicitFp16()) {
        if (src_type == dst_type) {
          need_explicit_conversion = false;
        }
      } else {
        if (src_type == DataType::FLOAT32 || src_type == DataType::FLOAT16) {
          need_explicit_conversion = false;
        }
      }
      break;
    case DataType::INT32:
    case DataType::INT16:
    case DataType::INT8:
      if (src_type == DataType::INT32 || src_type == DataType::INT16 ||
          src_type == DataType::INT8) {
        need_explicit_conversion = false;
      }
      break;
    case DataType::UINT32:
    case DataType::UINT16:
    case DataType::UINT8:
      if (src_type == DataType::UINT32 || src_type == DataType::UINT16 ||
          src_type == DataType::UINT8) {
        need_explicit_conversion = false;
      }
      break;
    default:
      break;
  }
  if (need_explicit_conversion) {
    return ToGlslShaderDataType(
        dst_type, vec_size,
        /*add_precision*/ false,
        /*explicit_fp16*/ gpu_info.IsGlslSupportsExplicitFp16());
  } else {
    return "";
  }
}
}  // namespace

std::string MemoryTypeToCLType(MemoryType type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_1(mht_1_v, 248, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "MemoryTypeToCLType");

  switch (type) {
    case MemoryType::GLOBAL:
      return "__global";
    case MemoryType::CONSTANT:
      return "__constant";
    case MemoryType::LOCAL:
      return "__local";
  }
  return "";
}

std::string MemoryTypeToMetalType(MemoryType type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_2(mht_2_v, 263, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "MemoryTypeToMetalType");

  switch (type) {
    case MemoryType::GLOBAL:
      return "device";
    case MemoryType::CONSTANT:
      return "constant";
      break;
    case MemoryType::LOCAL:
      return "threadgroup";
  }
  return "";
}

std::string GetXStrideCorrected(const std::string& src_x,
                                const std::string& batch_size,
                                const std::string& stride_x,
                                const std::string& padding_x) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("src_x: \"" + src_x + "\"");
   mht_3_v.push_back("batch_size: \"" + batch_size + "\"");
   mht_3_v.push_back("stride_x: \"" + stride_x + "\"");
   mht_3_v.push_back("padding_x: \"" + padding_x + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_3(mht_3_v, 286, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "GetXStrideCorrected");

  // int p0 = src_x / batch_size;\n";
  // int b0 = src_x % batch_size;\n";
  // return p0 * stride_x * batch_size + b0 + padding_x;\n";
  return absl::Substitute("((($0) / $1) * $2 * $1 + (($0) % $1) + $3)", src_x,
                          batch_size, stride_x, padding_x);
}

std::string GetXStrideCorrectedV2(const std::string& src_x,
                                  const std::string& batch_size,
                                  const std::string& stride_x,
                                  const std::string& padding_x) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("src_x: \"" + src_x + "\"");
   mht_4_v.push_back("batch_size: \"" + batch_size + "\"");
   mht_4_v.push_back("stride_x: \"" + stride_x + "\"");
   mht_4_v.push_back("padding_x: \"" + padding_x + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_4(mht_4_v, 304, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "GetXStrideCorrectedV2");

  // int p0 = src_x / batch_size;\n";
  // int b0 = src_x % batch_size;\n";
  // return (p0 * stride_x + padding_x) * batch_size + b0;\n";
  return absl::Substitute("(((($0) / $1) * $2 + $3) * $1 + ($0) % $1)", src_x,
                          batch_size, stride_x, padding_x);
}

float4 GetMaskForLastPlane(int channels) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_5(mht_5_v, 315, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "GetMaskForLastPlane");

  float4 mask = float4(0.0f);
  const int reminder = channels % 4 == 0 ? 4 : channels % 4;
  for (int i = 0; i < reminder; ++i) {
    mask[i] = 1.0f;
  }
  return mask;
}

int GetRecommendedBlockSizeForConv(const GpuInfo& gpu_info,
                                   CalculationsPrecision precision,
                                   int task_size) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_6(mht_6_v, 329, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "GetRecommendedBlockSizeForConv");

  const float task_size_per_cu =
      task_size / static_cast<float>(gpu_info.GetComputeUnitsCount());
  int block_size = 1;
  float threshold_1 = FLT_MAX;
  float threshold_2 = FLT_MAX;
  float threshold_4 = FLT_MAX;
  if (!gpu_info.IsMali()) {
    return 1;
  }
  MaliInfo mali_info = gpu_info.mali_info;
  switch (precision) {
    case CalculationsPrecision::F16:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
        threshold_4 = 256.0f * 8.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
        threshold_4 = 256.0f * 16.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 6.0f;
        threshold_4 = 256.0f * 16.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 4.0f;
        threshold_2 = 256.0f * 16.0f;
      }
      break;
    case CalculationsPrecision::F32_F16:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 3.0f;
        threshold_4 = 256.0f * 32.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 8.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 4.0f;
      }
      break;
    case CalculationsPrecision::F32:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 128.0f;
        threshold_2 = 256.0f * 4.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 12.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 16.0f;
      }
      break;
  }
  if (task_size_per_cu <= threshold_1) {
    block_size = 1;
  } else if (task_size_per_cu <= threshold_2) {
    block_size = 2;
  } else if (task_size_per_cu <= threshold_4) {
    block_size = 4;
  } else {
    block_size = 8;
  }
  return block_size;
}

int3 GetWorkGroupsCount(const int3& grid_size, const int3& work_group_size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_7(mht_7_v, 404, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "GetWorkGroupsCount");

  int3 work_groups_count;
  work_groups_count.x = DivideRoundUp(grid_size.x, work_group_size.x);
  work_groups_count.y = DivideRoundUp(grid_size.y, work_group_size.y);
  work_groups_count.z = DivideRoundUp(grid_size.z, work_group_size.z);
  return work_groups_count;
}

std::string GetTypeDeclaration(const GpuInfo& gpu_info, DataType data_type,
                               int vec_size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_8(mht_8_v, 416, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "GetTypeDeclaration");

  if (gpu_info.IsApiOpenCl()) {
    return ToCLDataType(data_type, vec_size);
  } else if (gpu_info.IsApiMetal()) {
    return ToMetalDataType(data_type, vec_size);
  } else if (gpu_info.IsGlsl()) {
    return ToGlslShaderDataType(data_type, vec_size, true,
                                gpu_info.IsGlslSupportsExplicitFp16());
  } else {
    return "";
  }
}

std::string GetZeroValue(const GpuInfo& gpu_info, DataType data_type,
                         int vec_size) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_9(mht_9_v, 433, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "GetZeroValue");

  if (gpu_info.IsApiOpenCl()) {
    return "(" + ToCLDataType(data_type, vec_size) + ")(0)";
  } else if (gpu_info.IsApiMetal()) {
    return ToMetalDataType(data_type, vec_size) + "(0)";
  } else if (gpu_info.IsGlsl()) {
    return ToGlslShaderDataType(data_type, vec_size, false,
                                gpu_info.IsGlslSupportsExplicitFp16()) +
           "(0)";
  } else {
    return "";
  }
}

std::string GetOneValue(const GpuInfo& gpu_info, DataType data_type,
                        int vec_size) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_10(mht_10_v, 451, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "GetOneValue");

  if (gpu_info.IsApiOpenCl()) {
    return "(" + ToCLDataType(data_type, vec_size) + ")(1)";
  } else if (gpu_info.IsApiMetal()) {
    return ToMetalDataType(data_type, vec_size) + "(1)";
  } else if (gpu_info.IsGlsl()) {
    return ToGlslShaderDataType(data_type, vec_size, false,
                                gpu_info.IsGlslSupportsExplicitFp16()) +
           "(1)";
  } else {
    return "";
  }
}

std::string GetTypeConvertion(const GpuInfo& gpu_info, DataType src_type,
                              DataType dst_type, int vec_size) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSutilDTcc mht_11(mht_11_v, 469, "", "./tensorflow/lite/delegates/gpu/common/task/util.cc", "GetTypeConvertion");

  if (src_type != dst_type) {
    if (gpu_info.IsApiOpenCl()) {
      return "convert_" + ToCLDataType(dst_type, vec_size);
    } else if (gpu_info.IsApiMetal()) {
      return ToMetalDataType(dst_type, vec_size);
    } else if (gpu_info.IsGlsl()) {
      return GetGlslConversion(gpu_info, src_type, dst_type, vec_size);
    }
  }
  return "";
}

}  // namespace gpu
}  // namespace tflite
