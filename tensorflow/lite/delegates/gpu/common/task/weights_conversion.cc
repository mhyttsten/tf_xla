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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSweights_conversionDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSweights_conversionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSweights_conversionDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"

namespace tflite {
namespace gpu {
uint GetTotalElementsCountForLayout(const WeightsDescription& weight_desc,
                                    const OHWDI& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSweights_conversionDTcc mht_0(mht_0_v, 190, "", "./tensorflow/lite/delegates/gpu/common/task/weights_conversion.cc", "GetTotalElementsCountForLayout");

  if (weight_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
      weight_desc.layout == WeightsLayout::kOSpatialIOGroupO4I4 ||
      weight_desc.layout == WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      weight_desc.layout == WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    uint i_aligned = AlignByN(shape.i, 4);
    uint o_aligned = AlignByN(shape.o, 4 * weight_desc.output_group_size);
    return i_aligned * o_aligned * shape.h * shape.w * shape.d;
  } else if (weight_desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
             weight_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    uint i_aligned = AlignByN(shape.i, 4);
    uint o_aligned = AlignByN(shape.o, 4);
    return i_aligned * o_aligned * weight_desc.spatial_remap.size();
  } else {
    return -1;
  }
}

uint GetTotalElementsCountForLayout(const WeightsDescription& weight_desc,
                                    const OHWI& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSweights_conversionDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/delegates/gpu/common/task/weights_conversion.cc", "GetTotalElementsCountForLayout");

  const OHWDI ohwdi_shape = OHWDI(shape.o, shape.h, shape.w, 1, shape.i);
  return GetTotalElementsCountForLayout(weight_desc, ohwdi_shape);
}

uint2 Get2dResourceSize(const WeightsDescription& weight_desc,
                        const OHWI& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSweights_conversionDTcc mht_2(mht_2_v, 221, "", "./tensorflow/lite/delegates/gpu/common/task/weights_conversion.cc", "Get2dResourceSize");

  const OHWDI ohwdi_shape = OHWDI(shape.o, shape.h, shape.w, 1, shape.i);
  return Get2dResourceSize(weight_desc, ohwdi_shape);
}

uint2 Get2dResourceSize(const WeightsDescription& weight_desc,
                        const OHWDI& shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSweights_conversionDTcc mht_3(mht_3_v, 230, "", "./tensorflow/lite/delegates/gpu/common/task/weights_conversion.cc", "Get2dResourceSize");

  const int dst_depth =
      AlignByN(DivideRoundUp(shape.o, 4), weight_desc.output_group_size);
  const int src_depth = DivideRoundUp(shape.i, 4);

  return uint2(dst_depth, src_depth * shape.h * shape.w * shape.d);
}

void RearrangeWeights(
    const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSweights_conversionDTcc mht_4(mht_4_v, 243, "", "./tensorflow/lite/delegates/gpu/common/task/weights_conversion.cc", "RearrangeWeights");

  const uint flt_count =
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape);
  if (dst_weight_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOHWIOGroupI4O4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOHWIOGroupI4O4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOSpatialIOGroupO4I4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOHWIOGroupO4I4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOHWIOGroupO4I4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOICustomSpatialI4O4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOICustomSpatialI4O4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOICustomSpatialI4O4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOICustomSpatialO4I4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOICustomSpatialO4I4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout ==
             WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToI4HWIOOGroupO4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToI4HWIOOGroupO4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout ==
             WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToO4HWIOOGroupI4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToO4HWIOOGroupI4(weights,
                                       dst_weight_desc.output_group_size,
                                       absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  }
}

void RearrangeWeights(
    const tflite::gpu::Tensor<OHWDI, DataType::FLOAT32>& weights,
    const WeightsDescription& dst_weight_desc, absl::Span<uint8_t> dst) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSweights_conversionDTcc mht_5(mht_5_v, 334, "", "./tensorflow/lite/delegates/gpu/common/task/weights_conversion.cc", "RearrangeWeights");

  const uint flt_count =
      GetTotalElementsCountForLayout(dst_weight_desc, weights.shape);
  if (dst_weight_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToODHWIOGroupI4O4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToODHWIOGroupI4O4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOSpatialIOGroupO4I4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToODHWIOGroupO4I4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToODHWIOGroupO4I4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOICustomSpatialI4O4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOICustomSpatialI4O4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOICustomSpatialI4O4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToOICustomSpatialO4I4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToOICustomSpatialO4I4(
          weights, dst_weight_desc.spatial_remap,
          absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout ==
             WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToI4DHWIOOGroupO4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToI4DHWIOOGroupO4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  } else if (dst_weight_desc.layout ==
             WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    if (dst_weight_desc.type == DataType::FLOAT32) {
      float4* f32_ptr = reinterpret_cast<float4*>(dst.data());
      RearrangeWeightsToO4DHWIOOGroupI4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f32_ptr, flt_count / 4));
    } else if (dst_weight_desc.type == DataType::FLOAT16) {
      half4* f16_ptr = reinterpret_cast<half4*>(dst.data());
      RearrangeWeightsToO4DHWIOOGroupI4(weights,
                                        dst_weight_desc.output_group_size,
                                        absl::MakeSpan(f16_ptr, flt_count / 4));
    }
    return;
  }
}

}  // namespace gpu
}  // namespace tflite
