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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_SUPPORT_H
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_SUPPORT_H
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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh() {
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


#include <string>

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace optimize {

static constexpr char kTfLiteReducedPrecisionKey[] =
    "reduced_precision_support";

static constexpr char kTfLiteFloat16String[] = "fp16";
static constexpr char kTfLiteBfloat16String[] = "bf16";
static constexpr char kTfLiteFloat32String[] = "fp32";
static constexpr char kTfLiteAccumulationString[] = "acc";

enum class ReducedPrecisionSupport : std::uint8_t {
  None = 0,
  Float16Inference = 0x1,
  Bfloat16Inference = 0x2,
  Float16Accumulation = 0x4,
  Float32Accumulation = 0x8,
};

inline ReducedPrecisionSupport operator|(ReducedPrecisionSupport a,
                                         ReducedPrecisionSupport b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_0(mht_0_v, 211, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "|");

  return static_cast<ReducedPrecisionSupport>(static_cast<std::uint32_t>(a) |
                                              static_cast<std::uint32_t>(b));
}

inline ReducedPrecisionSupport& operator|=(ReducedPrecisionSupport& a,
                                           ReducedPrecisionSupport b) {
  return a = static_cast<ReducedPrecisionSupport>(
             static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

inline ReducedPrecisionSupport operator&(ReducedPrecisionSupport a,
                                         ReducedPrecisionSupport b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_1(mht_1_v, 226, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "&");

  return static_cast<ReducedPrecisionSupport>(static_cast<std::uint32_t>(a) &
                                              static_cast<std::uint32_t>(b));
}

inline ReducedPrecisionSupport& operator&=(ReducedPrecisionSupport& a,
                                           ReducedPrecisionSupport b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_2(mht_2_v, 235, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "=");

  return a = static_cast<ReducedPrecisionSupport>(
             static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}

inline bool SupportsFP16Inference(const ReducedPrecisionSupport& mask) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_3(mht_3_v, 243, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "SupportsFP16Inference");

  return static_cast<bool>(mask & ReducedPrecisionSupport::Float16Inference);
}

inline bool SupportsBfloat16Inference(const ReducedPrecisionSupport& mask) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_4(mht_4_v, 250, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "SupportsBfloat16Inference");

  return static_cast<bool>(mask & ReducedPrecisionSupport::Bfloat16Inference);
}

inline bool SupportsFP16Accumulation(const ReducedPrecisionSupport& mask) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_5(mht_5_v, 257, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "SupportsFP16Accumulation");

  return static_cast<bool>(mask & ReducedPrecisionSupport::Float16Accumulation);
}

inline bool SupportsFP32Accumulation(const ReducedPrecisionSupport& mask) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_6(mht_6_v, 264, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "SupportsFP32Accumulation");

  return static_cast<bool>(mask & ReducedPrecisionSupport::Float32Accumulation);
}

inline bool SupportsReducedPrecisionInference(
    const ReducedPrecisionSupport& mask) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_7(mht_7_v, 272, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "SupportsReducedPrecisionInference");

  return SupportsFP16Inference(mask) || SupportsBfloat16Inference(mask);
}

inline bool SupportsEitherFP16OrFP32Accumulation(
    const ReducedPrecisionSupport& mask) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_8(mht_8_v, 280, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "SupportsEitherFP16OrFP32Accumulation");

  return SupportsFP16Accumulation(mask) != SupportsFP32Accumulation(mask);
}

// Return the key-value pair for reduced precision support metadata.
// Example: mask = Float16Inference | Bfloat16Inference | Float32Accumulation;
// Returned value would be <"reduced_precision_support", "fp16bf16accfp32">.
inline std::pair<std::string, std::string> MetadataForReducedPrecisionSupport(
    const ReducedPrecisionSupport& mask) {
  TFLITE_DCHECK(SupportsReducedPrecisionInference(mask));
  TFLITE_DCHECK(SupportsEitherFP16OrFP32Accumulation(mask));
  std::string value = "";
  if (SupportsFP16Inference(mask)) {
    value += kTfLiteFloat16String;
  }
  if (SupportsBfloat16Inference(mask)) {
    value += kTfLiteBfloat16String;
  }
  value += kTfLiteAccumulationString;
  if (SupportsFP16Accumulation(mask)) {
    value += kTfLiteFloat16String;
  } else if (SupportsFP32Accumulation(mask)) {
    value += kTfLiteFloat32String;
  }
  return std::make_pair(std::string(kTfLiteReducedPrecisionKey), value);
}

inline bool ReadInferenceType(const std::string& metadata, size_t* idx,
                              ReducedPrecisionSupport* mask) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("metadata: \"" + metadata + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_9(mht_9_v, 312, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "ReadInferenceType");

  if (metadata.substr(*idx, 4) == kTfLiteFloat16String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Float16Inference;
    return true;
  } else if (metadata.substr(*idx, 4) == kTfLiteBfloat16String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Bfloat16Inference;
    return true;
  }
  return false;
}

inline bool ReadAccumulationType(const std::string& metadata, size_t* idx,
                                 ReducedPrecisionSupport* mask) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("metadata: \"" + metadata + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_10(mht_10_v, 330, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "ReadAccumulationType");

  if (metadata.substr(*idx, 4) == kTfLiteFloat16String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Float16Accumulation;
    return true;
  } else if (metadata.substr(*idx, 4) == kTfLiteFloat32String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Float32Accumulation;
    return true;
  }
  return false;
}

// If the string is valid, set the given mask to indicate the state in
// string and return true. If the string is invalid, return false.
// A valid string is:
// >= 1 valid inference types + accumulation token + 1 valid accumulation type.
// Valid examples would be: "fp16accfp16", "bf16accfp32"
inline bool SetMaskFromReducedPrecisionMetadata(const std::string& metadata,
                                                ReducedPrecisionSupport* mask) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("metadata: \"" + metadata + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSreduced_precision_supportDTh mht_11(mht_11_v, 353, "", "./tensorflow/lite/tools/optimize/reduced_precision_support.h", "SetMaskFromReducedPrecisionMetadata");

  bool check = true;
  size_t idx = 0;
  ReducedPrecisionSupport rsp = ReducedPrecisionSupport::None;
  do {
    check = ReadInferenceType(metadata, &idx, &rsp);
  } while (check);
  // Ensure we read at least 1 inference type.
  if (idx == 0) {
    return false;
  }
  // Next read the accumulation token.
  if (metadata.substr(idx, 3) != kTfLiteAccumulationString) {
    return false;
  }
  idx += std::string(kTfLiteAccumulationString).size();
  // Next read a valid accumulation type.
  if (!ReadAccumulationType(metadata, &idx, &rsp)) {
    return false;
  }
  // This should be the end of string.
  if (idx != metadata.length()) {
    return false;
  }
  // The string is a valid mask description. Set the value and return.
  *mask = rsp;
  return true;
}

}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_SUPPORT_H
