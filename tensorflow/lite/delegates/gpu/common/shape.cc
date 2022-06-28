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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTcc() {
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
#include "tensorflow/lite/delegates/gpu/common/shape.h"

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace tflite {
namespace gpu {
namespace {

struct GetAxisByIndexFunc {
  template <Layout T>
  Axis operator()() const {
    return GetAxis<T>(index);
  }
  int32_t index;
};

struct GetIndexByAxisFunc {
  template <Layout T>
  int operator()() const {
    return GetAxisIndex<T>(axis);
  }
  Axis axis;
};

struct NumAxisFunc {
  template <Layout T>
  int operator()() const {
    return Size<T>();
  }
};

}  // namespace

std::string ToString(Axis axis) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTcc mht_0(mht_0_v, 223, "", "./tensorflow/lite/delegates/gpu/common/shape.cc", "ToString");

  switch (axis) {
    case Axis::BATCH:
      return "batch";
    case Axis::CHANNELS:
      return "channels";
    case Axis::INPUT_CHANNELS:
      return "input_channels";
    case Axis::OUTPUT_CHANNELS:
      return "output_channels";
    case Axis::HEIGHT:
      return "height";
    case Axis::WIDTH:
      return "width";
    case Axis::VALUE:
      return "value";
    case Axis::DEPTH:
      return "depth";
    case Axis::UNKNOWN:
      return "unknown";
  }
  return "undefined";
}

std::string ToString(Layout layout) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTcc mht_1(mht_1_v, 250, "", "./tensorflow/lite/delegates/gpu/common/shape.cc", "ToString");

  switch (layout) {
    case Layout::SCALAR:
      return "scalar";
    case Layout::LINEAR:
      return "linear";
    case Layout::HW:
      return "hw";
    case Layout::HWD:
      return "hwd";
    case Layout::CHW:
      return "chw";
    case Layout::HWC:
      return "hwc";
    case Layout::HWDC:
      return "hwdc";
    case Layout::OHWI:
      return "ohwi";
    case Layout::IHWO:
      return "ihwo";
    case Layout::OIHW:
      return "oihw";
    case Layout::IOHW:
      return "iohw";
    case Layout::BHWC:
      return "bhwc";
    case Layout::BHWDC:
      return "bhwdc";
    case Layout::OHWDI:
      return "ohwi";
    case Layout::UNKNOWN:
      return "unknown";
  }
  return "undefined";
}

Axis GetAxis(Layout layout, int32_t index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTcc mht_2(mht_2_v, 289, "", "./tensorflow/lite/delegates/gpu/common/shape.cc", "GetAxis");

  return DispatchByLayout(layout, GetAxisByIndexFunc{index});
}

int GetAxisIndex(Layout layout, Axis axis) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTcc mht_3(mht_3_v, 296, "", "./tensorflow/lite/delegates/gpu/common/shape.cc", "GetAxisIndex");

  return DispatchByLayout(layout, GetIndexByAxisFunc{axis});
}

bool HasAxis(Layout layout, Axis axis) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTcc mht_4(mht_4_v, 303, "", "./tensorflow/lite/delegates/gpu/common/shape.cc", "HasAxis");

  return GetAxisIndex(layout, axis) >= 0;
}

int Size(Layout layout) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTcc mht_5(mht_5_v, 310, "", "./tensorflow/lite/delegates/gpu/common/shape.cc", "Size");
 return DispatchByLayout(layout, NumAxisFunc()); }

std::string ToString(const Shape& s) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSshapeDTcc mht_6(mht_6_v, 315, "", "./tensorflow/lite/delegates/gpu/common/shape.cc", "ToString");

  return absl::StrCat("{", ToString(s.layout), ", {",
                      absl::StrJoin(s.dimensions, ", "), "}}");
}

}  // namespace gpu
}  // namespace tflite
