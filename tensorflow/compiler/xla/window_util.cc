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
class MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc() {
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

#include "tensorflow/compiler/xla/window_util.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace window_util {

Window MakeWindow(absl::Span<const int64_t> sizes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/xla/window_util.cc", "MakeWindow");

  Window window;
  for (int64_t size : sizes) {
    auto* dimension = window.add_dimensions();
    dimension->set_size(size);
    dimension->set_stride(1);
    dimension->set_base_dilation(1);
    dimension->set_window_dilation(1);
  }
  return window;
}

Window MakeWindow(absl::Span<const int64_t> sizes,
                  absl::Span<const int64_t> strides) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/xla/window_util.cc", "MakeWindow");

  Window window;
  CHECK_EQ(sizes.size(), strides.size());
  for (auto nb = 0; nb < sizes.size(); ++nb) {
    auto* dimension = window.add_dimensions();
    dimension->set_size(sizes[nb]);
    dimension->set_stride(strides[nb]);
    dimension->set_base_dilation(1);
    dimension->set_window_dilation(1);
  }
  return window;
}

PaddingConfig MakeSymmetricPadding(absl::Span<const int64_t> sizes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_2(mht_2_v, 230, "", "./tensorflow/compiler/xla/window_util.cc", "MakeSymmetricPadding");

  PaddingConfig config;
  for (int64_t size : sizes) {
    auto* dimension = config.add_dimensions();
    dimension->set_edge_padding_low(size);
    dimension->set_edge_padding_high(size);
  }
  return config;
}

/* static */ std::string ToString(const WindowDimension& dim) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_3(mht_3_v, 243, "", "./tensorflow/compiler/xla/window_util.cc", "ToString");

  using absl::StrAppend;
  using absl::StrCat;
  std::string str = StrCat("(size=", dim.size());
  if (dim.stride() != 1) {
    StrAppend(&str, ",stride=", dim.stride());
  }
  if (dim.padding_low() != 0) {
    StrAppend(&str, ",padding_low=", dim.padding_low());
  }
  if (dim.padding_high() != 0) {
    StrAppend(&str, ",padding_high=", dim.padding_high());
  }
  if (dim.base_dilation() != 1) {
    StrAppend(&str, ",base_dilation=", dim.base_dilation());
  }
  if (dim.window_dilation() != 1) {
    StrAppend(&str, ",window_dilation=", dim.window_dilation());
  }
  if (dim.window_reversal()) {
    StrAppend(&str, ",window_reversal");
  }
  StrAppend(&str, ")");
  return str;
}

std::string ToString(const Window& window) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_4(mht_4_v, 272, "", "./tensorflow/compiler/xla/window_util.cc", "ToString");

  using absl::StrAppend;
  using absl::StrCat;

  std::string str;
  const auto add_field =
      [&](const char* heading,
          std::function<std::string(const WindowDimension&)> format) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("heading: \"" + (heading == nullptr ? std::string("nullptr") : std::string((char*)heading)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_5(mht_5_v, 283, "", "./tensorflow/compiler/xla/window_util.cc", "lambda");

        StrAppend(&str, heading, "=");
        const char* prefix = "";
        for (const auto& window_dimension : window.dimensions()) {
          StrAppend(&str, prefix, format(window_dimension));
          prefix = "x";
        }
      };

  if (window.dimensions_size() > 0) {
    add_field("size",
              [](const WindowDimension& dim) { return StrCat(dim.size()); });
  }
  if (HasStride(window)) {
    add_field(" stride",
              [](const WindowDimension& dim) { return StrCat(dim.stride()); });
  }
  if (HasPadding(window)) {
    add_field(" pad", [](const WindowDimension& dim) {
      return StrCat(dim.padding_low(), "_", dim.padding_high());
    });
  }
  if (HasBaseDilation(window)) {
    add_field(" lhs_dilate", [](const WindowDimension& dim) {
      return StrCat(dim.base_dilation());
    });
  }
  if (HasWindowDilation(window)) {
    add_field(" rhs_dilate", [](const WindowDimension& dim) {
      return StrCat(dim.window_dilation());
    });
  }
  if (HasWindowReversal(window)) {
    add_field(" rhs_reversal", [](const WindowDimension& dim) {
      return StrCat(dim.window_reversal() ? 1 : 0);
    });
  }
  return str;
}

bool HasStride(const Window& window) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_6(mht_6_v, 326, "", "./tensorflow/compiler/xla/window_util.cc", "HasStride");

  for (const auto& dim : window.dimensions()) {
    if (dim.stride() != 1) {
      return true;
    }
  }
  return false;
}

bool HasPadding(const Window& window) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_7(mht_7_v, 338, "", "./tensorflow/compiler/xla/window_util.cc", "HasPadding");

  for (const auto& dim : window.dimensions()) {
    if (dim.padding_low() != 0 || dim.padding_high() != 0) {
      return true;
    }
  }
  return false;
}

bool HasSymmetricPadding(const Window& window) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_8(mht_8_v, 350, "", "./tensorflow/compiler/xla/window_util.cc", "HasSymmetricPadding");

  return absl::c_all_of(window.dimensions(), [](const WindowDimension& dim) {
    return dim.padding_low() == dim.padding_high();
  });
}

bool HasSymmetricPadding(const PaddingConfig& padding_config) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_9(mht_9_v, 359, "", "./tensorflow/compiler/xla/window_util.cc", "HasSymmetricPadding");

  return absl::c_all_of(padding_config.dimensions(),
                        [](const PaddingConfig::PaddingConfigDimension& dim) {
                          return dim.edge_padding_low() ==
                                 dim.edge_padding_high();
                        });
}

bool HasNegativePadding(const Window& window) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_10(mht_10_v, 370, "", "./tensorflow/compiler/xla/window_util.cc", "HasNegativePadding");

  return absl::c_any_of(window.dimensions(), [](const WindowDimension& dim) {
    return dim.padding_low() < 0 || dim.padding_high() < 0;
  });
}

bool HasBaseDilation(const Window& window) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_11(mht_11_v, 379, "", "./tensorflow/compiler/xla/window_util.cc", "HasBaseDilation");

  for (const auto& dim : window.dimensions()) {
    if (dim.base_dilation() != 1) {
      return true;
    }
  }
  return false;
}

bool HasWindowDilation(const Window& window) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_12(mht_12_v, 391, "", "./tensorflow/compiler/xla/window_util.cc", "HasWindowDilation");

  for (const auto& dim : window.dimensions()) {
    if (dim.window_dilation() != 1) {
      return true;
    }
  }
  return false;
}

bool HasWindowReversal(const Window& window) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_13(mht_13_v, 403, "", "./tensorflow/compiler/xla/window_util.cc", "HasWindowReversal");

  for (const auto& dim : window.dimensions()) {
    if (dim.window_reversal()) {
      return true;
    }
  }
  return false;
}

bool AllOrNoneReversed(const Window& window) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_14(mht_14_v, 415, "", "./tensorflow/compiler/xla/window_util.cc", "AllOrNoneReversed");

  if (window.dimensions().empty()) {
    return true;
  }
  bool reversed = window.dimensions()[0].window_reversal();
  return absl::c_all_of(window.dimensions(), [&](const WindowDimension& dim) {
    return dim.window_reversal() == reversed;
  });
}

bool HasDilation(const Window& window) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_15(mht_15_v, 428, "", "./tensorflow/compiler/xla/window_util.cc", "HasDilation");

  return HasBaseDilation(window) || HasWindowDilation(window);
}

bool IsTrivialWindowDimension(const WindowDimension& window_dimension) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_16(mht_16_v, 435, "", "./tensorflow/compiler/xla/window_util.cc", "IsTrivialWindowDimension");

  return window_dimension.size() == 1 && window_dimension.stride() == 1 &&
         window_dimension.padding_low() == 0 &&
         window_dimension.padding_high() == 0 &&
         window_dimension.window_dilation() == 1 &&
         window_dimension.base_dilation() == 1;
}

bool HasOverlappingWindow(const Window& window) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_17(mht_17_v, 446, "", "./tensorflow/compiler/xla/window_util.cc", "HasOverlappingWindow");

  for (const auto& dim : window.dimensions()) {
    if (dim.size() > dim.stride()) {
      return true;
    }
  }
  return false;
}

int64_t DilatedBound(int64_t bound, int64_t dilation) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_18(mht_18_v, 458, "", "./tensorflow/compiler/xla/window_util.cc", "DilatedBound");

  CHECK_GE(bound, 0);
  CHECK_GE(dilation, 1);
  if (bound == 0) {
    return 0;
  }

  // Suppose the array has three entries 123 and the dilation factor is 4. Then
  // the dilated array has 9 entries 1xxx2xxx3. Here, each original entry except
  // the last expands into 4 entries, so that is (bound - 1) * dilation. Then we
  // add 1 to account for the final input element.
  return (bound - 1) * dilation + 1;
}

int64_t StridedBound(int64_t bound, int64_t window_size, int64_t stride) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSwindow_utilDTcc mht_19(mht_19_v, 475, "", "./tensorflow/compiler/xla/window_util.cc", "StridedBound");

  CHECK_GE(window_size, 0);
  CHECK_GE(bound, 0);
  CHECK_GE(stride, 1);

  if (bound == 0 || window_size > bound) {
    return 0;
  }

  // Without considering stride, the maximum valid offset is bound -
  // window_size. Taking stride into account, the valid offsets then have the
  // form q * stride for q = 0, ..., Q such that q * stride <= bound -
  // window_size. This implies that Q equals floor(bound - window_size /
  // stride). There are Q + 1 valid values of q, yielding the formula below.
  return (bound - window_size) / stride + 1;
}

}  // namespace window_util
}  // namespace xla
