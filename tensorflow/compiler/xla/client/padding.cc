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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSpaddingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSpaddingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSpaddingDTcc() {
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

#include "tensorflow/compiler/xla/client/padding.h"

#include <algorithm>

#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

Status ValidatePaddingValues(absl::Span<const int64_t> input_dimensions,
                             absl::Span<const int64_t> window_dimensions,
                             absl::Span<const int64_t> window_strides) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSpaddingDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/xla/client/padding.cc", "ValidatePaddingValues");

  bool ok = input_dimensions.size() == window_dimensions.size() &&
            input_dimensions.size() == window_strides.size();
  if (!ok) {
    return InvalidArgument(
        "Want input dimensions size %u = window dimensions size %u = window "
        "strides size %u",
        input_dimensions.size(), window_dimensions.size(),
        window_strides.size());
  }
  return Status::OK();
}

std::vector<std::pair<int64_t, int64_t>> MakePadding(
    absl::Span<const int64_t> input_dimensions,
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides, Padding padding) {
  TF_CHECK_OK(ValidatePaddingValues(input_dimensions, window_dimensions,
                                    window_strides));
  std::vector<std::pair<int64_t, int64_t>> low_high_padding;
  switch (padding) {
    case Padding::kValid:
      low_high_padding.resize(window_dimensions.size(), {0, 0});
      return low_high_padding;

    case Padding::kSame:
      for (size_t i = 0; i < input_dimensions.size(); ++i) {
        int64_t input_dimension = input_dimensions[i];
        int64_t window_dimension = window_dimensions[i];
        int64_t window_stride = window_strides[i];
        // We follow the same convention as in Tensorflow, such that
        // output dimension := ceil(input_dimension / window_stride).
        // See tensorflow/tensorflow/python/ops/nn.py
        // for the reference. See also tensorflow/core/kernels/ops_util.cc
        // for the part where we avoid negative padding using max(0, x).
        //
        //
        // For an odd sized window dimension 2N+1 with stride 1, the middle
        // element is always inside the base area, so we can see it as N + 1 +
        // N elements. In the example below, we have a kernel of size
        // 2*3+1=7 so that the center element is 4 with 123 to the
        // left and 567 to the right.
        //
        //  base area:           ------------------------
        //  kernel at left:   1234567
        //  kernel at right:                         1234567
        //
        // We can see visually here that we need to pad the base area
        // by 3 on each side:
        //
        //  padded base area: 000------------------------000
        //
        // For an even number 2N, there are two options:
        //
        // *** Option A
        //
        // We view 2N as (N - 1) + 1 + N, so for N=3 we have 12 to the
        // left, 3 is the center and 456 is to the right, like this:
        //
        //  base area:           ------------------------
        //  kernel at left:    123456
        //  kernel at right:                          123456
        //  padded base area:  00------------------------000
        //
        // Note how we pad by one more to the right than to the left.
        //
        // *** Option B
        //
        // We view 2N as N + 1 + (N - 1), so for N=3 we have 123 to
        // the left, 4 is the center and 56 is to the right, like
        // this:
        //
        //  base area:           ------------------------
        //  kernel at left:   123456
        //  kernel at right:                         123456
        //  padded base area: 000------------------------00
        //
        // The choice here is arbitrary. We choose option A as this is
        // what DistBelief and Tensorflow do.
        //
        // When the stride is greater than 1, the output size is smaller than
        // the input base size. The base area is padded such that the last
        // window fully fits in the padded base area, and the padding amount is
        // evenly divided between the left and the right (or 1 more on the right
        // if odd size padding is required). The example below shows the
        // required padding when the base size is 10, the kernel size is 5, and
        // the stride is 3. In this example, the output size is 4.
        //
        // base area:           ----------
        // 1'st kernel:       12345
        // 2'nd kernel:          12345
        // 3'rd kernel:             12345
        // 4'th kernel:                12345
        // padded base area:  00----------00
        int64_t output_dimension =
            tensorflow::MathUtil::CeilOfRatio(input_dimension, window_stride);
        int64_t padding_size =
            std::max<int64_t>((output_dimension - 1) * window_stride +
                                  window_dimension - input_dimension,
                              0);
        low_high_padding.emplace_back(
            tensorflow::MathUtil::FloorOfRatio(padding_size, int64_t{2}),
            tensorflow::MathUtil::CeilOfRatio(padding_size, int64_t{2}));
      }
      break;
  }

  return low_high_padding;
}

}  // namespace xla
