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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_REDUCE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_REDUCE_TESTER_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh() {
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


#include <cstdint>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class ReduceTester {
 public:
  ReduceTester() = default;
  ReduceTester(const ReduceTester&) = delete;
  ReduceTester& operator=(const ReduceTester&) = delete;

  inline ReduceTester& InputShape(std::initializer_list<int32_t> shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh mht_0(mht_0_v, 205, "", "./tensorflow/lite/delegates/xnnpack/reduce_tester.h", "InputShape");

    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    input_shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    input_size_ = ReduceTester::ComputeSize(input_shape_);
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh mht_1(mht_1_v, 217, "", "./tensorflow/lite/delegates/xnnpack/reduce_tester.h", "InputShape");
 return input_shape_; }

  inline int32_t InputSize() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh mht_2(mht_2_v, 222, "", "./tensorflow/lite/delegates/xnnpack/reduce_tester.h", "InputSize");
 return input_size_; }

  inline ReduceTester& Axes(std::initializer_list<int32_t> axes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh mht_3(mht_3_v, 227, "", "./tensorflow/lite/delegates/xnnpack/reduce_tester.h", "Axes");

    for (auto it = axes.begin(); it != axes.end(); ++it) {
      EXPECT_GE(*it, 0);
    }
    axes_ = std::vector<int32_t>(axes.begin(), axes.end());
    return *this;
  }

  inline const std::vector<int32_t>& Axes() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh mht_4(mht_4_v, 238, "", "./tensorflow/lite/delegates/xnnpack/reduce_tester.h", "Axes");
 return axes_; }

  inline ReduceTester& KeepDims(bool keep_dims) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh mht_5(mht_5_v, 243, "", "./tensorflow/lite/delegates/xnnpack/reduce_tester.h", "KeepDims");

    keep_dims_ = keep_dims;
    return *this;
  }

  inline bool KeepDims() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh mht_6(mht_6_v, 251, "", "./tensorflow/lite/delegates/xnnpack/reduce_tester.h", "KeepDims");
 return keep_dims_; }

  inline std::vector<int32_t> OutputShape() const {
    std::vector<int32_t> output_shape;
    output_shape.reserve(InputShape().size());
    std::unordered_set<int32_t> axes_set(Axes().cbegin(), Axes().cend());
    for (int32_t i = 0; i < InputShape().size(); i++) {
      if (axes_set.count(i) != 0) {
        if (KeepDims()) {
          output_shape.push_back(1);
        }
      } else {
        output_shape.push_back(InputShape()[i]);
      }
    }
    return output_shape;
  }

  inline int32_t OutputSize() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh mht_7(mht_7_v, 272, "", "./tensorflow/lite/delegates/xnnpack/reduce_tester.h", "OutputSize");

    int32_t output_size = 1;
    std::unordered_set<int32_t> axes_set(Axes().cbegin(), Axes().cend());
    for (int32_t i = 0; i < InputShape().size(); i++) {
      if (axes_set.count(i) == 0) {
        output_size *= InputShape()[i];
      }
    }
    return output_size;
  }

  inline ReduceTester& RelativeTolerance(float relative_tolerance) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh mht_8(mht_8_v, 286, "", "./tensorflow/lite/delegates/xnnpack/reduce_tester.h", "RelativeTolerance");

    relative_tolerance_ = relative_tolerance;
    return *this;
  }

  inline float RelativeTolerance() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSreduce_testerDTh mht_9(mht_9_v, 294, "", "./tensorflow/lite/delegates/xnnpack/reduce_tester.h", "RelativeTolerance");
 return relative_tolerance_; }

  void Test(tflite::BuiltinOperator reduce_op, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(tflite::BuiltinOperator reduce_op) const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> input_shape_;
  std::vector<int32_t> axes_;
  int32_t input_size_;
  bool keep_dims_ = true;
  float relative_tolerance_ = 10.0f;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_REDUCE_TESTER_H_
