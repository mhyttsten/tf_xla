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

#ifndef TENSORFLOW_COMPILER_XLA_ARRAY3D_H_
#define TENSORFLOW_COMPILER_XLA_ARRAY3D_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSarray3dDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray3dDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSarray3dDTh() {
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


#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>

#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

// Simple 3D array structure.
template <typename T>
class Array3D : public Array<T> {
 public:
  Array3D() : Array<T>(std::vector<int64_t>{0, 0, 0}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray3dDTh mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/array3d.h", "Array3D");
}

  // Creates an array of dimensions n1 x n2 x n3, uninitialized values.
  Array3D(const int64_t n1, const int64_t n2, const int64_t n3)
      : Array<T>(std::vector<int64_t>{n1, n2, n3}) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray3dDTh mht_1(mht_1_v, 213, "", "./tensorflow/compiler/xla/array3d.h", "Array3D");
}

  // Creates an array of dimensions n1 x n2 x n3, initialized to value.
  Array3D(const int64_t n1, const int64_t n2, const int64_t n3, const T value)
      : Array<T>(std::vector<int64_t>{n1, n2, n3}, value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray3dDTh mht_2(mht_2_v, 220, "", "./tensorflow/compiler/xla/array3d.h", "Array3D");
}

  // Creates an array from the given nested initializer list. The outer
  // initializer list is the first dimension, and so on.
  //
  // For example {{{1, 2}, {3, 4}, {5, 6}, {7, 8}},
  //              {{9, 10}, {11, 12}, {13, 14}, {15, 16}},
  //              {{17, 18}, {19, 20}, {21, 22}, {23, 24}}}
  // results in an array with n1=3, n2=4, n3=2.
  Array3D(std::initializer_list<std::initializer_list<std::initializer_list<T>>>
              values)
      : Array<T>(values) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray3dDTh mht_3(mht_3_v, 234, "", "./tensorflow/compiler/xla/array3d.h", "Array3D");
}

  // Creates an array of a floating-point type (half, bfloat16, float,
  // or double) from the given nested initializer list of float values.
  template <typename T2, typename = typename std::enable_if<
                             (std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, bfloat16>::value ||
                              std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) &&
                             std::is_same<T2, float>::value>::type>
  Array3D(
      std::initializer_list<std::initializer_list<std::initializer_list<T2>>>
          values)
      : Array<T>(values) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray3dDTh mht_4(mht_4_v, 250, "", "./tensorflow/compiler/xla/array3d.h", "Array3D");
}

  int64_t n1() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray3dDTh mht_5(mht_5_v, 255, "", "./tensorflow/compiler/xla/array3d.h", "n1");
 return this->dim(0); }
  int64_t n2() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray3dDTh mht_6(mht_6_v, 259, "", "./tensorflow/compiler/xla/array3d.h", "n2");
 return this->dim(1); }
  int64_t n3() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray3dDTh mht_7(mht_7_v, 263, "", "./tensorflow/compiler/xla/array3d.h", "n3");
 return this->dim(2); }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY3D_H_
