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

#ifndef TENSORFLOW_COMPILER_XLA_ARRAY2D_H_
#define TENSORFLOW_COMPILER_XLA_ARRAY2D_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh() {
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
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

template <typename T>
class Array2D : public Array<T> {
 public:
  Array2D() : Array<T>(std::vector<int64_t>{0, 0}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/array2d.h", "Array2D");
}

  Array2D(const int64_t n1, const int64_t n2)
      : Array<T>(std::vector<int64_t>{n1, n2}) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_1(mht_1_v, 215, "", "./tensorflow/compiler/xla/array2d.h", "Array2D");
}

  Array2D(const int64_t n1, const int64_t n2, const T value)
      : Array<T>({n1, n2}, value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_2(mht_2_v, 221, "", "./tensorflow/compiler/xla/array2d.h", "Array2D");
}

  // Creates an array from the given nested initializer list. The outer
  // initializer list is the first dimension; the inner is the second dimension.
  // For example, {{1, 2, 3}, {4, 5, 6}} results in an array with n1=2 and n2=3.
  Array2D(std::initializer_list<std::initializer_list<T>> values)
      : Array<T>(values) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_3(mht_3_v, 230, "", "./tensorflow/compiler/xla/array2d.h", "Array2D");
}

  // Creates an array of a floating-point type (half, bfloat16, float,
  // or double) from the given nested initializer list of float values.
  template <typename T2, typename = typename std::enable_if<
                             (std::is_same<T, Eigen::half>::value ||
                              std::is_same<T, bfloat16>::value ||
                              std::is_same<T, float>::value ||
                              std::is_same<T, double>::value) &&
                             std::is_same<T2, float>::value>::type>
  Array2D(std::initializer_list<std::initializer_list<T2>> values)
      : Array<T>(values) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_4(mht_4_v, 244, "", "./tensorflow/compiler/xla/array2d.h", "Array2D");
}

  Array2D(const Array2D<T>& other) : Array<T>(other) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_5(mht_5_v, 249, "", "./tensorflow/compiler/xla/array2d.h", "Array2D");
}

  int64_t n1() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_6(mht_6_v, 254, "", "./tensorflow/compiler/xla/array2d.h", "n1");
 return this->dim(0); }
  int64_t n2() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_7(mht_7_v, 258, "", "./tensorflow/compiler/xla/array2d.h", "n2");
 return this->dim(1); }

  int64_t height() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_8(mht_8_v, 263, "", "./tensorflow/compiler/xla/array2d.h", "height");
 return this->dim(0); }
  int64_t width() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_9(mht_9_v, 267, "", "./tensorflow/compiler/xla/array2d.h", "width");
 return this->dim(1); }

  // Fills the array with a pattern of values of the form:
  //
  //    (rowno << log2ceil(width) | colno) + start_value
  //
  // This makes it easy to see distinct row/column values in the array.
  void FillUnique(T start_value = 0) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_10(mht_10_v, 277, "", "./tensorflow/compiler/xla/array2d.h", "FillUnique");

    int shift = Log2Ceiling<uint64_t>(n2());
    for (int64_t i0 = 0; i0 < n1(); ++i0) {
      for (int64_t i1 = 0; i1 < n2(); ++i1) {
        (*this)(i0, i1) = ((i0 << shift) | i1) + start_value;
      }
    }
  }

  // Applies f to all cells in this array, in row-major order.
  void Each(std::function<void(int64_t, int64_t, T*)> f) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_11(mht_11_v, 290, "", "./tensorflow/compiler/xla/array2d.h", "Each");

    for (int64_t i0 = 0; i0 < n1(); ++i0) {
      for (int64_t i1 = 0; i1 < n2(); ++i1) {
        f(i0, i1, &(*this)(i0, i1));
      }
    }
  }
};

// Returns a linspace-populated Array2D in the range [from, to] (inclusive)
// with dimensions n1 x n2.
template <typename NativeT = float>
std::unique_ptr<Array2D<NativeT>> MakeLinspaceArray2D(double from, double to,
                                                      int64_t n1, int64_t n2) {
  auto array = absl::make_unique<Array2D<NativeT>>(n1, n2);
  int64_t count = n1 * n2;
  NativeT step =
      static_cast<NativeT>((count > 1) ? (to - from) / (count - 1) : 0);
  auto set = [&array, n2](int64_t index, NativeT value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSarray2dDTh mht_12(mht_12_v, 311, "", "./tensorflow/compiler/xla/array2d.h", "lambda");

    (*array)(index / n2, index % n2) = value;
  };
  for (int64_t i = 0; i < count - 1; ++i) {
    set(i, (static_cast<NativeT>(from) +
            static_cast<NativeT>(i) * static_cast<NativeT>(step)));
  }
  set(count - 1, static_cast<NativeT>(to));
  return array;
}
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY2D_H_
