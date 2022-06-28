/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_SPARSE_DIM_COMPARATOR_H_
#define TENSORFLOW_CORE_UTIL_SPARSE_DIM_COMPARATOR_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSsparsePSdim_comparatorDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSdim_comparatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSsparsePSdim_comparatorDTh() {
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


#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace sparse {

/////////////////
// DimComparator
/////////////////
//
// Helper class, mainly used by the IndexSortOrder. This comparator
// can be passed to e.g. std::sort, or any other sorter, to sort two
// rows of an index matrix according to the dimension(s) of interest.
// The dimensions to sort by are passed to the constructor as "order".
//
// Example: if given index matrix IX, two rows ai and bi, and order = {2,1}.
// operator() compares
//    IX(ai,2) < IX(bi,2).
// If IX(ai,2) == IX(bi,2), it compares
//    IX(ai,1) < IX(bi,1).
//
// This can be used to sort a vector of row indices into IX according to
// the values in IX in particular columns (dimensions) of interest.
class DimComparator {
 public:
  typedef typename gtl::ArraySlice<int64_t> VarDimArray;

  DimComparator(const TTypes<int64_t>::Matrix& ix, const VarDimArray& order,
                const VarDimArray& shape)
      : ix_(ix), order_(order), dims_(shape.size()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSdim_comparatorDTh mht_0(mht_0_v, 220, "", "./tensorflow/core/util/sparse/dim_comparator.h", "DimComparator");

    DCHECK_GT(order.size(), size_t{0}) << "Must order using at least one index";
    DCHECK_LE(order.size(), shape.size()) << "Can only sort up to dims";
    for (size_t d = 0; d < order.size(); ++d) {
      DCHECK_GE(order[d], 0);
      DCHECK_LT(order[d], shape.size());
    }
  }

  inline bool operator()(const int64_t i, const int64_t j) const {
    for (int di = 0; di < dims_; ++di) {
      const int64_t d = order_[di];
      if (ix_(i, d) < ix_(j, d)) return true;
      if (ix_(i, d) > ix_(j, d)) return false;
    }
    return false;
  }

  // Compares two indices taken from corresponding index matrices, using the
  // standard, row-major (or lexicographic) order.  Useful for cases that need
  // to distinguish between all three orderings (<, ==, >).
  inline static int cmp(const TTypes<int64_t>::ConstMatrix& a_idx,
                        const TTypes<int64_t>::ConstMatrix& b_idx,
                        const int64_t a_row, const int64_t b_row,
                        const int dims) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSdim_comparatorDTh mht_1(mht_1_v, 247, "", "./tensorflow/core/util/sparse/dim_comparator.h", "cmp");

    for (int d = 0; d < dims; ++d) {
      const int64_t a = a_idx(a_row, d);
      const int64_t b = b_idx(b_row, d);
      if (a < b) {
        return -1;
      } else if (a > b) {
        return 1;
      }
    }
    return 0;
  }

 protected:
  const TTypes<int64_t>::Matrix ix_;
  const VarDimArray order_;
  const int dims_;
  const std::vector<int64_t>* ix_order_;
};

template <int ORDER_DIM>
class FixedDimComparator : DimComparator {
 public:
  FixedDimComparator(const TTypes<int64_t>::Matrix& ix,
                     const VarDimArray& order, const VarDimArray& shape)
      : DimComparator(ix, order, shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSdim_comparatorDTh mht_2(mht_2_v, 275, "", "./tensorflow/core/util/sparse/dim_comparator.h", "FixedDimComparator");

    DCHECK_EQ(order.size(), ORDER_DIM);
  }
  inline bool operator()(const int64_t i, const int64_t j) const {
    bool value = false;
    for (int di = 0; di < ORDER_DIM; ++di) {
      const int64_t d = order_[di];
      if (ix_(i, d) < ix_(j, d)) {
        value = true;
        break;
      }
      if (ix_(i, d) > ix_(j, d)) break;
    }
    return value;
  }
};

}  // namespace sparse
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_SPARSE_DIM_COMPARATOR_H_
