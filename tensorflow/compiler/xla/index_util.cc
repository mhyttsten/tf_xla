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
class MHTracer_DTPStensorflowPScompilerPSxlaPSindex_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSindex_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSindex_utilDTcc() {
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

#include "tensorflow/compiler/xla/index_util.h"

#include <algorithm>
#include <string>

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

/* static */ int64_t IndexUtil::MultidimensionalIndexToLinearIndex(
    const Shape& shape, absl::Span<const int64_t> multi_index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSindex_utilDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/xla/index_util.cc", "IndexUtil::MultidimensionalIndexToLinearIndex");

  DCHECK_EQ(shape.dimensions_size(), multi_index.size());

  for (size_t i = 0; i < multi_index.size(); ++i) {
    DCHECK_GE(multi_index[i], 0);
    DCHECK_LT(multi_index[i], shape.dimensions(i))
        << "indexing beyond extent in dimension " << i << ":"
        << "\n\tindex: " << absl::StrJoin(multi_index, ",")
        << "\n\tshape: " << ShapeUtil::HumanString(shape);
  }

  // Let the array be sized like so for dimensions i from 0 to n-1:
  //
  //   [D{n-1} x D{n-2} x .. x D{0}]
  //
  // Let the order of the dimensions in the minor_to_major field in
  // Layout be:
  //
  //   L(0), L(1), ... , L(n-1)
  //
  // where L(0) is the most-minor dimension and L(n-1) the most-major. The
  // multidimensional index:
  //
  //   [I{0}, I{1}, ... , I{n-1}]
  //
  // then corresponds to the following linear index:
  //
  // linear_index =
  //   (((  ... + I{L(2)}) * D{L(1)} + I{L(1)}) * D{L(0)} + I{L(0)}
  //
  // or equivalently:
  //
  // linear_index =
  //   I{L(n-1)} * (D{L(n-2)} * D{L(n-3)} * D{L(n-4)} *     ....    D{L(0)}) +
  //   I{L(n-2)} *             (D{L(n-3)} * D{L(n-4)} *     ....    D{L(0)}) +
  //   I{L(n-3)} *                         (D{L(n-4)} *     ....    D{L(0)}) +
  //                                   ...                                   +
  //   I{L(2)} *                                         (D{L(1)} * D{L(0)}) +
  //   I{L(1)} *                                                    D{L(0)}  +
  //   I{L(0)}
  //
  // We compute the linear index value by accumulating the terms above from
  // I{L(0)} up to I{L(n-1)}. Scale accumulates the product term D{L(0}} *
  // D{L(1)} * ...

  // Scale factor holding the growing product of D{L(i)} terms.
  int64_t scale = 1;
  int64_t linear_index = 0;
  bool first = true;
  for (auto dimension : LayoutUtil::MinorToMajor(shape)) {
    if (first) {
      // Avoid two multiplies on the first loop iteration
      linear_index = multi_index[dimension];
      scale = shape.dimensions(dimension);
      first = false;
    } else {
      linear_index += scale * multi_index[dimension];
      scale *= shape.dimensions(dimension);
    }
  }
  return linear_index;
}

/* static */ std::vector<int64_t> IndexUtil::LinearIndexToMultidimensionalIndex(
    const Shape& shape, int64_t linear_index) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSindex_utilDTcc mht_1(mht_1_v, 265, "", "./tensorflow/compiler/xla/index_util.cc", "IndexUtil::LinearIndexToMultidimensionalIndex");

  DCHECK_GE(linear_index, 0);
  DCHECK_LT(linear_index, ShapeUtil::ElementsIn(shape));

  // The following formula computes each element of the multidimensional index
  // (See comments in MultidimensionalIndexToLinearIndex for notation):
  //
  // I{L(0)} = linear_index % D{L(0)}
  // I{L(1)} = (linear_index / D{L(0)}) % D{L(1)}
  // I{L(2)} = (linear_index / (D{L(0)} * D{L(1)})) % D{L(2)}
  // ...
  std::vector<int64_t> multi_index(shape.dimensions_size());

  // Accumulated product D{L(0)} * D{L(1)} * ...
  int64_t divisor = 1;
  for (auto dimension : LayoutUtil::MinorToMajor(shape)) {
    multi_index[dimension] =
        (linear_index / divisor) % shape.dimensions(dimension);
    divisor *= shape.dimensions(dimension);
  }
  return multi_index;
}

/* static */ bool IndexUtil::BumpIndices(const Shape& shape,
                                         absl::Span<int64_t> indices) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSindex_utilDTcc mht_2(mht_2_v, 292, "", "./tensorflow/compiler/xla/index_util.cc", "IndexUtil::BumpIndices");

  for (int64_t dimno = indices.size() - 1; dimno >= 0; --dimno) {
    int64_t limit = shape.dimensions(dimno);
    if (indices[dimno] + 1 < limit) {
      indices[dimno]++;
      // Whenever an index of a dimension is increased, it means that all
      // following dimensions have maxed out, so they must go to 0.
      std::fill(indices.begin() + dimno + 1, indices.end(), 0);
      return true;
    }
  }
  return false;
}

/* static */ int64_t IndexUtil::GetDimensionStride(const Shape& shape,
                                                   int64_t dimension) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSindex_utilDTcc mht_3(mht_3_v, 310, "", "./tensorflow/compiler/xla/index_util.cc", "IndexUtil::GetDimensionStride");

  int64_t stride = 1;
  for (auto dim : LayoutUtil::MinorToMajor(shape)) {
    if (dim == dimension) {
      break;
    }
    stride *= shape.dimensions()[dim];
  }
  return stride;
}

/* static */ bool IndexUtil::IndexInBounds(const Shape& shape,
                                           absl::Span<const int64_t> index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSindex_utilDTcc mht_4(mht_4_v, 325, "", "./tensorflow/compiler/xla/index_util.cc", "IndexUtil::IndexInBounds");

  int64_t rank = shape.rank();
  const int64_t index_size = index.size();
  if (rank != index_size) {
    return false;
  }
  for (int64_t d = 0; d < rank; ++d) {
    if (index[d] >= shape.dimensions(d)) {
      return false;
    }
  }
  return true;
}

/* static */ int IndexUtil::CompareIndices(absl::Span<const int64_t> lhs,
                                           absl::Span<const int64_t> rhs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSindex_utilDTcc mht_5(mht_5_v, 343, "", "./tensorflow/compiler/xla/index_util.cc", "IndexUtil::CompareIndices");

  int64_t rank = lhs.size();
  const int64_t rhs_rank = rhs.size();
  CHECK_EQ(rhs_rank, rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (lhs[dim] < rhs[dim]) {
      return -1;
    } else if (lhs[dim] > rhs[dim]) {
      return 1;
    }
  }
  return 0;
}

}  // namespace xla
