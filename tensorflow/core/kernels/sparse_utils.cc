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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_utilsDTcc() {
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

#include "tensorflow/core/kernels/sparse_utils.h"

#include <cstddef>

#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace sparse_utils {

template <typename Tindices>
Tindices FindNextDenseRowStartIndex(
    const Tindices sparse_index_begin,
    const typename TTypes<Tindices>::ConstMatrix& indices_mat) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_utilsDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/sparse_utils.cc", "FindNextDenseRowStartIndex");

  // Search in the index range [begin, end) of indices_mat.
  Tindices begin = sparse_index_begin;
  Tindices end = indices_mat.dimension(0);
  const Tindices orig_sparse_index_end = end;

  // The first dense row we search.
  const Tindices orig_dense_index_begin = indices_mat(begin, 0);
  // Early exit if no next dense row index.
  if (orig_dense_index_begin == static_cast<int64_t>(indices_mat(end - 1, 0))) {
    return orig_sparse_index_end;
  }

  Tindices increment = 1;
  while (begin + increment < end &&
         indices_mat(begin + increment, 0) == orig_dense_index_begin) {
    increment *= 2;
  }
  // Narrow the search space as an optimization.
  if (begin + increment < end) {
    end = begin + increment;
  }
  begin += increment / 2;

  // Perform a binary search on the interval [begin, end) for
  // dense_row_index_to_find.
  const Tindices dense_row_index_to_find = orig_dense_index_begin;
  while (begin < end) {
    const Tindices m = begin + (end - begin) / 2;
    const Tindices m_dense_row_index = static_cast<Tindices>(indices_mat(m, 0));
    if (m_dense_row_index == dense_row_index_to_find &&
        (m + 1 == orig_sparse_index_end ||
         static_cast<Tindices>(indices_mat(m + 1, 0)) !=
             dense_row_index_to_find)) {
      return m + 1;
    } else if (m_dense_row_index <= dense_row_index_to_find) {
      begin = m + 1;
    } else {
      end = m;
    }
  }

  // No next dense row index.
  return orig_sparse_index_end;
}

template <typename Tindices>
std::vector<Tindices> GetStartIndicesOfEachDenseRow(
    const typename TTypes<Tindices>::ConstMatrix& indices_mat,
    bool* contains_empty_rows) {
  int64_t start_sparse_index_of_cur_dense_row = 0;
  std::vector<Tindices> segment_indices;
  const Tindices num_entries_in_sparse_tensor = indices_mat.dimension(0);
  const Tindices num_dense_rows_in_sparse_tensor =
      1 + indices_mat(num_entries_in_sparse_tensor - 1, 0);
  // Reserve an extra slot for the 0 we store in the first entry by convention.
  segment_indices.reserve(1 + num_dense_rows_in_sparse_tensor);
  segment_indices.push_back(0);
  for (Tindices i = 0; i < indices_mat(0, 0); ++i) {
    segment_indices.push_back(0);
  }
  *contains_empty_rows = indices_mat(0, 0) > 0;
  while (true) {
    const Tindices start_sparse_index_of_next_dense_row =
        FindNextDenseRowStartIndex<Tindices>(
            start_sparse_index_of_cur_dense_row, indices_mat);
    if (start_sparse_index_of_next_dense_row == num_entries_in_sparse_tensor) {
      segment_indices.push_back(start_sparse_index_of_next_dense_row);
      break;
    }
    // Encode the length of the current dense row as well as the lengths of all
    // the empty rows until the next dense row,
    for (Tindices i = 0;
         i < indices_mat(start_sparse_index_of_next_dense_row, 0) -
                 indices_mat(start_sparse_index_of_cur_dense_row, 0);
         ++i) {
      segment_indices.push_back(start_sparse_index_of_next_dense_row);
    }
    // If there is more than one row between the current and next non-empty
    // rows then those rows are empty.
    *contains_empty_rows |=
        indices_mat(start_sparse_index_of_next_dense_row, 0) -
            indices_mat(start_sparse_index_of_cur_dense_row, 0) >
        1;
    start_sparse_index_of_cur_dense_row = start_sparse_index_of_next_dense_row;
  }
  return segment_indices;
}

template <typename Tindices>
std::vector<Tindices> ParseRowStartIndices(
    const tensorflow::Tensor& tensor,
    const Tindices num_nonzero_entries_in_sparse_mat) {
  std::vector<Tindices> out;
  auto vec = tensor.vec<Tindices>();
  out.reserve(vec.size() + 1);
  for (size_t i = 0; i < vec.dimension(0); ++i) {
    out.push_back(vec(i));
  }
  out.push_back(num_nonzero_entries_in_sparse_mat);
  return out;
}

template <typename Tindices>
bool ContainsEmptyRows(const std::vector<Tindices>& row_start_indices) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_utilsDTcc mht_1(mht_1_v, 304, "", "./tensorflow/core/kernels/sparse_utils.cc", "ContainsEmptyRows");

  // Skip checking the length of the last dense row since it is
  // always non-empty.
  for (size_t i = 1; i < row_start_indices.size() - 1; ++i) {
    if (row_start_indices.at(i) - row_start_indices.at(i - 1) == 0) {
      return true;
    }
  }
  return false;
}

#define REGISTER_SPARSE_UTIL_FUNCTIONS(TypeIndex)                           \
  template TypeIndex FindNextDenseRowStartIndex<TypeIndex>(                 \
      const TypeIndex sparse_index_begin,                                   \
      const TTypes<TypeIndex>::ConstMatrix& indices_mat);                   \
  template std::vector<TypeIndex> GetStartIndicesOfEachDenseRow<TypeIndex>( \
      const TTypes<TypeIndex>::ConstMatrix& indices_mat,                    \
      bool* contains_empty_rows);                                           \
  template bool ContainsEmptyRows<TypeIndex>(                               \
      const std::vector<TypeIndex>& row_start_indices);                     \
  template std::vector<TypeIndex> ParseRowStartIndices<TypeIndex>(          \
      const tensorflow::Tensor& tensor,                                     \
      const TypeIndex num_nonzero_entries_in_sparse_mat);

REGISTER_SPARSE_UTIL_FUNCTIONS(int32);
REGISTER_SPARSE_UTIL_FUNCTIONS(int64);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint8);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint16);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint32);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint64);

}  // namespace sparse_utils
}  // namespace tensorflow
