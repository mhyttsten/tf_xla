/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_STRIDED_SLICE_LOGIC_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_STRIDED_SLICE_LOGIC_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSstrided_slice_logicDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSstrided_slice_logicDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSstrided_slice_logicDTh() {
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


#include <limits>
#include <vector>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace strided_slice {

// Use until std::clamp() is available from C++17.
inline int Clamp(const int v, const int lo, const int hi) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSstrided_slice_logicDTh mht_0(mht_0_v, 198, "", "./tensorflow/lite/kernels/internal/strided_slice_logic.h", "Clamp");

  TFLITE_DCHECK(!(hi < lo));
  if (hi < v) return hi;
  if (v < lo) return lo;
  return v;
}

inline void StridedSlicePadIndices(tflite::StridedSliceParams* p,
                                   int dim_count) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSstrided_slice_logicDTh mht_1(mht_1_v, 209, "", "./tensorflow/lite/kernels/internal/strided_slice_logic.h", "StridedSlicePadIndices");

  // Add indices and mask bits to fully include extra dimensions
  TFLITE_CHECK_LE(dim_count, 5);
  TFLITE_CHECK_GE(dim_count, p->start_indices_count);
  TFLITE_CHECK_EQ(p->start_indices_count, p->stop_indices_count);
  TFLITE_CHECK_EQ(p->stop_indices_count, p->strides_count);

  const int pad_count = dim_count - p->start_indices_count;

  // Pad indices at start, so move arrays by pad_count.
  for (int i = p->start_indices_count - 1; i >= 0; --i) {
    p->strides[i + pad_count] = p->strides[i];
    p->start_indices[i + pad_count] = p->start_indices[i];
    p->stop_indices[i + pad_count] = p->stop_indices[i];
  }
  for (int i = 0; i < pad_count; ++i) {
    p->start_indices[i] = 0;
    p->stop_indices[i] = 1;
    p->strides[i] = 1;
  }

  // Pad masks with 0s or 1s as required.
  p->shrink_axis_mask <<= pad_count;
  p->ellipsis_mask <<= pad_count;
  p->new_axis_mask <<= pad_count;
  p->begin_mask <<= pad_count;
  p->end_mask <<= pad_count;
  p->begin_mask |= (1 << pad_count) - 1;
  p->end_mask |= (1 << pad_count) - 1;

  p->start_indices_count = dim_count;
  p->stop_indices_count = dim_count;
  p->strides_count = dim_count;
}

// Return the index for the first element along that axis. This index will be a
// positive integer between [0, axis_size] (or [-1, axis_size -1] if stride < 0)
// that can be used to index directly into the data.
inline int StartForAxis(const tflite::StridedSliceParams& params,
                        const RuntimeShape& input_shape, int axis) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSstrided_slice_logicDTh mht_2(mht_2_v, 251, "", "./tensorflow/lite/kernels/internal/strided_slice_logic.h", "StartForAxis");

  const auto begin_mask = params.begin_mask;
  const auto* start_indices = params.start_indices;
  const auto* strides = params.strides;
  const int axis_size = input_shape.Dims(axis);
  if (axis_size == 0) {
    return 0;
  }
  // Begin with the specified index.
  int start = start_indices[axis];

  // begin_mask override
  if (begin_mask & 1 << axis) {
    if (strides[axis] > 0) {
      // Forward iteration - use the first element. These values will get
      // clamped below (Note: We could have set them to 0 and axis_size-1, but
      // use lowest() and max() to maintain symmetry with StopForAxis())
      start = std::numeric_limits<int>::lowest();
    } else {
      // Backward iteration - use the last element.
      start = std::numeric_limits<int>::max();
    }
  }

  // Handle negative indices
  if (start < 0) {
    start += axis_size;
  }

  // Clamping
  if (strides[axis] > 0) {
    // Forward iteration
    start = Clamp(start, 0, axis_size);
  } else {
    // Backward iteration
    start = Clamp(start, -1, axis_size - 1);
  }

  return start;
}

// Return the "real" index for the end of iteration along that axis. This is an
// "end" in the traditional C sense, in that it points to one past the last
// element. ie. So if you were iterating through all elements of a 1D array of
// size 4, this function would return 4 as the stop, because it is one past the
// "real" indices of 0, 1, 2 & 3.
inline int StopForAxis(const tflite::StridedSliceParams& params,
                       const RuntimeShape& input_shape, int axis,
                       int start_for_axis) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSstrided_slice_logicDTh mht_3(mht_3_v, 302, "", "./tensorflow/lite/kernels/internal/strided_slice_logic.h", "StopForAxis");

  const auto end_mask = params.end_mask;
  const auto shrink_axis_mask = params.shrink_axis_mask;
  const auto* stop_indices = params.stop_indices;
  const auto* strides = params.strides;
  const int axis_size = input_shape.Dims(axis);
  if (axis_size == 0) {
    return 0;
  }

  // Begin with the specified index
  const bool shrink_axis = shrink_axis_mask & (1 << axis);
  int stop = stop_indices[axis];

  // When shrinking an axis, the end position does not matter (and can be
  // incorrect when negative indexing is used, see Issue #19260). Always use
  // start_for_axis + 1 to generate a length 1 slice, since start_for_axis has
  // already been adjusted for negative indices.
  if (shrink_axis) {
    return start_for_axis + 1;
  }

  // end_mask override
  if (end_mask & (1 << axis)) {
    if (strides[axis] > 0) {
      // Forward iteration - use the last element. These values will get
      // clamped below
      stop = std::numeric_limits<int>::max();
    } else {
      // Backward iteration - use the first element.
      stop = std::numeric_limits<int>::lowest();
    }
  }

  // Handle negative indices
  if (stop < 0) {
    stop += axis_size;
  }

  // Clamping
  // Because the end index points one past the last element, we need slightly
  // different clamping ranges depending on the direction.
  if (strides[axis] > 0) {
    // Forward iteration
    stop = Clamp(stop, 0, axis_size);
  } else {
    // Backward iteration
    stop = Clamp(stop, -1, axis_size - 1);
  }

  return stop;
}

inline bool LoopCondition(int index, int stop, int stride) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSstrided_slice_logicDTh mht_4(mht_4_v, 358, "", "./tensorflow/lite/kernels/internal/strided_slice_logic.h", "LoopCondition");

  // True when we have reached the end of an axis and should loop.
  return stride > 0 ? index >= stop : index <= stop;
}

inline tflite::StridedSliceParams BuildStridedSliceParams(
    int begin_mask, int end_mask, int shrink_axis_mask,
    const std::vector<int>& start_indices, const std::vector<int>& stop_indices,
    const std::vector<int>& strides) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSstrided_slice_logicDTh mht_5(mht_5_v, 369, "", "./tensorflow/lite/kernels/internal/strided_slice_logic.h", "BuildStridedSliceParams");

  tflite::StridedSliceParams op_params;
  const int dims_count = start_indices.size();

  op_params.start_indices_count = dims_count;
  op_params.stop_indices_count = dims_count;
  op_params.strides_count = dims_count;
  for (int i = 0; i < dims_count; ++i) {
    op_params.start_indices[i] = start_indices[i];
    op_params.stop_indices[i] = stop_indices[i];
    op_params.strides[i] = strides[i];
  }

  op_params.begin_mask = begin_mask;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = end_mask;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = shrink_axis_mask;

  return op_params;
}

}  // namespace strided_slice

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_STRIDED_SLICE_LOGIC_H_
