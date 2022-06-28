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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PROCESS_BROADCAST_SHAPES_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PROCESS_BROADCAST_SHAPES_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSprocess_broadcast_shapesDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSprocess_broadcast_shapesDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSprocess_broadcast_shapesDTh() {
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


#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

// Consolidates dimensions in broadcast inputs, checks for five-fold pattern.
//
// For example, if sequence of dimensions of one input is
// ..., 1, 3, 1, 7, 9, 5,... and the other is ..., 2, 3, 1, 7, 1, 1, ...
// we can consolidate these as
// ..., 1, 3*7, 9*5, ... and 2, 3*7, 1.
//
// The category is updated in the less-frequent case of shapes that are
// not suited to a fivefold-loop broadcast.
//
// Falls back to generic pattern when it does not know how to process properly.
//
// Returns true iff there is some sort of broadcast, which includes five-fold
// patterns and falling back to generic broadcast.
inline bool ProcessBroadcastShapes(const RuntimeShape& shape0,
                                   const RuntimeShape& shape1,
                                   tflite::ArithmeticParams* params) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSprocess_broadcast_shapesDTh mht_0(mht_0_v, 209, "", "./tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h", "ProcessBroadcastShapes");

  const int dims_count =
      std::max(shape0.DimensionsCount(), shape1.DimensionsCount());

  params->broadcast_category = BroadcastableOpCategory::kGenericBroadcast;
  RuntimeShape scalar_shape(dims_count, 1);

  auto extended_shape0 = RuntimeShape::ExtendedShape(dims_count, shape0);
  auto extended_shape1 = RuntimeShape::ExtendedShape(dims_count, shape1);

  // Check for "exact" match, implicitly accepting any scalar shapes.
  if (extended_shape0 == extended_shape1) {
    params->broadcast_category = BroadcastableOpCategory::kNonBroadcast;
    return false;
  }

  for (int i = dims_count - 1; i >= 0; --i) {
    if (extended_shape0.Dims(i) == extended_shape1.Dims(i)) {
      continue;
    } else if (extended_shape0.Dims(i) == 1) {
      params->broadcast_category =
          BroadcastableOpCategory::kFirstInputBroadcastsFast;
      break;
    } else if (extended_shape1.Dims(i) == 1) {
      params->broadcast_category =
          BroadcastableOpCategory::kSecondInputBroadcastsFast;
      break;
    } else {
      // This case is erroneous: there is a dimension that does not match and
      // is not a broadcast from one shape to the other.
      params->broadcast_category = BroadcastableOpCategory::kGenericBroadcast;
      return true;
    }
  }

  if (params->broadcast_category !=
          BroadcastableOpCategory::kFirstInputBroadcastsFast &&
      params->broadcast_category !=
          BroadcastableOpCategory::kSecondInputBroadcastsFast) {
    // This is unreachable because at least one else clause in the above loop
    // must be reached.
    TFLITE_DCHECK(false);
    params->broadcast_category = BroadcastableOpCategory::kNonBroadcast;
    return false;
  }

  // From this point it is assumed contractually that corresponding dimensions
  // in shape0 and shape1 are either (a) equal or (b) one or other equals 1.
  const bool swap_inputs = params->broadcast_category ==
                           BroadcastableOpCategory::kSecondInputBroadcastsFast;
  const RuntimeShape* shape_a =
      swap_inputs ? &extended_shape1 : &extended_shape0;
  const RuntimeShape* shape_b =
      swap_inputs ? &extended_shape0 : &extended_shape1;

  int i = dims_count - 1;
  params->broadcast_shape[0] = 1;
  params->broadcast_shape[1] = 1;
  params->broadcast_shape[2] = 1;
  params->broadcast_shape[3] = 1;
  params->broadcast_shape[4] = 1;
  // y_0 is greedy: include dims if both or neither equal 1: in other words,
  // test for equality rather than (shape_a->Dims(i) != 1).
  while (i >= 0 && shape_a->Dims(i) == shape_b->Dims(i)) {
    params->broadcast_shape[4] *= shape_b->Dims(i);
    --i;
  }
  // Here either input_a or input_b has dim of 1 (if i >= 0).  If it is input_b
  // that has the unit dimension, the next two loops are not entered.
  while (i >= 0 && shape_a->Dims(i) == 1) {
    params->broadcast_shape[3] *= shape_b->Dims(i);
    --i;
  }
  while (i >= 0 && shape_a->Dims(i) == shape_b->Dims(i)) {
    params->broadcast_shape[2] *= shape_a->Dims(i);
    --i;
  }
  // Here either input_a or input_b has dim of 1 (if i >= 0).
  while (i >= 0 && shape_b->Dims(i) == 1) {
    params->broadcast_shape[1] *= shape_a->Dims(i);
    --i;
  }
  while (i >= 0 && shape_a->Dims(i) == shape_b->Dims(i)) {
    params->broadcast_shape[0] *= shape_b->Dims(i);
    --i;
  }

  // Rarer case is when the broadcast dimensions cannot be handled by a fivefold
  // loop.
  if (i >= 0) {
    params->broadcast_category = BroadcastableOpCategory::kGenericBroadcast;
  }
  return true;
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PROCESS_BROADCAST_SHAPES_H_
