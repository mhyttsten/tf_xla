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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_TO_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_TO_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSbroadcast_toDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSbroadcast_toDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSbroadcast_toDTh() {
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


#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace reference_ops {
template <int N>
void BroadcastImpl(const NdArrayDesc<N>& input_desc, const char* input_data,
                   const NdArrayDesc<N>& output_desc, char* output_data,
                   int indexes[N], int dim, const int last_broadcasting_dim,
                   const int type_size) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("input_data: \"" + (input_data == nullptr ? std::string("nullptr") : std::string((char*)input_data)) + "\"");
   mht_0_v.push_back("output_data: \"" + (output_data == nullptr ? std::string("nullptr") : std::string((char*)output_data)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSbroadcast_toDTh mht_0(mht_0_v, 198, "", "./tensorflow/lite/kernels/internal/reference/broadcast_to.h", "BroadcastImpl");

  // Copy data from input to output.
  if (dim == last_broadcasting_dim) {
    int copy_size = output_desc.strides[dim] * type_size;
    const char* data_src =
        input_data + SubscriptToIndex(input_desc, indexes) * type_size;
    char* data_dst =
        output_data + SubscriptToIndex(output_desc, indexes) * type_size;
    for (int i = 0; i < output_desc.extents[dim]; ++i, data_dst += copy_size) {
      memcpy(data_dst, data_src, copy_size);
    }
    return;
  }

  // Recursive call to find the next broadcasting.
  for (indexes[dim] = 0; indexes[dim] < input_desc.extents[dim];
       ++indexes[dim]) {
    BroadcastImpl<N>(input_desc, input_data, output_desc, output_data, indexes,
                     dim + 1, last_broadcasting_dim, type_size);
  }

  // Duplicate data in output tensor.
  indexes[dim] = 0;
  if (input_desc.extents[dim] != output_desc.extents[dim]) {
    int copy_size = output_desc.strides[dim] * type_size;
    char* data_src =
        output_data + SubscriptToIndex(output_desc, indexes) * type_size;
    char* data_dst = data_src + copy_size;
    for (int i = 1; i < output_desc.extents[dim]; ++i, data_dst += copy_size) {
      memcpy(data_dst, data_src, copy_size);
    }
  }
}

template <int N>
inline void BroadcastTo(const RuntimeShape& unextended_input_shape,
                        const char* input_data,
                        const RuntimeShape& unextended_output_shape,
                        char* output_data, TfLiteType data_type) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("input_data: \"" + (input_data == nullptr ? std::string("nullptr") : std::string((char*)input_data)) + "\"");
   mht_1_v.push_back("output_data: \"" + (output_data == nullptr ? std::string("nullptr") : std::string((char*)output_data)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSbroadcast_toDTh mht_1(mht_1_v, 241, "", "./tensorflow/lite/kernels/internal/reference/broadcast_to.h", "BroadcastTo");

  NdArrayDesc<N> input_desc;
  NdArrayDesc<N> output_desc;
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_input_shape),
                 &input_desc);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_output_shape),
                 &output_desc);

  // Get the last dimension has broadcasting. At this dimension, the data is
  // copied from input tensor to output tensor.
  int last_broadcast_dim = -1;
  for (int i = N - 1; i >= 0; --i) {
    if (input_desc.extents[i] != output_desc.extents[i]) {
      last_broadcast_dim = i;
      break;
    }
  }

  // If non-broadcasting, just copy data from input to output tensor.
  if (last_broadcast_dim == -1) {
    memcpy(output_data, input_data,
           unextended_input_shape.FlatSize() * TfLiteTypeGetSize(data_type));
    return;
  }

  // Broadcasting using memcpy.
  int indexes[N] = {0};
  BroadcastImpl<N>(input_desc, input_data, output_desc, output_data, indexes, 0,
                   last_broadcast_dim, TfLiteTypeGetSize(data_type));
}
}  // namespace reference_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_TO_H_
