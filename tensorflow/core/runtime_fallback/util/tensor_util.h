/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_UTIL_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPStensor_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPStensor_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPStensor_utilDTh() {
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


#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

struct TFTensorDeleter {
  void operator()(TF_Tensor* p) const { TF_DeleteTensor(p); }
};
using OwnedTFTensor = std::unique_ptr<TF_Tensor, TFTensorDeleter>;

// Moves one ref on HostBuffer to tensorflow::Tensor.
tensorflow::Tensor MoveHostBufferToTfTensor(
    tfrt::RCReference<tfrt::HostBuffer> host_buffer, tfrt::DType dtype,
    const tfrt::TensorShape& shape);

// Creates a tensorflow::Tensor based on StringHostTensor.
tensorflow::Tensor CopyShtToTfTensor(const tfrt::StringHostTensor& sht);

// Converts tfrt shape to tensorflow shape.
inline tensorflow::TensorShape GetTfShape(const tfrt::TensorShape& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPStensor_utilDTh mht_0(mht_0_v, 218, "", "./tensorflow/core/runtime_fallback/util/tensor_util.h", "GetTfShape");

  llvm::SmallVector<tfrt::Index, 4> dimensions;
  shape.GetDimensions(&dimensions);
  llvm::SmallVector<int64_t, 4> dims(dimensions.begin(), dimensions.end());
  return tensorflow::TensorShape(dims);
}

// Retrives TFRT TensorMetadata from a tensorflow::Tensor.
inline tfrt::TensorMetadata GetTensorMetadata(
    const tensorflow::Tensor& tf_tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPStensor_utilDTh mht_1(mht_1_v, 230, "", "./tensorflow/core/runtime_fallback/util/tensor_util.h", "GetTensorMetadata");

  auto dtype = tfd::GetTfrtDtype(tf_tensor.dtype());
  auto dim_sizes = tf_tensor.shape().dim_sizes();
  static_assert(sizeof(tfrt::Index) == sizeof(dim_sizes.front()),
                "Invalid dimension type size");
  auto shape = llvm::makeArrayRef(
      reinterpret_cast<tfrt::Index*>(dim_sizes.data()), dim_sizes.size());
  return tfrt::TensorMetadata(dtype, shape);
}

inline void CheckBoolCompatibility() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSutilPStensor_utilDTh mht_2(mht_2_v, 243, "", "./tensorflow/core/runtime_fallback/util/tensor_util.h", "CheckBoolCompatibility");

  // sizeof(bool) is implementation defined. The following may only work when
  // sizeof(bool) is 1.
  //
  // TODO(tfrt-devs): It is still undefined behavior to directly cast char*
  // between bool* and access the data. Consider allocating target objects and
  // using memcpy instead.
  static_assert(sizeof(bool) == 1, "Only support when bool is 1 byte.");
}

}  // namespace tfd
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_UTIL_TENSOR_UTIL_H_
