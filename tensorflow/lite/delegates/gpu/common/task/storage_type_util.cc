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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSstorage_type_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSstorage_type_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSstorage_type_utilDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

absl::Status SelectBestStorageType(const GpuInfo& gpu_info, const BHWC& shape,
                                   TensorStorageType desired,
                                   DataType data_type, Layout layout,
                                   TensorStorageType* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSstorage_type_utilDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/delegates/gpu/common/task/storage_type_util.cc", "SelectBestStorageType");

  if (TensorDescriptor{data_type, desired, layout}
          .CanCreateTensorWithShape(gpu_info, shape)
          .ok()) {
    *result = desired;
    return absl::OkStatus();
  }
  if (gpu_info.IsApiMetal()) {
    *result = TensorStorageType::BUFFER;
    return TensorDescriptor{data_type, TensorStorageType::BUFFER, layout}
        .CanCreateTensorWithShape(gpu_info, shape);
  }
  auto GetBestTypeAfterTexture2D = [&]() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSstorage_type_utilDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/delegates/gpu/common/task/storage_type_util.cc", "lambda");

    if (gpu_info.SupportsImageBuffer() &&
        TensorDescriptor{data_type, TensorStorageType::IMAGE_BUFFER, layout}
            .CanCreateTensorWithShape(gpu_info, shape)
            .ok()) {
      *result = TensorStorageType::IMAGE_BUFFER;
      return absl::OkStatus();
    } else {
      *result = TensorStorageType::BUFFER;
      return TensorDescriptor{data_type, TensorStorageType::BUFFER, layout}
          .CanCreateTensorWithShape(gpu_info, shape);
    }
  };
  auto GetBestTypeAfterTextureArray = [&]() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSstorage_type_utilDTcc mht_2(mht_2_v, 232, "", "./tensorflow/lite/delegates/gpu/common/task/storage_type_util.cc", "lambda");

    if (gpu_info.SupportsImageBuffer() &&
        TensorDescriptor{data_type, TensorStorageType::IMAGE_BUFFER, layout}
            .CanCreateTensorWithShape(gpu_info, shape)
            .ok()) {
      *result = TensorStorageType::IMAGE_BUFFER;
      return absl::OkStatus();
    } else {
      *result = TensorStorageType::BUFFER;
      return TensorDescriptor{data_type, TensorStorageType::BUFFER, layout}
          .CanCreateTensorWithShape(gpu_info, shape);
    }
  };
  auto GetBestTypeAfterTexture3D = [&]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSstorage_type_utilDTcc mht_3(mht_3_v, 248, "", "./tensorflow/lite/delegates/gpu/common/task/storage_type_util.cc", "lambda");

    if (TensorDescriptor{data_type, TensorStorageType::TEXTURE_2D, layout}
            .CanCreateTensorWithShape(gpu_info, shape)
            .ok()) {
      *result = TensorStorageType::TEXTURE_2D;
      return absl::OkStatus();
    } else {
      return GetBestTypeAfterTextureArray();
    }
  };
  switch (desired) {
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return GetBestTypeAfterTexture2D();
    case TensorStorageType::TEXTURE_ARRAY:
      return GetBestTypeAfterTextureArray();
    case TensorStorageType::TEXTURE_3D:
      return GetBestTypeAfterTexture3D();
    case TensorStorageType::IMAGE_BUFFER: {
      if (TensorDescriptor{data_type, TensorStorageType::IMAGE_BUFFER, layout}
              .CanCreateTensorWithShape(gpu_info, shape)
              .ok()) {
        *result = TensorStorageType::IMAGE_BUFFER;
        return absl::OkStatus();
      } else {
        *result = TensorStorageType::BUFFER;
        return TensorDescriptor{data_type, TensorStorageType::BUFFER, layout}
            .CanCreateTensorWithShape(gpu_info, shape);
      }
    }
    case TensorStorageType::BUFFER: {
      *result = TensorStorageType::BUFFER;
      return TensorDescriptor{data_type, TensorStorageType::BUFFER, layout}
          .CanCreateTensorWithShape(gpu_info, shape);
    }
    default:
      return absl::UnimplementedError(absl::StrCat(
          "No support of this storage type - ", ToString(desired)));
  }
}

LinearStorageType DeduceLinearStorageType(
    TensorStorageType tensor_storage_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSstorage_type_utilDTcc mht_4(mht_4_v, 293, "", "./tensorflow/lite/delegates/gpu/common/task/storage_type_util.cc", "DeduceLinearStorageType");

  if (tensor_storage_type == TensorStorageType::BUFFER) {
    return LinearStorageType::BUFFER;
  } else {
    return LinearStorageType::TEXTURE_2D;
  }
}

}  // namespace gpu
}  // namespace tflite
