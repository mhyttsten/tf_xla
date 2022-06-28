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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpreluDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpreluDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpreluDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/prelu.h"

#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

GPUOperation CreatePReLU(const GpuInfo& gpu_info,
                         const OperationDef& definition,
                         const PReLUAttributes& attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSpreluDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/gpu/common/tasks/prelu.cc", "CreatePReLU");

  GPUOperation result(definition);
  result.elementwise_ = true;

  std::string alpha_read;
  auto alpha_linear =
      absl::get_if<tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
  if (alpha_linear) {
    TensorLinearDescriptor desc;
    desc.storage_type =
        DeduceLinearStorageType(definition.GetPrimaryStorageType());
    desc.element_type = definition.GetPrimaryDataType();
    desc.UploadLinearData(*alpha_linear);
    result.args_.AddObject(
        "alpha", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
    alpha_read = "FLT4 alpha_val = args.alpha.Read(S_COORD);\n";
  }

  auto alpha_hwc =
      absl::get_if<tflite::gpu::Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
  if (alpha_hwc) {
    const BHWC shape =
        BHWC(1, alpha_hwc->shape.h, alpha_hwc->shape.w, alpha_hwc->shape.c);
    TensorStorageType storage_type;
    auto status = SelectBestStorageType(
        gpu_info, shape, definition.GetPrimaryStorageType(),
        definition.GetDataType(), Layout::HWC, &storage_type);
    if (!status.ok()) {
      storage_type = TensorStorageType::BUFFER;
    }
    TensorDescriptor desc{definition.GetDataType(), storage_type, Layout::HWC};
    desc.UploadData(*alpha_hwc);
    result.args_.AddObject(
        "alpha", absl::make_unique<TensorDescriptor>(std::move(desc)));
    const std::string x_coord = shape.w == 1 ? "0" : "X_COORD";
    const std::string y_coord = shape.h == 1 ? "0" : "Y_COORD";
    const std::string s_coord = shape.c == 1 ? "0" : "S_COORD";
    alpha_read = absl::StrCat("FLT4 alpha_val = args.alpha.Read(", x_coord,
                              ", ", y_coord, ", ", s_coord, ");\n");
    if (shape.c == 1) {
      alpha_read += "  alpha_val.y = alpha_val.x;\n";
      alpha_read += "  alpha_val.z = alpha_val.x;\n";
      alpha_read += "  alpha_val.w = alpha_val.x;\n";
    }
  }

  if (attr.clip != 0) {
    if (definition.precision == CalculationsPrecision::F32) {
      result.args_.AddFloat("clip", attr.clip);
    } else {
      result.args_.AddHalf("clip", half(attr.clip));
    }
    result.code_ = alpha_read +
                   "in_out_value = clamp(in_out_value, INIT_FLT4(0.0f), "
                   "INIT_FLT4(args.clip)) + "
                   "min(INIT_FLT4(0.0f), in_out_value) * alpha_val;";
  } else {
    result.code_ = alpha_read +
                   "in_out_value = max(INIT_FLT4(0.0f), in_out_value) + "
                   "min(INIT_FLT4(0.0f), "
                   "in_out_value) * alpha_val;";
  }

  return result;
}

}  // namespace gpu
}  // namespace tflite
