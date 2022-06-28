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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc() {
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
#include "tensorflow/core/common_runtime/eager/tensor_handle_data.h"

#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

class Status;

Status LocalTensorHandleData::Tensor(const tensorflow::Tensor** t) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::Tensor");

  TF_RETURN_IF_ERROR(WaitReady("Tensor"));

  *t = &tensor_;

  return Status::OK();
}

Status LocalTensorHandleData::TensorValue(tensorflow::TensorValue* t) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_1(mht_1_v, 205, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::TensorValue");

  TF_RETURN_IF_ERROR(WaitReady("TensorValue"));

  tensorflow::Tensor& tensor = tensor_;
  *t = tensorflow::TensorValue(&tensor);

  return Status::OK();
}

Status LocalTensorHandleData::Shape(TensorShape* shape) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_2(mht_2_v, 217, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::Shape");

  TF_RETURN_IF_ERROR(WaitReady("Shape"));

  *shape = tensor_.shape();

  return Status::OK();
}

Status LocalTensorHandleData::NumDims(int* num_dims) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_3(mht_3_v, 228, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::NumDims");

  TF_RETURN_IF_ERROR(WaitReady("NumDims"));

  *num_dims = tensor_.dims();

  return Status::OK();
}

Status LocalTensorHandleData::Dim(int dim_index, int64_t* dim) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_4(mht_4_v, 239, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::Dim");

  TF_RETURN_IF_ERROR(WaitReady("Dim"));

  *dim = tensor_.dim_size(dim_index);

  return Status::OK();
}

Status LocalTensorHandleData::NumElements(int64_t* num_elements) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_5(mht_5_v, 250, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::NumElements");

  TF_RETURN_IF_ERROR(WaitReady("NumElements"));

  *num_elements = tensor_.NumElements();

  return Status::OK();
}

Status LocalTensorHandleData::Unprotect() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_6(mht_6_v, 261, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::Unprotect");

  if (!IsReady()) {
    return errors::Internal("Cannot unprotect a non-ready tensor");
  }

  forwarding_protection_tensor_ = tensorflow::Tensor();

  return Status::OK();
}

Status LocalTensorHandleData::SetTensor(tensorflow::Tensor&& t) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_7(mht_7_v, 274, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::SetTensor");

  DCHECK(!IsReady()) << "SetTensor is only called on non-ready handles.";

  tensor_ = std::move(t);
  // Create copy of original tensor to avoid forwarding
  forwarding_protection_tensor_ = tensor_;

  auto& state = absl::get<BlockingControl>(ctrl_);
  state.SetReady();

  return Status::OK();
}

string LocalTensorHandleData::DebugString() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_8(mht_8_v, 290, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::DebugString");

  if (IsReady()) {
    return tensor_.DeviceSafeDebugString();
  } else {
    return "LocalTensorHandleData";
  }
}

void LocalTensorHandleData::BlockingControl::SetReady() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_9(mht_9_v, 301, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::BlockingControl::SetReady");

  mutex_lock l(mu_);
  is_ready_ = true;
}

Status LocalTensorHandleData::BlockingControl::WaitReady(
    const char* caller) const {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("caller: \"" + (caller == nullptr ? std::string("nullptr") : std::string((char*)caller)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_10(mht_10_v, 311, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::BlockingControl::WaitReady");

  tf_shared_lock l(mu_);
  if (!is_ready_) {
    profiler::TraceMe activity(
        [caller] { return absl::StrCat(caller, " WaitReady"); },

        profiler::TraceMeLevel::kInfo);
    DVLOG(3) << "WaitReady: " << caller << " " << this;
    mu_.Await(Condition(&is_ready_));
  }

  return is_poisoned_;
}

void LocalTensorHandleData::BlockingControl::Poison(Status status) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTcc mht_11(mht_11_v, 328, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.cc", "LocalTensorHandleData::BlockingControl::Poison");

  mutex_lock l(mu_);
  if (is_ready_) {
    LOG(ERROR) << "Poison can only be called on non-ready handle: " << this;
    return;
  }
  is_poisoned_ = status;
  is_ready_ = true;
}

}  // namespace tensorflow
