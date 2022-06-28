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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_stateDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_stateDTcc() {
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

#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/graph/tensor_id.h"

namespace tensorflow {

// Adjust value in third_party/tensorflow/python/client/tf_session_wrapper.cc
// in the get_tensor_handle_key function if adjusting the value for
// kTensorHandleResourceTypeName.
const char* SessionState::kTensorHandleResourceTypeName = "TensorHandle";

Status SessionState::GetTensor(const string& handle, Tensor* tensor) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_stateDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/common_runtime/session_state.cc", "SessionState::GetTensor");

  mutex_lock l(state_lock_);
  auto it = tensors_.find(handle);
  if (it == tensors_.end()) {
    return errors::InvalidArgument("The tensor with handle '", handle,
                                   "' is not in the session store.");
  }
  *tensor = it->second;
  return Status::OK();
}

Status SessionState::AddTensor(const string& handle, const Tensor& tensor) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_stateDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/common_runtime/session_state.cc", "SessionState::AddTensor");

  mutex_lock l(state_lock_);
  if (!tensors_.insert({handle, tensor}).second) {
    return errors::InvalidArgument("Failed to add a tensor with handle '",
                                   handle, "' to the session store.");
  }
  return Status::OK();
}

Status SessionState::DeleteTensor(const string& handle) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_stateDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/common_runtime/session_state.cc", "SessionState::DeleteTensor");

  mutex_lock l(state_lock_);
  if (tensors_.erase(handle) == 0) {
    return errors::InvalidArgument("Failed to delete a tensor with handle '",
                                   handle, "' in the session store.");
  }
  return Status::OK();
}

int64_t SessionState::GetNewId() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_stateDTcc mht_3(mht_3_v, 236, "", "./tensorflow/core/common_runtime/session_state.cc", "SessionState::GetNewId");

  mutex_lock l(state_lock_);
  return tensor_id_++;
}

Status TensorStore::AddTensor(const string& name, const TensorAndKey& tk) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_stateDTcc mht_4(mht_4_v, 245, "", "./tensorflow/core/common_runtime/session_state.cc", "TensorStore::AddTensor");

  mutex_lock l(lock_);
  if (!tensors_.insert({name, tk}).second) {
    return errors::InvalidArgument("Failed to add a tensor with name '", name,
                                   "' to the tensor store.");
  }
  dirty_ = true;
  return Status::OK();
}

Status TensorStore::SaveTensors(const std::vector<string>& output_names,
                                SessionState* session_state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_stateDTcc mht_5(mht_5_v, 259, "", "./tensorflow/core/common_runtime/session_state.cc", "TensorStore::SaveTensors");

  mutex_lock l(lock_);
  if (!tensors_.empty()) {
    // Save only the tensors in output_names in the session.
    for (const string& name : output_names) {
      TensorId id(ParseTensorName(name));
      const string op_name(id.first);
      auto it = tensors_.find(op_name);
      if (it != tensors_.end()) {
        // Save the tensor to the session state.
        string key = it->second.GetHandle(op_name);
        TF_RETURN_IF_ERROR(session_state->AddTensor(key, it->second.tensor));
      }
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
