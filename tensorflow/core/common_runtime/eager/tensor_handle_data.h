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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_DATA_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_DATA_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh() {
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


#include "absl/types/variant.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Local Tensor Handle: Handle to a Tensor present on the local host.
class LocalTensorHandleData {
 public:
  LocalTensorHandleData() : ctrl_(absl::in_place_type<BlockingControl>) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_0(mht_0_v, 197, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "LocalTensorHandleData");
}
  explicit LocalTensorHandleData(tensorflow::Tensor&& t)
      : tensor_(std::move(t)),
        forwarding_protection_tensor_(tensor_),
        ctrl_(absl::in_place_type<NonBlockingControl>) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_1(mht_1_v, 204, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "LocalTensorHandleData");
}

  // A local tensor handle should be able to satisfy all of these requests.
  Status Tensor(const tensorflow::Tensor** t) const;
  Status TensorValue(tensorflow::TensorValue* t);
  Status Shape(TensorShape* shape) const;
  Status NumDims(int* num_dims) const;
  Status Dim(int dim_index, int64_t* dim) const;
  Status NumElements(int64_t* num_elements) const;
  Status Unprotect();

  bool IsReady() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_2(mht_2_v, 218, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "IsReady");

    return absl::visit([](auto& data) { return data.IsReady(); }, ctrl_);
  }

  Status WaitReady(const char* caller) const {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("caller: \"" + (caller == nullptr ? std::string("nullptr") : std::string((char*)caller)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_3(mht_3_v, 226, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "WaitReady");

    return absl::visit([caller](auto& data) { return data.WaitReady(caller); },
                       ctrl_);
  }
  void Poison(Status status) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_4(mht_4_v, 233, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "Poison");

    return absl::visit([status](auto& data) { data.Poison(status); }, ctrl_);
  }
  Status IsPoisoned() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_5(mht_5_v, 239, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "IsPoisoned");

    return absl::visit([](auto& data) { return data.IsPoisoned(); }, ctrl_);
  }

  Status SetTensor(tensorflow::Tensor&& t);

  string DebugString() const;

 private:
  tensorflow::Tensor tensor_;
  // TensorHandle has its own reference counting which is distinct from the
  // backing Tensor. As a result, if the Tensor reference count is 1 while
  // executing an op, the TensorBuffer could be reused for the output. We avoid
  // this behavior maintaining another reference count with the
  // forwarding_protection_tensor_ Tensor. When Unprotect() is called, we
  // release this Tensor to allow forwarding.
  tensorflow::Tensor forwarding_protection_tensor_;

  // We distinguish between ready and empty tensors with the ctrl_ variant.
  // which contains 2 implementations of the waiting logic. The
  // NonBlockingControl is a simple no-op class whereas the BlockingControl
  // actually uses a mutex. By using a variant we avoid the overhead of
  // constructing and destructing the mutex for ready local tensors.
  class NonBlockingControl {
   public:
    bool IsReady() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_6(mht_6_v, 267, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "IsReady");
 return true; }
    Status WaitReady(const char* caller) const {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("caller: \"" + (caller == nullptr ? std::string("nullptr") : std::string((char*)caller)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_7(mht_7_v, 272, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "WaitReady");
 return Status::OK(); }
    void Poison(Status status) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_8(mht_8_v, 276, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "Poison");
}
    Status IsPoisoned() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_9(mht_9_v, 280, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "IsPoisoned");
 return Status::OK(); }
  };

  class BlockingControl {
   public:
    bool IsReady() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_10(mht_10_v, 288, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "IsReady");

      tf_shared_lock l(mu_);
      return is_ready_;
    }
    void SetReady();
    Status WaitReady(const char* caller) const;
    void Poison(Status status);
    Status IsPoisoned() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPStensor_handle_dataDTh mht_11(mht_11_v, 298, "", "./tensorflow/core/common_runtime/eager/tensor_handle_data.h", "IsPoisoned");

      tf_shared_lock l(mu_);
      return is_poisoned_;
    }

   private:
    mutable mutex mu_;
    bool is_ready_ TF_GUARDED_BY(mu_);
    Status is_poisoned_ TF_GUARDED_BY(mu_);
  };

  absl::variant<NonBlockingControl, BlockingControl> ctrl_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_DATA_H_
