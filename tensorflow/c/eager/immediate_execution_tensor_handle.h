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
#ifndef TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_TENSOR_HANDLE_H_
#define TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_TENSOR_HANDLE_H_
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
class MHTracer_DTPStensorflowPScPSeagerPSimmediate_execution_tensor_handleDTh {
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
   MHTracer_DTPStensorflowPScPSeagerPSimmediate_execution_tensor_handleDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSimmediate_execution_tensor_handleDTh() {
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


#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Abstract interface to a TensorHandle.
//
// A TensorHandle is management class around a Tensor which may track additional
// metadata and synchronization.
//
// This allows us to hide concrete implementations of TensorHandle from header
// files. The interface lists the common functionality that must be provided by
// any concrete implementation. However, in cases where the true concrete class
// is needed a static_cast can be applied.
class ImmediateExecutionTensorHandle : public AbstractTensorHandle {
 public:
  // Returns number of dimensions.
  virtual Status NumDims(int* num_dims) const = 0;
  // Returns number of elements across all dimensions.
  virtual Status NumElements(int64_t* num_elements) const = 0;
  // Returns size of specified dimension
  //
  // -1 indicates an unknown axis length; this is unreachable for most standard
  // ImmediateExecutionTensorHandles, but comes up for example when computing
  // the shape of a parallel tensor with component shapes differing across
  // devices.
  virtual Status Dim(int dim_index, int64_t* dim) const = 0;

  // Returns the device which created the handle.
  virtual const char* DeviceName(Status* status) const = 0;
  // Returns the device where the tensor was placed.
  virtual const char* BackingDeviceName(Status* status) const = 0;
  // Returns the device type which created the handle.
  virtual const char* DeviceType(Status* status) const = 0;
  // Returns the device ID which created the handle.
  virtual int DeviceId(Status* status) const = 0;
  // Returns a tensor for the handle. If tensor is remote, it will be copied.
  virtual AbstractTensorInterface* Resolve(Status* status) = 0;

  // Return a copy of the handle.
  virtual ImmediateExecutionTensorHandle* Copy() = 0;

  std::string DebugString() const override;

  // Returns a Boolean hint indicating whether callers should prefer
  // `SummarizeValue` to resolving this handle and formatting the tensor.
  //
  // For example some tensor handles may represent distributed values, in which
  // case placement information is lost when resolving the handle.
  //
  // If false, a caller might implement pretty-printing by resolving and
  // iterating over the resulting tensor. This may still be viable if resolving
  // the handle loses information, but `SummarizeValue` would be more precise.
  virtual bool PreferCustomSummarizer() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSimmediate_execution_tensor_handleDTh mht_0(mht_0_v, 242, "", "./tensorflow/c/eager/immediate_execution_tensor_handle.h", "PreferCustomSummarizer");
 return false; }

  // Returns a string which summarizes the value of this TensorHandle, for
  // debugging. Does not include a shape or dtype.
  //
  // Included in the default implementation of DebugString.
  virtual Status SummarizeValue(std::string& summary) const;

  // Release any underlying resources, including the interface object.
  //
  // WARNING: The destructor of this class is marked as protected to disallow
  // clients from directly destroying this object since it may manage its own
  // lifetime through ref counting. Thus this must be allocated on the heap and
  // clients MUST call Release() in order to destroy an instance of this class.
  virtual void Release() = 0;

  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSimmediate_execution_tensor_handleDTh mht_1(mht_1_v, 262, "", "./tensorflow/c/eager/immediate_execution_tensor_handle.h", "classof");

    return ptr->getKind() == kEager || ptr->getKind() == kTfrt;
  }

 protected:
  explicit ImmediateExecutionTensorHandle(AbstractTensorHandleKind kind)
      : AbstractTensorHandle(kind) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSimmediate_execution_tensor_handleDTh mht_2(mht_2_v, 271, "", "./tensorflow/c/eager/immediate_execution_tensor_handle.h", "ImmediateExecutionTensorHandle");
}
  ~ImmediateExecutionTensorHandle() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSimmediate_execution_tensor_handleDTh mht_3(mht_3_v, 275, "", "./tensorflow/c/eager/immediate_execution_tensor_handle.h", "~ImmediateExecutionTensorHandle");
}
};

namespace internal {
struct ImmediateExecutionTensorHandleDeleter {
  void operator()(ImmediateExecutionTensorHandle* p) const {
    if (p != nullptr) {
      p->Release();
    }
  }
};
}  // namespace internal

using ImmediateTensorHandlePtr =
    std::unique_ptr<ImmediateExecutionTensorHandle,
                    internal::ImmediateExecutionTensorHandleDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_IMMEDIATE_EXECUTION_TENSOR_HANDLE_H_
