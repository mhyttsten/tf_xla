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
#ifndef TENSORFLOW_C_EAGER_ABSTRACT_OPERATION_H_
#define TENSORFLOW_C_EAGER_ABSTRACT_OPERATION_H_
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
class MHTracer_DTPStensorflowPScPSeagerPSabstract_operationDTh {
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
   MHTracer_DTPStensorflowPScPSeagerPSabstract_operationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSabstract_operationDTh() {
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


#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Abstract interface to an operation.
// This interface allows building and executing an operation in either
// tracing or immediate execution mode.
class AbstractOperation {
 protected:
  enum AbstractOperationKind {
    kGraph,
    kMlir,
    kEager,
    kTfrt,
    kTape,
    kOpHandler
  };
  explicit AbstractOperation(AbstractOperationKind kind) : kind_(kind) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSabstract_operationDTh mht_0(mht_0_v, 211, "", "./tensorflow/c/eager/abstract_operation.h", "AbstractOperation");
}
  virtual ~AbstractOperation() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSabstract_operationDTh mht_1(mht_1_v, 215, "", "./tensorflow/c/eager/abstract_operation.h", "~AbstractOperation");
}

 public:
  AbstractOperationKind getKind() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSabstract_operationDTh mht_2(mht_2_v, 221, "", "./tensorflow/c/eager/abstract_operation.h", "getKind");
 return kind_; }

  // Release any underlying resources, including the interface object.
  //
  // WARNING: The destructor of this class is marked as protected to disallow
  // clients from directly destroying this object since it may manage it's own
  // lifetime through ref counting. Thus this must be allocated on the heap and
  // clients MUST call Release() in order to destroy an instance of this class.
  virtual void Release() = 0;

  virtual Status Reset(const char* op, const char* raw_device_name) = 0;

  virtual const string& Name() const = 0;

  // Returns the operation's device name.
  //
  // The value returned may be different from the one set by SetDeviceName, but
  // it will be compatible with it: the name will be updated by device placement
  // logic to refer to the specific device chosen.
  //
  // Example: If one calls `op->SetDeviceName("/device:GPU")`, the value
  // returned by DeviceName should be "/device:GPU:*" until a particular GPU is
  // chosen for the operation by the device placement logic in the
  // executor. After that, the value returned by DeviceName will be a full
  // device name such as "/job:localhost/replica:0/task:0/device:GPU:1".
  virtual const string& DeviceName() const = 0;

  // Sets the operation device name.
  //
  // The given `name` must be parseable by DeviceNameUtils::ParseFullName, and
  // the result will be used as a constraint for device placement. See the
  // documentation for DeviceName for more details.
  //
  // The value will override the previous value - that is, no "merging" of
  // existing and given constraints will be performed.
  virtual Status SetDeviceName(const char* name) = 0;

  virtual Status AddInput(AbstractTensorHandle* input) = 0;
  virtual Status AddInputList(
      absl::Span<AbstractTensorHandle* const> inputs) = 0;
  virtual Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                         int* num_retvals) = 0;

  virtual Status SetAttrString(const char* attr_name, const char* data,
                               size_t length) = 0;
  virtual Status SetAttrInt(const char* attr_name, int64_t value) = 0;
  virtual Status SetAttrFloat(const char* attr_name, float value) = 0;
  virtual Status SetAttrBool(const char* attr_name, bool value) = 0;
  virtual Status SetAttrType(const char* attr_name, DataType value) = 0;
  virtual Status SetAttrShape(const char* attr_name, const int64_t* dims,
                              const int num_dims) = 0;
  virtual Status SetAttrShape(const char* attr_name,
                              const PartialTensorShape shape);
  virtual Status SetAttrFunction(const char* attr_name,
                                 const AbstractOperation* value) = 0;
  virtual Status SetAttrFunctionName(const char* attr_name, const char* value,
                                     size_t length) = 0;
  virtual Status SetAttrTensor(const char* attr_name,
                               AbstractTensorInterface* tensor) = 0;
  virtual Status SetAttrStringList(const char* attr_name,
                                   const void* const* values,
                                   const size_t* lengths, int num_values) = 0;
  virtual Status SetAttrStringList(const char* attr_name,
                                   absl::Span<string const> values);
  virtual Status SetAttrFloatList(const char* attr_name, const float* values,
                                  int num_values) = 0;
  virtual Status SetAttrIntList(const char* attr_name, const int64_t* values,
                                int num_values) = 0;
  virtual Status SetAttrTypeList(const char* attr_name, const DataType* values,
                                 int num_values) = 0;
  virtual Status SetAttrBoolList(const char* attr_name,
                                 const unsigned char* values,
                                 int num_values) = 0;
  virtual Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                                  const int* num_dims, int num_values) = 0;
  virtual Status SetAttrFunctionList(
      const char* attr_name, absl::Span<const AbstractOperation*> values) = 0;

 private:
  const AbstractOperationKind kind_;
};

// TODO(b/193656009): Defining these in a cc file causes linker errors with
// fastbuild.
inline Status AbstractOperation::SetAttrShape(const char* attr_name,
                                              const PartialTensorShape shape) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSabstract_operationDTh mht_3(mht_3_v, 310, "", "./tensorflow/c/eager/abstract_operation.h", "AbstractOperation::SetAttrShape");

  return SetAttrShape(attr_name, shape.dim_sizes().data(), shape.dims());
}

inline Status AbstractOperation::SetAttrStringList(
    const char* attr_name, absl::Span<string const> values) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSabstract_operationDTh mht_4(mht_4_v, 319, "", "./tensorflow/c/eager/abstract_operation.h", "AbstractOperation::SetAttrStringList");

  std::vector<const char*> raw_strs;
  std::vector<size_t> lengths;
  raw_strs.reserve(values.size());
  lengths.reserve(values.size());
  for (const auto& s : values) {
    raw_strs.emplace_back(s.data());
    lengths.emplace_back(s.size());
  }
  return SetAttrStringList(attr_name,
                           reinterpret_cast<const void**>(raw_strs.data()),
                           lengths.data(), values.size());
}

namespace internal {
struct AbstractOperationDeleter {
  void operator()(AbstractOperation* p) const {
    if (p != nullptr) {
      p->Release();
    }
  }
};
}  // namespace internal

using AbstractOperationPtr =
    std::unique_ptr<AbstractOperation, internal::AbstractOperationDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_ABSTRACT_OPERATION_H_
