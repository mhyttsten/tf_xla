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
#ifndef TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_TAPE_TAPE_OPERATION_H_
#define TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_TAPE_TAPE_OPERATION_H_
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
class MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTh {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTh() {
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


#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/gradients.h"

namespace tensorflow {
namespace gradients {
class TapeOperation : public AbstractOperation {
 public:
  explicit TapeOperation(AbstractOperation*, Tape*, const GradientRegistry&);
  void Release() override;
  Status Reset(const char* op, const char* raw_device_name) override;
  const string& Name() const override;
  const string& DeviceName() const override;
  Status SetDeviceName(const char* name) override;
  Status AddInput(AbstractTensorHandle* input) override;
  Status AddInputList(absl::Span<AbstractTensorHandle* const> inputs) override;
  Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                 int* num_retvals) override;
  Status SetAttrString(const char* attr_name, const char* data,
                       size_t length) override;
  Status SetAttrInt(const char* attr_name, int64_t value) override;
  Status SetAttrFloat(const char* attr_name, float value) override;
  Status SetAttrBool(const char* attr_name, bool value) override;
  Status SetAttrType(const char* attr_name, DataType value) override;
  Status SetAttrShape(const char* attr_name, const int64_t* dims,
                      const int num_dims) override;
  Status SetAttrFunction(const char* attr_name,
                         const AbstractOperation* value) override;
  Status SetAttrFunctionName(const char* attr_name, const char* value,
                             size_t length) override;
  Status SetAttrTensor(const char* attr_name,
                       AbstractTensorInterface* tensor) override;
  Status SetAttrStringList(const char* attr_name, const void* const* values,
                           const size_t* lengths, int num_values) override;
  Status SetAttrFloatList(const char* attr_name, const float* values,
                          int num_values) override;
  Status SetAttrIntList(const char* attr_name, const int64_t* values,
                        int num_values) override;
  Status SetAttrTypeList(const char* attr_name, const DataType* values,
                         int num_values) override;
  Status SetAttrBoolList(const char* attr_name, const unsigned char* values,
                         int num_values) override;
  Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                          const int* num_dims, int num_values) override;
  Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperation*> values) override;
  AbstractOperation* GetBackingOperation();
  // For LLVM style RTTI.
  static bool classof(const AbstractOperation* ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTh mht_0(mht_0_v, 235, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.h", "classof");

    return ptr->getKind() == kTape;
  }
  ~TapeOperation() override;

 private:
  AbstractOperation* parent_op_;
  ForwardOperation forward_op_;
  Tape* tape_;
  const GradientRegistry& registry_;
};

}  // namespace gradients
}  // namespace tensorflow
#endif  // TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_TAPE_TAPE_OPERATION_H_
