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

#ifndef TENSORFLOW_CORE_FRAMEWORK_LOOKUP_INTERFACE_H_
#define TENSORFLOW_CORE_FRAMEWORK_LOOKUP_INTERFACE_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSlookup_interfaceDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSlookup_interfaceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSlookup_interfaceDTh() {
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


#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class OpKernelContext;

namespace lookup {

// Forward declaration so we can define GetInitializableLookupTable() in
// LookupInterface.
class InitializableLookupTable;

// Lookup interface for batch lookups used by table lookup ops.
class LookupInterface : public ResourceBase {
 public:
  // Performs batch lookups, for every element in the key tensor, Find returns
  // the corresponding value into the values tensor.
  // If an element is not present in the table, the given default value is used.

  // For tables that require initialization, Find is available once the table
  // is marked as initialized.

  // Returns the following statuses:
  // - OK: when the find finishes successfully.
  // - FailedPrecondition: if the table is not initialized.
  // - InvalidArgument: if any of the preconditions on the lookup key or value
  //   fails.
  // - In addition, other implementations may provide another non-OK status
  //   specific to their failure modes.
  virtual Status Find(OpKernelContext* ctx, const Tensor& keys, Tensor* values,
                      const Tensor& default_value) = 0;

  // Inserts elements into the table. Each element of the key tensor is
  // associated with the corresponding element in the value tensor.
  // This method is only implemented in mutable tables that can be updated over
  // the execution of the graph. It returns Status::NotImplemented for read-only
  // tables that are initialized once before they can be looked up.

  // Returns the following statuses:
  // - OK: when the insert finishes successfully.
  // - InvalidArgument: if any of the preconditions on the lookup key or value
  //   fails.
  // - Unimplemented: if the table does not support insertions.
  virtual Status Insert(OpKernelContext* ctx, const Tensor& keys,
                        const Tensor& values) = 0;

  // Removes elements from the table.
  // This method is only implemented in mutable tables that can be updated over
  // the execution of the graph. It returns Status::NotImplemented for read-only
  // tables that are initialized once before they can be looked up.

  // Returns the following statuses:
  // - OK: when the remove finishes successfully.
  // - InvalidArgument: if any of the preconditions on the lookup key fails.
  // - Unimplemented: if the table does not support removals.
  virtual Status Remove(OpKernelContext* ctx, const Tensor& keys) = 0;

  // Returns the number of elements in the table.
  virtual size_t size() const = 0;

  // Exports the values of the table to two tensors named keys and values.
  // Note that the shape of the tensors is completely up to the implementation
  // of the table and can be different than the tensors used for the Insert
  // function above.
  virtual Status ExportValues(OpKernelContext* ctx) = 0;

  // Imports previously exported keys and values.
  // As mentioned above, the shape of the keys and values tensors are determined
  // by the ExportValues function above and can be different than for the
  // Insert function.
  virtual Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                              const Tensor& values) = 0;

  // Returns the data type of the key.
  virtual DataType key_dtype() const = 0;

  // Returns the data type of the value.
  virtual DataType value_dtype() const = 0;

  // Returns the shape of a key in the table.
  virtual TensorShape key_shape() const = 0;

  // Returns the shape of a value in the table.
  virtual TensorShape value_shape() const = 0;

  // Check format of the key and value tensors for the Insert function.
  // Returns OK if all the following requirements are satisfied, otherwise it
  // returns InvalidArgument:
  // - DataType of the tensor keys equals to the table key_dtype
  // - DataType of the tensor values equals to the table value_dtype
  // - the values tensor has the required shape given keys and the tables's
  //   value shape.
  virtual Status CheckKeyAndValueTensorsForInsert(const Tensor& keys,
                                                  const Tensor& values);

  // Similar to the function above but instead checks eligibility for the Import
  // function.
  virtual Status CheckKeyAndValueTensorsForImport(const Tensor& keys,
                                                  const Tensor& values);

  // Check format of the key tensor for the Remove function.
  // Returns OK if all the following requirements are satisfied, otherwise it
  // returns InvalidArgument:
  // - DataType of the tensor keys equals to the table key_dtype
  virtual Status CheckKeyTensorForRemove(const Tensor& keys);

  // Check the arguments of a find operation. Returns OK if all the following
  // requirements are satisfied, otherwise it returns InvalidArgument:
  // - DataType of the tensor keys equals to the table key_dtype
  // - DataType of the tensor default_value equals to the table value_dtype
  // - the default_value tensor has the required shape given keys and the
  //   tables's value shape.
  Status CheckFindArguments(const Tensor& keys, const Tensor& default_value);

  string DebugString() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlookup_interfaceDTh mht_0(mht_0_v, 304, "", "./tensorflow/core/framework/lookup_interface.h", "DebugString");

    return strings::StrCat("A lookup table of size: ", size());
  }

  // Returns an InitializableLookupTable, a subclass of LookupInterface, if the
  // current object is an InitializableLookupTable. Otherwise, returns nullptr.
  virtual InitializableLookupTable* GetInitializableLookupTable() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlookup_interfaceDTh mht_1(mht_1_v, 313, "", "./tensorflow/core/framework/lookup_interface.h", "GetInitializableLookupTable");

    return nullptr;
  }

 protected:
  virtual ~LookupInterface() = default;

  // Makes sure that the key and value tensor DataType's match the table
  // key_dtype and value_dtype.
  Status CheckKeyAndValueTypes(const Tensor& keys, const Tensor& values);

  // Makes sure that the provided shape is consistent with the table keys shape.
  Status CheckKeyShape(const TensorShape& shape);

 private:
  Status CheckKeyAndValueTensorsHelper(const Tensor& keys,
                                       const Tensor& values);
};

}  // namespace lookup
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_LOOKUP_INTERFACE_H_
