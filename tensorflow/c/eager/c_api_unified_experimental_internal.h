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

#ifndef TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_INTERNAL_H_
#define TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_INTERNAL_H_
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
class MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_internalDTh {
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
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_internalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_internalDTh() {
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


#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/conversion_macros.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Represents the results of the execution of an operation.
struct OutputList {
  std::vector<AbstractTensorHandle*> outputs;
  int expected_num_outputs = -1;
};

namespace tracing {

// =============================================================================
// Implementation detail for the unified execution APIs for Eager and tracing
// backends (graph/MLIR).
//
// This defines a set of abstract classes that are intended to provide the
// functionality of the opaque C types exposed in the public APIs defined in the
// `c_api_unified_experimental.h` header.
// =============================================================================

// Represents either a MlirTensor or a GraphTensor.
// This base class does not expose any public methods other than to distinguish
// which subclass it actually is. The user is responsible to use the right
// type of AbstractTensor in their context (do not pass an MlirTensor to a
// GraphContext and vice-versa).
class TracingTensorHandle : public AbstractTensorHandle {
 protected:
  explicit TracingTensorHandle(AbstractTensorHandleKind kind)
      : AbstractTensorHandle(kind) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_internalDTh mht_0(mht_0_v, 229, "", "./tensorflow/c/eager/c_api_unified_experimental_internal.h", "TracingTensorHandle");
}

 public:
  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_internalDTh mht_1(mht_1_v, 236, "", "./tensorflow/c/eager/c_api_unified_experimental_internal.h", "classof");

    return ptr->getKind() == kGraph || ptr->getKind() == kMlir;
  }
};

// An abstract operation describes an operation by its type, name, and
// attributes. It can be "executed" by the context with some input tensors.
// It is allowed to reusing the same abstract operation for multiple execution
// on a given context, with the same or different input tensors.
class TracingOperation : public AbstractOperation {
 protected:
  explicit TracingOperation(AbstractOperationKind kind)
      : AbstractOperation(kind) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_internalDTh mht_2(mht_2_v, 251, "", "./tensorflow/c/eager/c_api_unified_experimental_internal.h", "TracingOperation");
}

 public:
  // Sets the name of the operation: this is an optional identifier that is
  // not intended to carry semantics and preserved/propagated without
  // guarantees.
  virtual Status SetOpName(const char* op_name) = 0;

  // For LLVM style RTTI.
  static bool classof(const AbstractOperation* ptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_internalDTh mht_3(mht_3_v, 263, "", "./tensorflow/c/eager/c_api_unified_experimental_internal.h", "classof");

    return ptr->getKind() == kGraph || ptr->getKind() == kMlir;
  }
};

namespace internal {
struct TracingOperationDeleter {
  void operator()(TracingOperation* p) const {
    if (p != nullptr) {
      p->Release();
    }
  }
};
}  // namespace internal

using TracingOperationPtr =
    std::unique_ptr<TracingOperation, internal::TracingOperationDeleter>;

// This holds the context for the execution: dispatching operations either to an
// MLIR implementation or to a graph implementation.
class TracingContext : public AbstractContext {
 protected:
  explicit TracingContext(AbstractContextKind kind) : AbstractContext(kind) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_internalDTh mht_4(mht_4_v, 288, "", "./tensorflow/c/eager/c_api_unified_experimental_internal.h", "TracingContext");
}

 public:
  // Add a function parameter and return the corresponding tensor.
  virtual Status AddParameter(DataType dtype, const PartialTensorShape& shape,
                              TracingTensorHandle**) = 0;

  // Finalize this context and make a function out of it. The context is in a
  // invalid state after this call and must be destroyed.
  virtual Status Finalize(OutputList* outputs, AbstractFunction**) = 0;

  // For LLVM style RTTI.
  static bool classof(const AbstractContext* ptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimental_internalDTh mht_5(mht_5_v, 303, "", "./tensorflow/c/eager/c_api_unified_experimental_internal.h", "classof");

    return ptr->getKind() == kGraph || ptr->getKind() == kMlir;
  }
};

typedef TracingContext* (*FactoryFunction)(const char* fn_name, TF_Status*);
Status SetDefaultTracingEngine(const char* name);
void RegisterTracingEngineFactory(const ::tensorflow::string& name,
                                  FactoryFunction factory);
}  // namespace tracing

DEFINE_CONVERSION_FUNCTIONS(AbstractContext, TF_ExecutionContext)
DEFINE_CONVERSION_FUNCTIONS(AbstractTensorHandle, TF_AbstractTensor)
DEFINE_CONVERSION_FUNCTIONS(AbstractFunction, TF_AbstractFunction)
DEFINE_CONVERSION_FUNCTIONS(AbstractOperation, TF_AbstractOp)
DEFINE_CONVERSION_FUNCTIONS(OutputList, TF_OutputList)
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_INTERNAL_H_
