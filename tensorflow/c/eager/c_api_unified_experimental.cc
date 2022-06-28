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
class MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc() {
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

#include "tensorflow/c/eager/c_api_unified_experimental.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::string;

namespace tensorflow {
namespace tracing {
typedef absl::flat_hash_map<std::string, tracing::FactoryFunction> FactoriesMap;

static FactoriesMap& GetFactories() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_0(mht_0_v, 206, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "GetFactories");

  static FactoriesMap* factories = new FactoriesMap;
  return *factories;
}

static tracing::FactoryFunction default_factory;

void RegisterTracingEngineFactory(const string& name, FactoryFunction factory) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_1(mht_1_v, 217, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "RegisterTracingEngineFactory");

  assert((!GetFactories().count(name)) ||
         (GetFactories()[name] == factory) &&
             "Duplicate tracing factory registration");
  GetFactories()[name] = factory;
}

Status SetDefaultTracingEngine(const char* name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_2(mht_2_v, 228, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "SetDefaultTracingEngine");

  auto entry = GetFactories().find(name);
  if (entry != GetFactories().end()) {
    default_factory = GetFactories().find(name)->second;
    return Status::OK();
  }
  string msg = absl::StrCat(
      "No tracing engine factory has been registered with the key '", name,
      "' (available: ");
  // Ensure deterministic (sorted) order in the error message
  std::set<string> factories_sorted;
  for (const auto& factory : GetFactories())
    factories_sorted.insert(factory.first);
  const char* comma = "";
  for (const string& factory : factories_sorted) {
    msg += comma + factory;
    comma = ", ";
  }
  msg += ")";

  return errors::InvalidArgument(msg.c_str());
}

static TracingContext* CreateTracingExecutionContext(const char* fn_name,
                                                     TF_Status* s) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("fn_name: \"" + (fn_name == nullptr ? std::string("nullptr") : std::string((char*)fn_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_3(mht_3_v, 256, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "CreateTracingExecutionContext");

  if (default_factory) {
    return default_factory(fn_name, s);
  }
  Set_TF_Status_from_Status(
      s, errors::FailedPrecondition("default_factory is nullptr"));
  return nullptr;
}

}  // end namespace tracing
}  // end namespace tensorflow

// =============================================================================
// Public C API entry points
//
// These are only the generic entry points for the C API. This file does not
// have any visibility into the graph/eager implementation and is only providing
// C bindings to the abstract classes defined in the
// c_api_unified_experimental_internal.h header.
//
// =============================================================================

using tensorflow::AbstractFunction;
using tensorflow::AbstractTensorHandle;
using tensorflow::DataType;
using tensorflow::dyn_cast;
using tensorflow::OutputList;
using tensorflow::Status;
using tensorflow::unwrap;
using tensorflow::wrap;
using tensorflow::tracing::CreateTracingExecutionContext;
using tensorflow::tracing::SetDefaultTracingEngine;
using tensorflow::tracing::TracingContext;
using tensorflow::tracing::TracingOperation;
using tensorflow::tracing::TracingTensorHandle;

void TF_SetTracingImplementation(const char* name, TF_Status* s) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_4(mht_4_v, 296, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_SetTracingImplementation");

  Set_TF_Status_from_Status(s, SetDefaultTracingEngine(name));
}

// Creates a new TensorFlow function, it is an execution context attached to a
// given tracing context.
TF_ExecutionContext* TF_CreateFunction(const char* fn_name, TF_Status* s) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("fn_name: \"" + (fn_name == nullptr ? std::string("nullptr") : std::string((char*)fn_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_5(mht_5_v, 306, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_CreateFunction");

  return wrap(CreateTracingExecutionContext(fn_name, s));
}

TF_AbstractFunction* TF_FinalizeFunction(TF_ExecutionContext* ctx,
                                         TF_OutputList* outputs, TF_Status* s) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_6(mht_6_v, 314, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_FinalizeFunction");

  AbstractFunction* func;
  TracingContext* tracing_ctx = dyn_cast<TracingContext>(unwrap(ctx));
  if (!tracing_ctx) {
    Set_TF_Status_from_Status(
        s, tensorflow::errors::InvalidArgument(
               "Only TracingContext can be converted into a function."));
    return nullptr;
  }
  Set_TF_Status_from_Status(s, tracing_ctx->Finalize(unwrap(outputs), &func));
  TF_DeleteExecutionContext(ctx);
  return wrap(func);
}

TF_AbstractTensor* TF_AddFunctionParameter(TF_ExecutionContext* func,
                                           TF_DataType dtype, TF_Shape shape,
                                           TF_Status* s) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_7(mht_7_v, 333, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_AddFunctionParameter");

  DCHECK_GE(shape.num_dims, -1);
  TracingTensorHandle* t;
  TracingContext* tracing_ctx = dyn_cast<TracingContext>(unwrap(func));
  if (!tracing_ctx) {
    Set_TF_Status_from_Status(
        s, tensorflow::errors::InvalidArgument(
               "TF_AddFunctionParameter must be called on a TracingContext."));
    return nullptr;
  }
  tensorflow::PartialTensorShape partial_shape;
  if (shape.num_dims != -1) {
    DCHECK(shape.dim_sizes != nullptr);
    Status status = tensorflow::PartialTensorShape::MakePartialShape(
        reinterpret_cast<int64_t*>(shape.dim_sizes), shape.num_dims,
        &partial_shape);
    if (!status.ok()) {
      Set_TF_Status_from_Status(s, status);
      return nullptr;
    }
  }
  Set_TF_Status_from_Status(
      s, tracing_ctx->AddParameter(static_cast<DataType>(dtype), partial_shape,
                                   &t));
  return wrap(t);
}

void TF_DeleteExecutionContext(TF_ExecutionContext* c) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_8(mht_8_v, 363, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_DeleteExecutionContext");
 unwrap(c)->Release(); }

TF_AbstractOp* TF_NewAbstractOp(TF_ExecutionContext* c) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_9(mht_9_v, 368, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_NewAbstractOp");

  return wrap((unwrap(c)->CreateOperation()));
}

void TF_DeleteAbstractOp(TF_AbstractOp* op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_10(mht_10_v, 375, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_DeleteAbstractOp");
 unwrap(op)->Release(); }

void TF_DeleteAbstractTensor(TF_AbstractTensor* t) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_11(mht_11_v, 380, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_DeleteAbstractTensor");
 unwrap(t)->Unref(); }

TF_OutputList* TF_NewOutputList() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_12(mht_12_v, 385, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_NewOutputList");
 return wrap(new OutputList); }
void TF_DeleteOutputList(TF_OutputList* o) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_13(mht_13_v, 389, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_DeleteOutputList");
 delete unwrap(o); }
void TF_OutputListSetNumOutputs(TF_OutputList* o, int num_outputs,
                                TF_Status* s) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_14(mht_14_v, 394, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_OutputListSetNumOutputs");

  unwrap(o)->expected_num_outputs = num_outputs;
  unwrap(o)->outputs.clear();
  unwrap(o)->outputs.resize(num_outputs);
}
int TF_OutputListNumOutputs(TF_OutputList* o) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_15(mht_15_v, 402, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_OutputListNumOutputs");

  return unwrap(o)->outputs.size();
}
TF_AbstractTensor* TF_OutputListGet(TF_OutputList* o, int i) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_16(mht_16_v, 408, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_OutputListGet");

  return wrap(unwrap(o)->outputs[i]);
}
void TF_OutputListPushBack(TF_OutputList* o, TF_AbstractTensor* tensor,
                           TF_Status* s) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_17(mht_17_v, 415, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_OutputListPushBack");

  unwrap(o)->outputs.push_back(unwrap(tensor));
}

void TF_AbstractOpSetOpType(TF_AbstractOp* op, const char* const op_type,
                            TF_Status* s) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_18(mht_18_v, 423, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_AbstractOpSetOpType");

  Set_TF_Status_from_Status(s, unwrap(op)->Reset(op_type,
                                                 /*raw_device_name=*/nullptr));
}

void TF_AbstractOpSetOpName(TF_AbstractOp* op, const char* const op_name,
                            TF_Status* s) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_19(mht_19_v, 432, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_AbstractOpSetOpName");

  TracingOperation* tracing_op = dyn_cast<TracingOperation>(unwrap(op));
  if (!tracing_op) {
    Set_TF_Status_from_Status(
        s, tensorflow::errors::InvalidArgument(
               "TF_AbstractOpSetOpName must be called on a TracingOperation."));
    return;
  }
  Set_TF_Status_from_Status(s, tracing_op->SetOpName(op_name));
}

void TF_AbstractOpSetAttrType(TF_AbstractOp* op, const char* const attr_name,
                              TF_DataType value, TF_Status* s) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_20(mht_20_v, 447, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_AbstractOpSetAttrType");

  Status status =
      unwrap(op)->SetAttrType(attr_name, static_cast<DataType>(value));
  TF_SetStatus(s, static_cast<TF_Code>(status.code()),
               status.error_message().c_str());
}

void TF_ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                         TF_AbstractTensor* const* inputs, TF_OutputList* o,
                         TF_Status* s) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_21(mht_21_v, 459, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_ExecuteOperation");

  for (int i = 0; i < num_inputs; i++) {
    Set_TF_Status_from_Status(s, unwrap(op)->AddInput(unwrap(inputs[i])));
    if (TF_GetCode(s) != TF_OK) {
      return;
    }
  }
  int num_outputs = unwrap(o)->expected_num_outputs;
  Set_TF_Status_from_Status(
      s, unwrap(op)->Execute(
             absl::MakeSpan(reinterpret_cast<AbstractTensorHandle**>(
                                unwrap(o)->outputs.data()),
                            unwrap(o)->outputs.size()),
             &num_outputs));
}

void TF_DeleteAbstractFunction(TF_AbstractFunction* func) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_22(mht_22_v, 478, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_DeleteAbstractFunction");

  unwrap(func)->Unref();
}

void TF_ExecutionContextRegisterFunction(TF_ExecutionContext* ctx,
                                         TF_AbstractFunction* func,
                                         TF_Status* s) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_unified_experimentalDTcc mht_23(mht_23_v, 487, "", "./tensorflow/c/eager/c_api_unified_experimental.cc", "TF_ExecutionContextRegisterFunction");

  Set_TF_Status_from_Status(s, unwrap(ctx)->RegisterFunction(unwrap(func)));
}
