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
class MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/experimental/libtf/function.h"

#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/cc/experimental/libtf/value_iostream.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

namespace tf {
namespace libtf {

using tensorflow::AbstractContext;
using tensorflow::AbstractFunctionPtr;
using tensorflow::AbstractOperationPtr;
using tensorflow::AbstractTensorHandle;
using tensorflow::Status;
using tensorflow::StatusOr;

// TODO(srbs): Move this to unified execution API.
tensorflow::Status ExecuteFunction(
    AbstractFunctionPtr trace, AbstractContext* ctx,
    absl::Span<tensorflow::AbstractTensorHandle* const> inputs,
    absl::Span<tensorflow::AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc mht_0(mht_0_v, 215, "", "./tensorflow/cc/experimental/libtf/function.cc", "ExecuteFunction");

  // TODO(srbs): Provide a function execution API on ctx so that we do not
  // expose the internals of how functions are to be executed here.
  std::string fname;
  {
    tensorflow::FunctionDef* fdef = nullptr;
    TF_RETURN_IF_ERROR(trace->GetFunctionDef(&fdef));
    fname = fdef->signature().name();
  }
  // TODO(srbs): Update RegisterFunction to accept AbstractFunctionPtr.
  TF_RETURN_IF_ERROR(ctx->RegisterFunction(trace.get()));
  auto cleanup = absl::MakeCleanup(
      [fname, ctx]() { ctx->RemoveFunction(fname).IgnoreError(); });
  auto call_op = AbstractOperationPtr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      call_op->Reset(fname.c_str(), /*raw_device_name=*/nullptr));
  for (auto t : inputs) {
    TF_RETURN_IF_ERROR(call_op->AddInput(t));
  }
  int num_outputs = outputs.size();
  return call_op->Execute(outputs, &num_outputs);
}

Status VerifySupportedSignature(TaggedValue signature) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc mht_1(mht_1_v, 241, "", "./tensorflow/cc/experimental/libtf/function.cc", "VerifySupportedSignature");

  if (signature.type() == TaggedValue::Type::TENSOR_SPEC) {
    return Status::OK();
  }
  if (signature.type() == TaggedValue::Type::TUPLE) {
    for (const auto& t : signature.tuple()) {
      if (t.type() != TaggedValue::Type::TENSOR_SPEC) {
        break;
      }
    }
    return Status::OK();
  }
  return tensorflow::errors::Unimplemented(
      "Only functions with inputs/outputs containing a single tensor or a tuple"
      " of tensors are supported right now.");
}

Status VerifySupportedArgs(TaggedValue args) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc mht_2(mht_2_v, 261, "", "./tensorflow/cc/experimental/libtf/function.cc", "VerifySupportedArgs");

  if (args.type() == TaggedValue::Type::TENSOR) {
    return Status::OK();
  }
  if (args.type() == TaggedValue::Type::TUPLE) {
    for (const auto& t : args.tuple()) {
      if (t.type() != TaggedValue::Type::TENSOR) {
        break;
      }
    }
    return Status::OK();
  }
  return tensorflow::errors::Unimplemented(
      "Only functions with inputs/outputs containing a single tensor or a tuple"
      " of tensors are supported right now.");
}

Status Function::RegisterTrace(AbstractFunctionPtr fn,
                               TaggedValue input_signature,
                               TaggedValue output_signature) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc mht_3(mht_3_v, 283, "", "./tensorflow/cc/experimental/libtf/function.cc", "Function::RegisterTrace");

  TF_RETURN_IF_ERROR(VerifySupportedSignature(input_signature));
  TF_RETURN_IF_ERROR(VerifySupportedSignature(output_signature));
  concrete_fns_.push_back({fn, input_signature, output_signature});
  return Status::OK();
}

bool Match(TaggedValue signature, TaggedValue value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc mht_4(mht_4_v, 293, "", "./tensorflow/cc/experimental/libtf/function.cc", "Match");

  // TODO(b/187216309): Extend this to handle more elaborate signatures and
  // values.
  switch (signature.type()) {
    case TaggedValue::Type::TENSOR_SPEC: {
      if (value.type() != TaggedValue::Type::TENSOR) {
        return false;
      }
      auto spec = signature.tensor_spec();
      const auto& tensor = value.tensor();
      if (tensor->DataType() != spec.dtype) {
        return false;
      }
      tensorflow::PartialTensorShape tensor_shape;
      DCHECK(tensor->Shape(&tensor_shape).ok());
      if (!tensor_shape.IsCompatibleWith(spec.shape)) {
        return false;
      }
    } break;
    case TaggedValue::Type::TUPLE: {
      if (value.type() != TaggedValue::Type::TUPLE) {
        return false;
      }
      if (value.tuple().size() != signature.tuple().size()) {
        return false;
      }
      for (auto i = 0; i < value.tuple().size(); i++) {
        if (!Match(signature.tuple()[i], value.tuple()[i])) {
          return false;
        }
      }
    } break;
    default:
      return false;
  }
  return true;
}

// TODO(b/190203981): Move to a separate nest-like library.
void Flatten(const TaggedValue& value,
             std::vector<AbstractTensorHandle*>* flat_args) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc mht_5(mht_5_v, 336, "", "./tensorflow/cc/experimental/libtf/function.cc", "Flatten");

  if (value.type() == TaggedValue::Type::TENSOR) {
    flat_args->emplace_back(value.tensor().get());
  } else if (value.type() == TaggedValue::Type::TUPLE) {
    for (const auto& t : value.tuple()) {
      Flatten(t, flat_args);
    }
  } else {
    // TODO(b/190203981): Supported arbitrary structures.
    LOG(ERROR) << "Unimplemented";
  }
}

StatusOr<TaggedValue> Unflatten(
    absl::Span<AbstractTensorHandle* const> flat_args, TaggedValue structure) {
  if (structure.type() == TaggedValue::Type::TENSOR_SPEC) {
    if (flat_args.size() != 1) {
      // Denotes a corrupted SavedModel in which output_signature does not match
      // FunctionDef outputs.
      return tensorflow::errors::Internal("Expected single tensor but found ",
                                          flat_args.size());
    }
    TaggedValue wrapped_t =
        TaggedValue(impl::TaggedValueTensor(flat_args[0], /*add_ref=*/true));
    if (!Match(structure, wrapped_t)) {
      // Denotes a corrupted SavedModel in which output_signature does not match
      // FunctionDef outputs.
      std::stringstream stream;
      stream << "Shape and dtype of tensor " << wrapped_t
             << " does not match that in signature " << structure;
      return tensorflow::errors::Internal(stream.str());
    }
    return wrapped_t;
  } else if (structure.type() == TaggedValue::Type::TUPLE) {
    // TODO(b/190203981): Remove this check when handling nested structures
    // inside tuples.
    if (flat_args.size() != structure.tuple().size()) {
      return tensorflow::errors::InvalidArgument(
          "Tuple length ", structure.tuple().size(),
          " does not match length of flat args ", flat_args.size());
    }
    auto result = impl::TaggedValue::Tuple();
    for (auto i = 0; i < structure.tuple().size(); i++) {
      TF_ASSIGN_OR_RETURN(TaggedValue ele,
                          Unflatten({flat_args[i]}, structure.tuple()[i]));
      result.tuple().emplace_back(std::move(ele));
    }
    return result;
  } else {
    // TODO(b/190203981): Support arbitrary structures.
    return tensorflow::errors::Unimplemented(
        "Only tensors and tuples of tensors are supported right now.");
  }
}

size_t GetFlatSize(const TaggedValue& value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc mht_6(mht_6_v, 394, "", "./tensorflow/cc/experimental/libtf/function.cc", "GetFlatSize");

  if (value.type() == TaggedValue::Type::TUPLE) {
    size_t result = 0;
    for (const auto& t : value.tuple()) {
      result += GetFlatSize(t);
    }
    return result;
  } else if (value.type() == TaggedValue::Type::LIST) {
    size_t result = 0;
    for (const auto& t : value.list()) {
      result += GetFlatSize(t);
    }
    return result;
  } else if (value.type() == TaggedValue::Type::DICT) {
    size_t result = 0;
    for (const auto& t : value.dict()) {
      result += GetFlatSize(t.second);
    }
    return result;
  }
  return 1;
}

StatusOr<TaggedValue> Function::Execute(AbstractContext* ctx,
                                        TaggedValue value) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc mht_7(mht_7_v, 421, "", "./tensorflow/cc/experimental/libtf/function.cc", "Function::Execute");

  TF_RETURN_IF_ERROR(VerifySupportedArgs(value));
  TF_ASSIGN_OR_RETURN(auto concrete_fn, GetConcreteFunction(value));
  std::vector<AbstractTensorHandle*> args;
  Flatten(value, &args);
  std::vector<AbstractTensorHandle*> outs(
      GetFlatSize(concrete_fn.output_signature));
  TF_RETURN_IF_ERROR(
      ExecuteFunction(concrete_fn.trace, ctx, args, absl::MakeSpan(outs)));
  auto cleanup_tensors = absl::MakeCleanup([outs]() {
    for (auto t : outs) {
      t->Unref();
    }
  });
  return Unflatten(outs, concrete_fn.output_signature);
}

StatusOr<Function::ConcreteFunction> Function::GetConcreteFunction(
    TaggedValue value) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSfunctionDTcc mht_8(mht_8_v, 442, "", "./tensorflow/cc/experimental/libtf/function.cc", "Function::GetConcreteFunction");

  if (concrete_fns_.empty()) {
    return tensorflow::errors::FailedPrecondition(
        "No registered ConcreteFunctions.");
  }
  for (auto& spec : concrete_fns_) {
    if (Match(spec.input_signature, value)) {
      return spec;
    }
  }
  return tensorflow::errors::InvalidArgument("No match found.");
}

}  // namespace libtf
}  // namespace tf
