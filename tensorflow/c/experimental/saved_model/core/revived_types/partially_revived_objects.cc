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
class MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc() {
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

#include "tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/restored_resource.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/restored_resource_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/revived_objects.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_signature_def_function.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_signature_def_function_revival_state.h"
#include "tensorflow/c/experimental/saved_model/core/signature_def_function_metadata.h"
#include "tensorflow/c/experimental/saved_model/core/tensor_spec.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace tensorflow {

namespace {

using StructuredValueDictEntry =
    protobuf::MapPair<std::string, StructuredValue>;

using NamedParamMap =
    gtl::FlatMap<StringPiece, const TensorSpecProto*, StringPieceHasher>;

Status AssertAllCreateResourceFunctionsHaveNoCaptures(
    const PartiallyRevivedObjects& objects) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_0(mht_0_v, 225, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "AssertAllCreateResourceFunctionsHaveNoCaptures");

  for (const auto& id_and_resource : objects.restored_resources) {
    int node_id = id_and_resource.first;
    const RestoredResourceRevivalState& resource = id_and_resource.second;
    const TFConcreteFunctionRevivalState* create_resource_fn =
        resource.create_resource;
    if (create_resource_fn == nullptr) {
      return errors::FailedPrecondition(
          "Resource at node ", node_id,
          " did not have a create_resource() function");
    }
    const SavedConcreteFunction* saved_create_resource_fn =
        create_resource_fn->saved_concrete_func;
    if (!saved_create_resource_fn->bound_inputs().empty()) {
      // TODO(b/124045874): Support loading resource functions via a top sort
      return errors::Unimplemented(
          "Create Resource functions with captures are currently unsupported.");
    }
  }
  return Status();
}

// Retrieves the TensorHandle associated with `node_id` from `obj_graph`, and
// set `*handle` to point to it.
Status TensorHandleFromNode(int node_id, const SavedObjectGraph& obj_graph,
                            const PartiallyRevivedObjects& objects,
                            ImmediateExecutionTensorHandle** handle) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_1(mht_1_v, 254, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "TensorHandleFromNode");

  const SavedObject& node = obj_graph.nodes(node_id);
  SavedObject::KindCase kind = node.kind_case();
  switch (kind) {
    case SavedObject::kVariable: {
      const auto& variables_iter = objects.variables.find(node_id);
      if (variables_iter == objects.variables.end()) {
        return errors::FailedPrecondition(
            "Tried to convert node id ", node_id,
            " of type variable to tensor but the variable wasn't initialized");
      }
      *handle = variables_iter->second->handle();
      return Status();
    }
    case SavedObject::kConstant: {
      const auto& constants_iter = objects.constants.find(node_id);
      if (constants_iter == objects.constants.end()) {
        return errors::FailedPrecondition("Tried to convert node id ", node_id,
                                          " of type constant to tensor but the "
                                          "constant wasn't initialized");
      }
      *handle = constants_iter->second->handle();
      return Status();
    }
    case SavedObject::kAsset: {
      const auto& assets_iter = objects.assets.find(node_id);
      if (assets_iter == objects.assets.end()) {
        return errors::FailedPrecondition(
            "Tried to convert node id ", node_id,
            " of type asset to tensor but the asset wasn't initialized");
      }
      *handle = assets_iter->second->handle();
      return Status();
    }
    case SavedObject::kResource: {
      const auto& resource_iter = objects.restored_resources.find(node_id);
      if (resource_iter == objects.restored_resources.end()) {
        return errors::FailedPrecondition(
            "Tried to convert node id ", node_id,
            " of type Resource to tensor but the Resource wasn't initialized");
      }
      const RestoredResourceRevivalState& resource = resource_iter->second;
      if (resource.resource_handle == nullptr) {
        return errors::FailedPrecondition(
            "Resource with node id ", node_id,
            " should have its resource_handle created, but was nullptr.");
      }
      *handle = resource.resource_handle.get();
      return Status();
    }
    default: {
      return errors::FailedPrecondition(
          "Only objects of type variable, constant, asset, and resources have "
          "capturable tensorhandles. Encountered object of kind ",
          node.kind_case(), " at node id: ", node_id);
    }
  }
}

std::vector<SignatureDefParam> SignatureDefParamsFromNamedParamMap(
    const NamedParamMap& params) {
  // The underlying functiondef associated with the SignatureDef has
  // nest.flattened inputs and outputs, which are sorted by string key.
  std::vector<SignatureDefParam> result;
  result.reserve(params.size());
  for (const auto& named_param : params) {
    result.push_back(SignatureDefParam(std::string(named_param.first),
                                       TensorSpec(*named_param.second)));
  }
  std::sort(result.begin(), result.end(),
            [](const SignatureDefParam& x, const SignatureDefParam& y) {
              return x.name() < y.name();
            });

  return result;
}

// SignatureDefArgsFromInputs takes the "canonicalized_input_signature"
// field of a SavedConcreteFunction, ensures it conforms to the structure of
// tuple(tuple(), dict<string,TensorSpec>()), and "returns" a list of
// SignatureDefParams of the SignatureDefFunction's arguments.
Status SignatureDefArgsFromInputs(
    const StructuredValue& canonicalized_input_signature,
    std::vector<SignatureDefParam>* out) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_2(mht_2_v, 340, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "SignatureDefArgsFromInputs");

  // Note(bmzhao): canonicalized_input_signature should be a tuple of
  // (args, kwargs), where args is an empty tuple, and kwargs is a dictionary of
  // string keys to TensorSpecs.
  if (!canonicalized_input_signature.has_tuple_value()) {
    return errors::FailedPrecondition(
        "SignatureDefFunction's canonicalized_input_signature should be "
        "of form tuple(tuple(), dict()), but was instead: \n",
        canonicalized_input_signature.DebugString());
  }

  const TupleValue& args_kwargs_tuple =
      canonicalized_input_signature.tuple_value();
  if (args_kwargs_tuple.values_size() != 2) {
    return errors::FailedPrecondition(
        "SignatureDefFunction's canonicalized_input_signature should be "
        "a tuple of two elements (args, kwargs), but was instead: \n",
        args_kwargs_tuple.DebugString());
  }

  const StructuredValue& args = args_kwargs_tuple.values(0);
  if (!args.has_tuple_value() || !args.tuple_value().values().empty()) {
    return errors::FailedPrecondition(
        "SignatureDefFunction's canonicalized_input_signature's args"
        "should be an empty tuple, but instead got: \n",
        args.DebugString());
  }

  const StructuredValue& kwargs = args_kwargs_tuple.values(1);
  if (!kwargs.has_dict_value()) {
    return errors::FailedPrecondition(
        "SignatureDefFunction's canonicalized_input_signature's kwargs"
        "should be a dictionary, but instead got: \n",
        kwargs.DebugString());
  }

  const DictValue& kwargs_dict = kwargs.dict_value();
  NamedParamMap result;
  result.reserve(kwargs_dict.fields_size());

  for (const auto& key_value : kwargs_dict.fields()) {
    const std::string& key = key_value.first;
    const StructuredValue& value = key_value.second;
    if (!value.has_tensor_spec_value()) {
      return errors::FailedPrecondition(
          "SignatureDefFunction's canonicalized_input_signature's kwargs"
          "dictionary contained a non-tensorspec value for key-value pair: \n",
          "Key: ", key, "Value: \n", value.DebugString());
    }
    result[key] = &value.tensor_spec_value();
  }

  *out = SignatureDefParamsFromNamedParamMap(result);

  return Status();
}

// SignatureDefReturnsFromOutputs takes the "output_signature" field of a
// SavedConcreteFunction, ensures it conforms to the structure of
// dict<string,TensorSpec>(), and "returns" a list of SignatureDefParams of the
// SignatureDefFunction's returns.
Status SignatureDefReturnsFromOutputs(const StructuredValue& output_signature,
                                      std::vector<SignatureDefParam>* out) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_3(mht_3_v, 405, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "SignatureDefReturnsFromOutputs");

  if (!output_signature.has_dict_value()) {
    return errors::FailedPrecondition(
        "SignatureDefFunction's output_signature must be a dictionary, but "
        "instead got: ",
        output_signature.DebugString());
  }

  const DictValue& output_dict = output_signature.dict_value();
  NamedParamMap result;
  result.reserve(output_dict.fields_size());

  for (const auto& key_value : output_dict.fields()) {
    const std::string& key = key_value.first;
    const StructuredValue& value = key_value.second;
    if (!value.has_tensor_spec_value()) {
      return errors::FailedPrecondition(
          "SignatureDefFunction's output_signature dictionary contained a "
          "non-tensorspec value for key-value pair: \n",
          "Key: ", key, "Value: \n", value.DebugString());
    }
    result[key] = &value.tensor_spec_value();
  }
  *out = SignatureDefParamsFromNamedParamMap(result);

  return Status();
}

// The implementation takes advantage of the fact that SignatureDefFunction's
// "traced" Signature wrapper function always has inputs/outputs of dictionaries
// https://github.com/tensorflow/tensorflow/blob/53cdd5e87c423b195f33775753273286fd5a1a65/tensorflow/python/saved_model/signature_serialization.py#L119-L126
// https://github.com/tensorflow/tensorflow/blob/53cdd5e87c423b195f33775753273286fd5a1a65/tensorflow/python/saved_model/signature_serialization.py#L153-L178
// Additionally, we take advantage of the fact that the SignatureDefFunction's
// associated functiondef has lexicographically ordered inputs/outputs due to
// nest.flatten.
Status LoadSignatureDefFunctionMetadata(
    const SavedConcreteFunction& saved_concrete_function,
    SignatureDefFunctionMetadata* out) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_4(mht_4_v, 445, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "LoadSignatureDefFunctionMetadata");

  std::vector<SignatureDefParam> args;
  TF_RETURN_IF_ERROR(SignatureDefArgsFromInputs(
      saved_concrete_function.canonicalized_input_signature(), &args));

  std::vector<SignatureDefParam> rets;
  TF_RETURN_IF_ERROR(SignatureDefReturnsFromOutputs(
      saved_concrete_function.output_signature(), &rets));

  *out = SignatureDefFunctionMetadata(std::move(args), std::move(rets));
  return Status();
}

// This function finds the necessary captures, then forwards to the builder
// method
Status CreateConcreteFunction(ImmediateExecutionContext* ctx,
                              const TFConcreteFunctionRevivalState& builder,
                              const SavedObjectGraph& obj_graph,
                              const PartiallyRevivedObjects& objects,
                              std::unique_ptr<TFConcreteFunction>* out) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_5(mht_5_v, 467, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "CreateConcreteFunction");

  const auto& capture_node_ids = builder.saved_concrete_func->bound_inputs();
  std::vector<ImmediateExecutionTensorHandle*> captures;
  captures.reserve(capture_node_ids.size());
  for (int capture_node_id : capture_node_ids) {
    ImmediateExecutionTensorHandle* capture_handle;
    TF_RETURN_IF_ERROR(TensorHandleFromNode(capture_node_id, obj_graph, objects,
                                            &capture_handle));
    captures.push_back(capture_handle);
  }
  // TODO(bmzhao): Create Metadata here
  return TFConcreteFunction::Create(/*function_def=*/builder.fdef,
                                    /*captures=*/std::move(captures),
                                    /*metadata=*/{},
                                    /*ctx=*/ctx,
                                    /*out=*/out);
}

Status CreateSignatureDefFunction(
    ImmediateExecutionContext* ctx,
    const TFSignatureDefFunctionRevivalState& builder,
    const SavedObjectGraph& obj_graph, const PartiallyRevivedObjects& objects,
    std::unique_ptr<TFSignatureDefFunction>* out) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_6(mht_6_v, 492, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "CreateSignatureDefFunction");

  const auto& capture_node_ids = builder.saved_concrete_func->bound_inputs();
  std::vector<ImmediateExecutionTensorHandle*> captures;
  captures.reserve(capture_node_ids.size());
  for (int capture_node_id : capture_node_ids) {
    ImmediateExecutionTensorHandle* capture_handle;
    TF_RETURN_IF_ERROR(TensorHandleFromNode(capture_node_id, obj_graph, objects,
                                            &capture_handle));
    captures.push_back(capture_handle);
  }

  SignatureDefFunctionMetadata metadata;
  TF_RETURN_IF_ERROR(LoadSignatureDefFunctionMetadata(
      *builder.saved_concrete_func, &metadata));

  return TFSignatureDefFunction::Create(/*function_def=*/builder.fdef,
                                        /*captures=*/std::move(captures),
                                        /*metadata=*/std::move(metadata),
                                        /*ctx=*/ctx,
                                        /*out=*/out);
}

Status InitializeCreateResourceFunctions(ImmediateExecutionContext* ctx,
                                         const SavedObjectGraph& obj_graph,
                                         const PartiallyRevivedObjects& objects,
                                         RevivedObjects* revived) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_7(mht_7_v, 520, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "InitializeCreateResourceFunctions");

  for (const auto& id_and_resource : objects.restored_resources) {
    const RestoredResourceRevivalState& resource = id_and_resource.second;
    const TFConcreteFunctionRevivalState* create_resource_fn =
        resource.create_resource;

    const SavedConcreteFunction* saved_create_resource_fn =
        create_resource_fn->saved_concrete_func;
    if (!saved_create_resource_fn->bound_inputs().empty()) {
      // TODO(b/124045874): Load resource functions via a topological sort
      return errors::Unimplemented(
          "Create Resource functions with captures are currently unsupported.");
    }
    std::unique_ptr<TFConcreteFunction> out;
    TF_RETURN_IF_ERROR(CreateConcreteFunction(ctx, *create_resource_fn,
                                              obj_graph, objects, &out));
    revived->concrete_functions.Insert(std::move(out),
                                       create_resource_fn->node_id);
  }
  return Status();
}

Status InitializeAllFunctions(ImmediateExecutionContext* ctx,
                              const SavedObjectGraph& obj_graph,
                              const PartiallyRevivedObjects& objects,
                              RevivedObjects* revived) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_8(mht_8_v, 548, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "InitializeAllFunctions");

  gtl::FlatMap<int, std::unique_ptr<TFSignatureDefFunction>>*
      destination_sig_map = &revived->signature_def_functions;

  for (const auto& id_and_func : objects.concrete_functions) {
    int node_id = id_and_func.first;
    const TFConcreteFunctionRevivalState& func = id_and_func.second;

    if (revived->concrete_functions.Find(node_id)) {
      // The function has already been initialized in the destination_map,
      // so we can skip this node. This can occur because we initialize
      // CreateResource functions before calling this function.
      continue;
    }

    std::unique_ptr<TFConcreteFunction> out;
    TF_RETURN_IF_ERROR(
        CreateConcreteFunction(ctx, func, obj_graph, objects, &out));
    revived->concrete_functions.Insert(std::move(out), node_id);
  }

  for (const auto& id_and_func : objects.signature_def_functions) {
    int node_id = id_and_func.first;
    const TFSignatureDefFunctionRevivalState& func = id_and_func.second;

    if (destination_sig_map->find(node_id) != destination_sig_map->end()) {
      continue;
    }

    std::unique_ptr<TFSignatureDefFunction> out;
    TF_RETURN_IF_ERROR(
        CreateSignatureDefFunction(ctx, func, obj_graph, objects, &out));
    (*destination_sig_map)[node_id] = std::move(out);
  }

  return Status();
}

Status CreateAllResourceHandles(ImmediateExecutionContext* ctx,
                                const SavedObjectGraph& obj_graph,
                                PartiallyRevivedObjects* objects,
                                RevivedObjects* revived) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_9(mht_9_v, 592, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "CreateAllResourceHandles");

  for (auto& id_and_resource : objects->restored_resources) {
    RestoredResourceRevivalState& resource = id_and_resource.second;
    int create_resource_fn_node = resource.create_resource->node_id;

    const TFConcreteFunction* create_resource_fn =
        revived->concrete_functions.Find(create_resource_fn_node);
    if (create_resource_fn == nullptr) {
      return errors::FailedPrecondition(
          "ConcreteFunction at node ", create_resource_fn_node,
          " should have been initialized prior to being called.");
    }
    ImmediateOpPtr function_op;
    TF_RETURN_IF_ERROR(create_resource_fn->MakeCallOp({}, &function_op));
    TF_RETURN_IF_ERROR(function_op->SetDeviceName(resource.device.c_str()));

    AbstractTensorHandle* resource_handle = nullptr;
    int num_retvals = 1;
    TF_RETURN_IF_ERROR(function_op->Execute(
        absl::MakeSpan(&resource_handle, num_retvals), &num_retvals));
    AbstractTensorHandlePtr owned_resource_handle(resource_handle);
    if (!tensorflow::isa<ImmediateExecutionTensorHandle>(
            owned_resource_handle.get())) {
      return errors::Internal("Unexpected tensor handle kind.");
    }
    ImmediateTensorHandlePtr result(
        reinterpret_cast<ImmediateExecutionTensorHandle*>(
            owned_resource_handle.release()));
    resource.resource_handle = std::move(result);
  }
  return Status();
}

Status BuildResources(ImmediateExecutionContext* ctx,
                      const SavedObjectGraph& obj_graph,
                      PartiallyRevivedObjects* objects,
                      RevivedObjects* revived) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_10(mht_10_v, 631, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "BuildResources");

  for (auto& id_and_resource : objects->restored_resources) {
    int node_id = id_and_resource.first;
    RestoredResourceRevivalState& resource_revival_state =
        id_and_resource.second;

    TFConcreteFunction* create_resource = nullptr;

    // Check all the functions associated with the resource have already been
    // initialized in `revived`
    if (resource_revival_state.create_resource != nullptr) {
      create_resource = revived->concrete_functions.Find(
          resource_revival_state.create_resource->node_id);
      if (create_resource == nullptr) {
        return errors::FailedPrecondition(
            "'create_resource' function with node id ",
            resource_revival_state.create_resource->node_id, " not found");
      }
    }

    TFConcreteFunction* initialize = nullptr;
    if (resource_revival_state.initialize != nullptr) {
      initialize = revived->concrete_functions.Find(
          resource_revival_state.initialize->node_id);
      if (initialize == nullptr) {
        return errors::FailedPrecondition(
            "'initialize' function with node id ",
            resource_revival_state.initialize->node_id, " not found");
      }
    }

    TFConcreteFunction* destroy_resource = nullptr;
    if (resource_revival_state.destroy_resource != nullptr) {
      destroy_resource = revived->concrete_functions.Find(
          resource_revival_state.destroy_resource->node_id);
      if (destroy_resource == nullptr) {
        return errors::FailedPrecondition(
            "'destroy_resource' function with node id ",
            resource_revival_state.destroy_resource->node_id, " not found");
      }
    }

    if (resource_revival_state.resource_handle == nullptr) {
      return errors::FailedPrecondition("Resource at node id ", node_id,
                                        " does not have a resource handle.");
    }

    revived->restored_resources.emplace(
        node_id, RestoredResource(
                     /*device=*/resource_revival_state.device,
                     /*create_resource=*/create_resource,
                     /*initialize=*/initialize,
                     /*destroy_resource=*/destroy_resource,
                     /*resource_handle=*/
                     std::move(resource_revival_state.resource_handle)));
  }
  return Status();
}

}  // namespace

Status PartiallyRevivedObjects::Build(ImmediateExecutionContext* ctx,
                                      const SavedObjectGraph& obj_graph,
                                      RevivedObjects* revived) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSrevived_typesPSpartially_revived_objectsDTcc mht_11(mht_11_v, 697, "", "./tensorflow/c/experimental/saved_model/core/revived_types/partially_revived_objects.cc", "PartiallyRevivedObjects::Build");

  // Step 1: We would like to initialize all functions; this requires setting up
  // their captured tensorhandles, which may come from variables, assets,
  // constants, or resources. The first three are trivial; However,
  // tensorhandles that correspond to resources must be created by invoking
  // their "create_resource" function.
  // https://github.com/tensorflow/tensorflow/blob/f19c6efb4a8ba60e2492eedc98ef5375abb39dc7/tensorflow/python/saved_model/load.py#L240
  // https://github.com/tensorflow/tensorflow/blob/f19c6efb4a8ba60e2492eedc98ef5375abb39dc7/tensorflow/python/training/tracking/tracking.py#L233
  // For now, we assert that all create_resource functions must have no
  // captures. This aligns with the current behavior in python.
  // https://github.com/tensorflow/tensorflow/blob/50eac986bf7a0ad12594e080f083181f277e0b49/tensorflow/python/saved_model/load.py#L152-L155
  // TODO(bmzhao): We should do a topological sort instead.

  // 1a. Make sure all CreateResource functions have no captures.
  TF_RETURN_IF_ERROR(AssertAllCreateResourceFunctionsHaveNoCaptures(*this));

  // 1b. Initialize all CreateResource functions, storing them in `revived`
  TF_RETURN_IF_ERROR(
      InitializeCreateResourceFunctions(ctx, obj_graph, *this, revived));

  // 1c. Invoke all "CreateResource" functions and store their ResourceHandles
  // https://github.com/tensorflow/tensorflow/blob/3b6b41b68a95dc70c26dc816b29d359bfb88c116/tensorflow/python/training/tracking/tracking.py#L241-L247
  // in *this->resources.
  // TODO(bmzhao): Maybe store them separately, not in *this?
  TF_RETURN_IF_ERROR(CreateAllResourceHandles(ctx, obj_graph, this, revived));

  // 2. Initialize all the rest of the functions
  TF_RETURN_IF_ERROR(InitializeAllFunctions(ctx, obj_graph, *this, revived));

  // 3a. Move over all non-function, non-resource objects
  revived->variables = std::move(variables);
  revived->assets = std::move(assets);
  revived->constants = std::move(constants);
  revived->signatures_map = std::move(signatures_map);

  // 3b. Move over resources.
  TF_RETURN_IF_ERROR(BuildResources(ctx, obj_graph, this, revived));

  return Status();
}

}  // namespace tensorflow
