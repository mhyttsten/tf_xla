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
class MHTracer_DTPStensorflowPScorePSirPSimportexportPStestsPSroundtripPSroundtripDTcc {
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
   MHTracer_DTPStensorflowPScorePSirPSimportexportPStestsPSroundtripPSroundtripDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSirPSimportexportPStestsPSroundtripPSroundtripDTcc() {
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

#include "tensorflow/core/ir/importexport/tests/roundtrip/roundtrip.h"

#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/ir/importexport/export.h"
#include "tensorflow/core/ir/importexport/import.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"

using mlir::MLIRContext;

namespace tensorflow {

// Applies various normalization to a NodeDef to make it possible to perform
// textual comparison (for example splat constant are detected, NaN are removed,
// control input are alphabetically sorted, etc).
void NormalizeNode(NodeDef* node, bool add_fulltype) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPStestsPSroundtripPSroundtripDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/ir/importexport/tests/roundtrip/roundtrip.cc", "NormalizeNode");

  for (auto& named_attr : (*node->mutable_attr())) {
    AttrValue& attr_val = named_attr.second;
    if (attr_val.has_tensor()) {
      auto* tensor = attr_val.mutable_tensor();
      switch (tensor->dtype()) {
        // There is no compression or canonicalization for DT_STRING, let's
        // just strip it entirely for now so it is ignored from the comparison.
        case DT_STRING: {
          const TensorShape shape(tensor->tensor_shape());
          if (!tensor->tensor_content().empty()) {
            tensor->mutable_tensor_content()->clear();
          } else {
            tensor->mutable_string_val()->Clear();
          }
          break;
        }
        case DT_FLOAT:
          tensor::CompressTensorProtoInPlace(1, 1.0, tensor);
          for (float& val : *tensor->mutable_float_val())
            if (std::isnan(val)) val = -42.;
          break;
        case DT_DOUBLE:
          tensor::CompressTensorProtoInPlace(1, 1.0, tensor);
          for (double& val : *tensor->mutable_double_val())
            if (std::isnan(val)) val = -42.;
          break;
        case DT_COMPLEX64:
          tensor::CompressTensorProtoInPlace(1, 1.0, tensor);
          for (float& val : *tensor->mutable_scomplex_val())
            if (std::isnan(val)) val = -42.;
          break;
        case DT_COMPLEX128:
          tensor::CompressTensorProtoInPlace(1, 1.0, tensor);
          for (double& val : *tensor->mutable_dcomplex_val())
            if (std::isnan(val)) val = -42.;
          break;
        case DT_VARIANT: {
          Tensor t;
          if (t.FromProto(*tensor)) t.AsProtoField(tensor);
          break;
        }
        default:
          tensor::CompressTensorProtoInPlace(1, 1.0, tensor);
      }
    }
  }
  // Sort control inputs alphabetically.
  for (auto it = node->mutable_input()->begin(),
            end = node->mutable_input()->end();
       it != end; ++it) {
    if (it->empty() || it->front() != '^') continue;
    std::sort(it, end);
  }

  const OpDef* op_def = nullptr;
  (void)tensorflow::OpRegistry::Global()->LookUpOpDef(node->op(), &op_def);

  // Following logic in Graph::AddNode to avoid false positives due to
  // type refinement.
  if (add_fulltype) {
    if (node->has_experimental_type()) {
      VLOG(3) << "node has type set, skipping type constructor "
              << node->name();
    } else {
      const OpRegistrationData* op_reg_data;
      (void)tensorflow::OpRegistry::Global()->LookUp(node->op(), &op_reg_data);
      if (op_reg_data && op_reg_data->type_ctor != nullptr) {
        VLOG(3) << "found type constructor for " << node->name();
        (void)full_type::SpecializeType(AttrSlice(*node), op_reg_data->op_def,
                                        *(node->mutable_experimental_type()));
      } else {
        VLOG(3) << "no type constructor for " << node->name();
      }
    }
  }

  if (op_def) StripDefaultsFromNodeDef(*op_def, node);
}

void NormalizeTensorData(GraphDef& graphdef, bool add_fulltype) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPStestsPSroundtripPSroundtripDTcc mht_1(mht_1_v, 293, "", "./tensorflow/core/ir/importexport/tests/roundtrip/roundtrip.cc", "NormalizeTensorData");

  FunctionDefLibrary* library = graphdef.mutable_library();
  llvm::sort(*library->mutable_function(),
             [](FunctionDef& lhs, FunctionDef& rhs) {
               return lhs.signature().name() < rhs.signature().name();
             });

  for (int i = 0; i < graphdef.node_size(); ++i)
    NormalizeNode(graphdef.mutable_node(i), add_fulltype);
  llvm::sort(*graphdef.mutable_node(),
             [](const NodeDef& lhs, const NodeDef& rhs) {
               return lhs.name() < rhs.name();
             });
  for (int func_id = 0; func_id < library->function_size(); ++func_id) {
    FunctionDef* func = library->mutable_function(func_id);
    llvm::sort(*func->mutable_node_def(), [](NodeDef& lhs, NodeDef& rhs) {
      return lhs.name() < rhs.name();
    });
    for (int node_id = 0; node_id < func->node_def_size(); ++node_id) {
      NodeDef* node = func->mutable_node_def(node_id);
      NormalizeNode(node, add_fulltype);
    }
    for (const auto& it : *func->mutable_ret()) {
      func->mutable_ret()->at(it.first) = it.second;
      // Eliminate empty arg_attr entries.
      llvm::SmallVector<int> to_erase;
      for (auto& arg_attr : *func->mutable_arg_attr()) {
        if (arg_attr.second.attr().empty()) {
          to_erase.push_back(arg_attr.first);
        }
      }
      for (int idx : to_erase) func->mutable_arg_attr()->erase(idx);
      for (int i = 0; i < func->node_def_size(); ++i)
        NormalizeNode(func->mutable_node_def(i), add_fulltype);
    }
  }
}

Status TestRoundTrip(GraphDef& graphdef) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPStestsPSroundtripPSroundtripDTcc mht_2(mht_2_v, 334, "", "./tensorflow/core/ir/importexport/tests/roundtrip/roundtrip.cc", "TestRoundTrip");

  MLIRContext context;
  GraphDebugInfo debug_info;
  auto errorOrModule =
      mlir::tfg::ImportGraphDefToMlir(&context, debug_info, graphdef);
  if (!errorOrModule.ok()) {
    LOG(ERROR) << errorOrModule.status();
    llvm::errs()
        << "\n\n=========\n=========\n=========\n=========\n=========\n"
        << graphdef.DebugString()
        << "=========\n=========\n=========\n=========\n";
    return errorOrModule.status();
  }
  GraphDef new_graph;
  auto module = errorOrModule.ValueOrDie().get();
  Status status = tensorflow::ExportMlirToGraphdef(module, &new_graph);
  if (!status.ok()) {
    LOG(ERROR) << "Error exporting MLIR module to GraphDef: " << status;
    return status;
  }
  GraphDef original_graph;
  {
    GraphConstructorOptions options;
    options.allow_internal_ops = true;
    options.add_default_attributes = true;
    Graph graph(OpRegistry::Global());
    GraphDef preprocessed_graphdef(graphdef);
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
        options, std::move(preprocessed_graphdef), &graph));
    graph.ToGraphDef(&original_graph);
  }
  NormalizeTensorData(new_graph, /*add_fulltype=*/false);
  NormalizeTensorData(original_graph, /*add_fulltype=*/true);

  tensorflow::protobuf::util::MessageDifferencer differencer;
  if (!differencer.Equivalent(original_graph, new_graph)) {
    LOG(ERROR) << "GraphDef didn't Roundtrip:";
    llvm::errs()
        << "\n=========\n\n"
        << module
        << "\n\n=========\n=========\n=========\n=========\n=========\n"
        << graphdef.DebugString()
        << "=========\n=========\n=========\n=========\n";
    return errors::InvalidArgument("GraphDef didn't roundtrip");
  }
  return {};
}

}  // namespace tensorflow
