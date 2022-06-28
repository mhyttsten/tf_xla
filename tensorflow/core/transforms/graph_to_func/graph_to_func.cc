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
class MHTracer_DTPStensorflowPScorePStransformsPSgraph_to_funcPSgraph_to_funcDTcc {
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
   MHTracer_DTPStensorflowPScorePStransformsPSgraph_to_funcPSgraph_to_funcDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStransformsPSgraph_to_funcPSgraph_to_funcDTcc() {
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

#include "tensorflow/core/transforms/graph_to_func/graph_to_func.h"

#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"
#include "tensorflow/core/platform/errors.h"

using tensorflow::Status;
using tensorflow::errors::InvalidArgument;

namespace mlir {
namespace tfg {

// TODO(jpienaar): Move to helper header/this shouldn't be needed once we
// upgrade to C++17.
static inline absl::string_view ToStringView(llvm::StringRef ref) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStransformsPSgraph_to_funcPSgraph_to_funcDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/transforms/graph_to_func/graph_to_func.cc", "ToStringView");

  return {ref.data(), ref.size()};
}

static std::string OpResultToSlotName(OpResult value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStransformsPSgraph_to_funcPSgraph_to_funcDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/transforms/graph_to_func/graph_to_func.cc", "OpResultToSlotName");

  return (TFOp(*value.getDefiningOp()).name() + ":" +
          Twine(value.cast<OpResult>().getResultNumber()))
      .str();
}

tensorflow::Status GraphToFunc(GraphOp graph, ArrayRef<Value> feeds,
                               ArrayRef<Value> fetches,
                               ArrayRef<Value> control_rets, StringRef name) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStransformsPSgraph_to_funcPSgraph_to_funcDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/transforms/graph_to_func/graph_to_func.cc", "GraphToFunc");

  OpBuilder builder(graph);
  ControlType control_ty = ControlType::get(graph.getContext());
  llvm::SmallVector<Type> arg_types;
  llvm::SmallVector<Type> ret_types;
  for (Value feed : feeds) {
    arg_types.push_back(feed.getType());
    arg_types.push_back(control_ty);
  }
  for (Value fetch : fetches) ret_types.push_back(fetch.getType());
  FunctionType func_type = builder.getFunctionType(arg_types, ret_types);
  auto loc = graph.getLoc();
  auto func_op = builder.create<GraphFuncOp>(loc, name, func_type,
                                             /*generic=*/false);
  func_op.getRegion().takeBody(graph.getRegion());
  Block *body = func_op.getBody();
  llvm::SmallVector<Attribute> args_rets_attrs;
  for (Value feed : feeds) {
    feed.replaceAllUsesWith(body->addArgument(feed.getType(), loc));
    body->addArgument(control_ty, loc);
    llvm::SmallVector<NamedAttribute> arg_attrs;
    std::string slot = OpResultToSlotName(feed.cast<OpResult>());
    arg_attrs.push_back(
        builder.getNamedAttr("tfg.name", builder.getStringAttr(slot)));
    args_rets_attrs.push_back(builder.getDictionaryAttr(arg_attrs));
    args_rets_attrs.push_back(Attribute{});
  }
  func_op.setAllArgAttrs(args_rets_attrs);

  args_rets_attrs.clear();
  for (Value fetch : fetches) {
    llvm::SmallVector<NamedAttribute> arg_attrs;
    std::string slot = OpResultToSlotName(fetch.cast<OpResult>());
    arg_attrs.push_back(
        builder.getNamedAttr("tfg.name", builder.getStringAttr(slot)));
    args_rets_attrs.push_back(builder.getDictionaryAttr(arg_attrs));
  }
  func_op.setAllResultAttrs(args_rets_attrs);

  OpBuilder body_builder = OpBuilder::atBlockEnd(func_op.getBody());
  body_builder.create<ReturnOp>(loc, fetches, control_rets);
  graph.erase();
  return Status::OK();
}

Status GraphToFunc(GraphOp graph, ArrayRef<std::string> feeds_names,
                   ArrayRef<std::string> fetches_names,
                   ArrayRef<std::string> control_rets_names, StringRef name) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStransformsPSgraph_to_funcPSgraph_to_funcDTcc mht_3(mht_3_v, 276, "", "./tensorflow/core/transforms/graph_to_func/graph_to_func.cc", "GraphToFunc");

  DenseMap<StringRef, int> feeds_to_position;
  feeds_to_position.reserve(feeds_names.size());
  for (const auto &indexed_name : llvm::enumerate(feeds_names)) {
    const std::string &name = indexed_name.value();
    if (!feeds_to_position.insert({StringRef(name), indexed_name.index()})
             .second)
      return InvalidArgument("GraphToFunc: got duplicated feed name: ", name);
  }
  DenseMap<StringRef, int> fetches_to_position;
  fetches_to_position.reserve(fetches_names.size());
  for (const auto &indexed_name : llvm::enumerate(fetches_names)) {
    const std::string &name = indexed_name.value();
    if (feeds_to_position.count(name))
      return InvalidArgument("GraphToFunc: name is both a feed and a fetch: '",
                             name, "'");
    if (!fetches_to_position.insert({StringRef(name), indexed_name.index()})
             .second)
      return InvalidArgument("GraphToFunc: got duplicated fetch name: '", name,
                             "'");
  }
  DenseMap<StringRef, int> control_rets_to_position;
  control_rets_to_position.reserve(control_rets_names.size());
  for (const auto &indexed_name : llvm::enumerate(control_rets_names)) {
    if (!control_rets_to_position
             .insert({StringRef(indexed_name.value()), indexed_name.index()})
             .second)
      return InvalidArgument("GraphToFunc: got duplicated control_ret name: '",
                             indexed_name.value(), "'");
  }

  SmallVector<Value> feeds(feeds_names.size());
  SmallVector<Value> fetches(fetches_names.size());
  SmallVector<Value> control_rets(control_rets_names.size());
  for (Operation &op : *graph.getBody()) {
    TFOp tf_op(op);
    StringRef node_name = tf_op.name();
    // A slot is the node name + the output index separated by a colon.
    std::string slot;
    for (Value result : op.getResults()) {
      slot = OpResultToSlotName(result.cast<OpResult>());
      auto feed_pos = feeds_to_position.find(slot);
      if (feed_pos != feeds_to_position.end()) {
        feeds[feed_pos->second] = result;
        continue;
      }
      auto fetch_pos = fetches_to_position.find(slot);
      if (fetch_pos != fetches_to_position.end()) {
        fetches[fetch_pos->second] = result;
        continue;
      }
      auto control_ret_pos = control_rets_to_position.find(node_name);
      if (control_ret_pos != control_rets_to_position.end()) {
        control_rets[control_ret_pos->second] = tf_op.controlRet();
        continue;
      }
    }
  }
  for (const auto &feed_info : feeds_to_position) {
    if (!feeds[feed_info.second])
      return InvalidArgument("Can't find feed: '",
                             ToStringView(feed_info.first), "'");
  }
  for (const auto &fetch_info : fetches_to_position) {
    if (!fetches[fetch_info.second])
      return InvalidArgument("Can't find fetch: '",
                             ToStringView(fetch_info.first), "'");
  }
  for (const auto &control_ret_info : control_rets_to_position) {
    if (!control_rets[control_ret_info.second])
      return InvalidArgument("Can't find control rets: '",
                             ToStringView(control_ret_info.first), "'");
  }

  return GraphToFunc(graph, feeds, fetches, control_rets, name);
}

}  // namespace tfg
}  // namespace mlir
