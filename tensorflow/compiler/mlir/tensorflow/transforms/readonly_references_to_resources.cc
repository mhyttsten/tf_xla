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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreadonly_references_to_resourcesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreadonly_references_to_resourcesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreadonly_references_to_resourcesDTcc() {
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TF {
namespace {

// Location attribute.
constexpr StringRef kClassAttr = "_class";
constexpr StringRef kSharedNameAttr = "shared_name";
constexpr StringRef kLocationPrefix = "loc:@";

// A pass that converts readonly reference variables to the corresponding
// resource variables.
//
// It converts (VariableV2 -> Identity) to (VarHandle -> ReadVariable).
//
// For the background, this pass is a part of hoisting VariableV2 ops by
// re-using the pipeline for hoisting (VarHandle -> ReadVariable) cases, which
//  can be done by the following passes:
//  - Capturing resource values into global tensors (importing saved model).
//  - Promoting VarHandle ops to function input/outputs.
//  - Freezing global tensor pass.
//
// This path assumes that all the VariableV2 ops is read-only via verifying the
// heuristic method that assumes that all the users of them is Identity op,
// fed directly.
class ConvertReadonlyReferenceVariablesToResourceVariablesPass
    : public ConvertReadonlyReferenceVariablesToResourceVariablesPassBase<
          ConvertReadonlyReferenceVariablesToResourceVariablesPass> {
  void runOnOperation() override;
};

// Parse node name from "_class" or "shared_name" attributes.
StringRef GetNodeNameFromClassAttrOrSharedNameAttr(Operation *op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreadonly_references_to_resourcesDTcc mht_0(mht_0_v, 234, "", "./tensorflow/compiler/mlir/tensorflow/transforms/readonly_references_to_resources.cc", "GetNodeNameFromClassAttrOrSharedNameAttr");

  // Parse node name from the `shared_name` attribute first. The variable v2 op
  // relies on the share name to look up from the TensorFlow's resource manager.
  StringAttr shared_name_attr = op->getAttrOfType<StringAttr>(kSharedNameAttr);
  if (shared_name_attr) {
    auto shared_name = StringRef(shared_name_attr.getValue());
    if (!shared_name.empty()) {
      return shared_name;
    }
  }
  // Attempt to parse "_class" attribute if there is no "shared_name"
  // attribute.
  ArrayAttr classes_attr = op->getAttrOfType<ArrayAttr>(kClassAttr);
  if (!classes_attr) {
    // Attempt to parse "_class" from the IdentityOp that follows VariableV2.
    // For read-only reference variables, IdentityOp should be the only user of
    // VariableV2.
    auto identity_op = op->getUsers().begin();
    classes_attr = identity_op->getAttrOfType<ArrayAttr>(kClassAttr);
    if (!classes_attr) {
      op->emitOpError() << "has no '_class' and 'shared_name' attributes";
      return StringRef();
    }
  }

  StringRef result;
  for (Attribute class_attr : classes_attr) {
    StringRef node_name = class_attr.cast<StringAttr>().getValue();
    if (!node_name.startswith(kLocationPrefix)) {
      continue;
    }
    if (!result.empty()) {
      // Invalid case since there are multiple loc:@ attributes.
      op->emitOpError()
          << "expects only one named location in '_class' attribute, but got "
          << classes_attr;
      return StringRef();
    }
    result = node_name.drop_front(kLocationPrefix.size());
  }
  if (result.empty()) {
    op->emitOpError() << "expects variable name in '_class' attribute, but got "
                      << classes_attr;
  }
  return result;
}

void ConvertReadonlyReferenceVariablesToResourceVariablesPass::
    runOnOperation() {
  FuncOp func = getOperation();

  OpBuilder builder(func.getContext());
  SmallVector<VariableV2Op, 4> variable_v2s_to_replace;

  // Checks all the VariableV2 ops is read-only via verifying the heuristic
  // method that assumes that all the users of them is Identity op, feeded
  // directly.
  auto read_only_vars_fn = [&variable_v2s_to_replace](
                               VariableV2Op variable_v2_op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreadonly_references_to_resourcesDTcc mht_1(mht_1_v, 295, "", "./tensorflow/compiler/mlir/tensorflow/transforms/readonly_references_to_resources.cc", "lambda");

    if (variable_v2_op.getResult().use_empty()) {
      // Erase the op when there is no user.
      variable_v2_op.erase();
      return mlir::WalkResult::advance();
    }
    if (!all_of(variable_v2_op.getResult().getUsers(), [&variable_v2_op](
                                                           Operation *user) {
          if (!isa<IdentityOp>(user)) {
            variable_v2_op.emitOpError()
                << "expects all users to be 'tf.Identity', but got user "
                << user->getName();
            return false;
          }
          return true;
        })) {
      return mlir::WalkResult::interrupt();
    }
    variable_v2s_to_replace.push_back(variable_v2_op);
    return mlir::WalkResult::advance();
  };

  WalkResult walk_res = func.walk(read_only_vars_fn);
  if (walk_res.wasInterrupted()) return signalPassFailure();

  for (VariableV2Op variable_v2_op : variable_v2s_to_replace) {
    builder.setInsertionPoint(variable_v2_op);
    ShapedType shaped_type =
        variable_v2_op.getResult().getType().cast<ShapedType>();
    TensorType tensor_type = DropRefType(shaped_type).cast<TensorType>();
    StringAttr device_attr =
        variable_v2_op->getAttrOfType<StringAttr>("device");
    if (!device_attr) device_attr = builder.getStringAttr("");
    StringRef variable_name =
        GetNodeNameFromClassAttrOrSharedNameAttr(variable_v2_op);
    if (variable_name.empty()) {
      return signalPassFailure();
    }
    VarHandleOp var_handle_op = builder.create<VarHandleOp>(
        variable_v2_op.getLoc(),
        ArrayRef<Type>{RankedTensorType::get(
            {}, TF::ResourceType::get(ArrayRef<TensorType>{tensor_type},
                                      builder.getContext()))},
        ArrayRef<Value>{},
        ArrayRef<NamedAttribute>{
            builder.getNamedAttr("device", device_attr),
            builder.getNamedAttr("container", variable_v2_op.containerAttr()),
            builder.getNamedAttr("shared_name",
                                 builder.getStringAttr(variable_name))});
    for (Operation *user :
         make_early_inc_range(variable_v2_op.getResult().getUsers())) {
      builder.setInsertionPoint(user);
      ReadVariableOp read_variable_op = builder.create<ReadVariableOp>(
          user->getLoc(), ArrayRef<Type>{tensor_type},
          ArrayRef<Value>{var_handle_op});
      user->getResult(0).replaceAllUsesWith(read_variable_op.getResult());
      user->erase();
    }
    variable_v2_op.erase();
  }
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateConvertReadonlyReferenceVariablesToResourceVariablesPass() {
  return std::make_unique<
      ConvertReadonlyReferenceVariablesToResourceVariablesPass>();
}

}  // namespace TF

}  // namespace mlir
