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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSeval_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSeval_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSeval_utilDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/eval_util.h"

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

using gtl::MakeCleanup;

#define RETURN_FAILURE_IF_ERROR(expr) \
  if (!IsOk(expr)) {                  \
    return mlir::failure();           \
  }

static bool IsOk(const TF_Status* s) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSeval_utilDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/mlir/tensorflow/utils/eval_util.cc", "IsOk");

  if (TF_GetCode(s) == TF_OK) return true;
  VLOG(2) << TF_Message(s);
  return false;
}

static bool IsOk(const Status& s) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSeval_utilDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/mlir/tensorflow/utils/eval_util.cc", "IsOk");

  if (s.ok()) return true;
  VLOG(2) << s.error_message();
  return false;
}

mlir::LogicalResult EvaluateOperation(
    mlir::Operation* inst, llvm::ArrayRef<mlir::ElementsAttr> operands,
    TFE_Context* context, llvm::SmallVectorImpl<mlir::Attribute>* results) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSeval_utilDTcc mht_2(mht_2_v, 235, "", "./tensorflow/compiler/mlir/tensorflow/utils/eval_util.cc", "EvaluateOperation");

  if (!context) {
    VLOG(1) << "Can't evaluate with null context.";
    return mlir::failure();
  }
  // If any operand is nullptr returns true for a failure.
  // TODO(b/120678030): remove this constraint if we find operators can be
  // evaluated with some unknown operands.
  if (std::any_of(operands.begin(), operands.end(),
                  [](mlir::Attribute operand) { return !operand; })) {
    VLOG(1) << "Can't evaluate since not all operands are constant.";
    return mlir::failure();
  }

  TF_Status* status = TF_NewStatus();
  auto clean_status = MakeCleanup([status] { TF_DeleteStatus(status); });

  // Builds TF operation and sets all the attributes.
  std::string node_name = "unnamed";
  if (auto attr = inst->getAttrOfType<mlir::StringAttr>("name")) {
    node_name = std::string(attr.getValue());
  }
  auto node_def_or = ConvertTFDialectOpToNodeDef(
      inst, node_name.c_str(), /*ignore_unregistered_attrs=*/true);
  RETURN_FAILURE_IF_ERROR(node_def_or.status());
  const auto& node_def = node_def_or.ValueOrDie();

  TFE_Op* op = TFE_NewOp(context, node_def->op().c_str(), status);
  RETURN_FAILURE_IF_ERROR(status);
  auto clean_op = MakeCleanup([op] { TFE_DeleteOp(op); });

  // Explicitly set device to Host CPU instead of the device present in device
  // attribute of the MLIR op. The assigned device might be remote, not
  // available during compilation or compilation only device for on demand
  // execution which may create a recursion if used for constant folding.
  constexpr char kHostCpu[] = "/job:localhost/replica:0/task:0/CPU:0";
  TFE_OpSetDevice(op, kHostCpu, status);
  RETURN_FAILURE_IF_ERROR(status);
  for (const auto& attr : node_def->attr()) {
    SetOpAttrValueScalar(context, op, attr.second, attr.first.c_str(), status);
    RETURN_FAILURE_IF_ERROR(status);
  }

  VLOG(1) << "Start to evaluate node: " << node_def->DebugString();

  // Adds inputs to the TF operation.
  for (const auto operand : operands) {
    Tensor tensor;
    RETURN_FAILURE_IF_ERROR(ConvertToTensor(operand, &tensor));
    TF_Tensor* tf_tensor = TF_TensorFromTensor(tensor, &status->status);
    RETURN_FAILURE_IF_ERROR(status);
    auto clean_tensor =
        MakeCleanup([tf_tensor] { TF_DeleteTensor(tf_tensor); });
    TFE_TensorHandle* input_handle = TFE_NewTensorHandle(tf_tensor, status);
    RETURN_FAILURE_IF_ERROR(status);
    auto clean_input_handle =
        MakeCleanup([input_handle] { TFE_DeleteTensorHandle(input_handle); });
    TFE_OpAddInput(op, input_handle, status);
    RETURN_FAILURE_IF_ERROR(status);
  }

  // Executes the TF operation.
  int num_outputs = inst->getNumResults();
  absl::InlinedVector<TFE_TensorHandle*, 2> outputs(num_outputs);
  TFE_Execute(op, outputs.data(), &num_outputs, status);
  RETURN_FAILURE_IF_ERROR(status);
  auto clean_outputs = MakeCleanup([&outputs] {
    for (TFE_TensorHandle* tensor_handle : outputs) {
      TFE_DeleteTensorHandle(tensor_handle);
    }
  });

  // Converts the outputs to MLIR attributes.
  mlir::Builder builder(inst->getContext());
  for (TFE_TensorHandle* tensor_handle : outputs) {
    TF_Tensor* tf_tensor = TFE_TensorHandleResolve(tensor_handle, status);
    RETURN_FAILURE_IF_ERROR(status);
    auto clean_tensor =
        MakeCleanup([tf_tensor] { TF_DeleteTensor(tf_tensor); });
    Tensor tensor;
    RETURN_FAILURE_IF_ERROR(TF_TensorToTensor(tf_tensor, &tensor));
    auto attr_or = ConvertTensor(tensor, &builder);
    RETURN_FAILURE_IF_ERROR(attr_or.status());
    results->push_back(attr_or.ValueOrDie());
  }

  VLOG(1) << "Evaluate node " << node_name << " successfully!";

  return mlir::success();
}

#undef RETURN_FAILURE_IF_ERROR
}  // namespace tensorflow
