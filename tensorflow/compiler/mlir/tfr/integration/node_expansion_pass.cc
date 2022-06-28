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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPSnode_expansion_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPSnode_expansion_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPSnode_expansion_passDTcc() {
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
#include "tensorflow/compiler/mlir/tfr/integration/node_expansion_pass.h"

#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {

auto* tf_core_op_expansion_node_counter =
    monitoring::Counter<0>::New("/tensorflow/core/op_expansion/node_counter",
                                "The number of nodes being op expanded.");
}  // namespace

namespace tfr {

Status CompositeOpExpansion::Run(EagerOperation* orig_op,
                                 std::unique_ptr<EagerOperation>* out_op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPSnode_expansion_passDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/mlir/tfr/integration/node_expansion_pass.cc", "CompositeOpExpansion::Run");

  if (!IsEnabled()) return Status::OK();
  // This can be the default cpu device.
  if (orig_op->Device() != kVariantDeviceNull) return Status::OK();
  if (orig_op->is_function()) return Status::OK();

  // TODO(fengliuai): We need a better condition to skip the rewrite. Currently,
  // The rewrite is enabled for all the tf ops and it is a no-op if the tf op
  // isn't a composite op. The following ops are explicitly skipped here because
  // their "no-op" expansion is known to cause problems in some cases.
  static const char* kOpsToSkip[] = {
      "IdentityOp",
      "NoOp",              // b/174596063
      "OptionalHasValue",  // b/173136483
      "OptionalGetValue",  // b/173136483
      "VarHandleOp",       // b/176819198
  };
  for (const char* skip : kOpsToSkip) {
    if (absl::StartsWith(orig_op->op_name(), skip)) return Status::OK();
  }

  tf_core_op_expansion_node_counter->GetCell()->IncrementBy(1);

  LOG_FIRST_N(INFO, 1) << "Run Node Expansion Passes";

  // Get the FunctionDef and insert that into the context
  const NodeDef& ndef = orig_op->MutableAttrs()->BuildNodeDef();
  auto& ctx = orig_op->EagerContext();
  Fprint128 cache_key =
      orig_op->MutableAttrs()->CacheKey(orig_op->DeviceName());
  // Include soft placement policy in cache key since the placement strategy
  // can change and thus affect which kernel is picked.
  auto x = FingerprintCat64(cache_key.high64, cache_key.low64);
  std::string fname =
      absl::StrCat("_expanded_", ndef.name(), "_", std::to_string(x));
  if (!ctx.FindFunctionByName(fname)) {
    TF_ASSIGN_OR_RETURN(auto func, ExpandNode(ndef, fname));
    TF_RETURN_IF_ERROR(ctx.AddFunctionDef(func));
  }

  // Rewrite the out_op to be the call op. This essentially a deep copy of the
  // orig_op, except the op name.
  auto* new_op = new EagerOperation(&ctx);
  TF_RETURN_IF_ERROR(
      new_op->Reset(fname.c_str(), orig_op->DeviceName().c_str()));
  for (auto input : orig_op->GetInputs()) {
    TF_RETURN_IF_ERROR(new_op->AddInput(input));
  }
  new_op->MutableAttrs()->CopyAttributes(orig_op->Attrs());
  out_op->reset(new_op);

  LOG_FIRST_N(INFO, 1)
      << "Finish Node Expansion Passes. Rewrite the op to call function: "
      << fname;

  return Status::OK();
}

REGISTER_REWRITE(EagerOpRewriteRegistry::POST_PLACEMENT, 20000,
                 CompositeOpExpansion);

}  // namespace tfr
}  // namespace tensorflow
