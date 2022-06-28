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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_inplaceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_inplaceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_inplaceDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_inplace.h"

#include <cstring>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

static const char* kInplacePrefix = "inplace_update:\0";

class EmptyInplaceRewrite : public InlineRewrite {
 public:
  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("input: \"" + std::string(input.data(), input.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_inplaceDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/delegates/gpu/gl/compiler/fuse_inplace.cc", "Rewrite");

    if (input.compare(0, strlen(kInplacePrefix), kInplacePrefix) == 0) {
      num_rewrites_++;
      return RewriteStatus::SUCCESS;
    }
    return RewriteStatus::NOT_RECOGNIZED;
  }

  int num_rewrites() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_inplaceDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/delegates/gpu/gl/compiler/fuse_inplace.cc", "num_rewrites");
 return num_rewrites_; }

 private:
  int num_rewrites_ = 0;
};

// Takes a code as an input. Replaces 'value_0' in the code with a value that
// comes in a rewrite. For example:
//   code:    value_0 = max(value_0, 0);
//   rewrite: inplace_update:result_12 -> result_12 = max(result_12, 0);
//
class InplaceCodeRewrite : public InlineRewrite {
 public:
  explicit InplaceCodeRewrite(const std::string& code) : code_(code) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("code: \"" + code + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_inplaceDTcc mht_2(mht_2_v, 238, "", "./tensorflow/lite/delegates/gpu/gl/compiler/fuse_inplace.cc", "InplaceCodeRewrite");
}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("input: \"" + std::string(input.data(), input.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_inplaceDTcc mht_3(mht_3_v, 244, "", "./tensorflow/lite/delegates/gpu/gl/compiler/fuse_inplace.cc", "Rewrite");

    int len = strlen(kInplacePrefix);
    if (input.compare(0, len, kInplacePrefix) == 0) {
      auto variable_name = input.substr(len);
      absl::StrAppend(output,
                      absl::StrReplaceAll(code_, {{"value_0", variable_name}}));
      return RewriteStatus::SUCCESS;
    }
    return RewriteStatus::NOT_RECOGNIZED;
  }

 private:
  std::string code_;
};

}  // namespace

TransformResult RemoveUnusedInplaceUpdates::ApplyToNode(Node* node,
                                                        GraphFloat32* graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_inplaceDTcc mht_4(mht_4_v, 265, "", "./tensorflow/lite/delegates/gpu/gl/compiler/fuse_inplace.cc", "RemoveUnusedInplaceUpdates::ApplyToNode");

  auto& attr =
      absl::any_cast<CompiledNodeAttributes&>(node->operation.attributes);
  // Remove inplace block by rewriting to empty string.
  EmptyInplaceRewrite rewrite;
  TextPreprocessor preprocessor('$', true);
  preprocessor.AddRewrite(&rewrite);
  if (!preprocessor.Rewrite(attr.code.source_code, &attr.code.source_code)
           .ok()) {
    return {TransformStatus::INVALID, ""};
  }
  return {rewrite.num_rewrites() > 0 ? TransformStatus::APPLIED
                                     : TransformStatus::SKIPPED,
          ""};
}

TransformResult FuseInplaceUpdate::ApplyToNodesSequence(
    const std::vector<Node*>& sequence, GraphFloat32* graph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_inplaceDTcc mht_5(mht_5_v, 285, "", "./tensorflow/lite/delegates/gpu/gl/compiler/fuse_inplace.cc", "FuseInplaceUpdate::ApplyToNodesSequence");

  Node* node1 = sequence.front();
  Node* node2 = sequence.back();
  auto& attr1 =
      absl::any_cast<CompiledNodeAttributes&>(node1->operation.attributes);
  auto& attr2 =
      absl::any_cast<CompiledNodeAttributes&>(node2->operation.attributes);

  if (graph->FindInputs(node2->id).size() != 1 ||
      graph->FindOutputs(node2->id).size() != 1 ||
      attr2.code.output != IOStructure::AUTO ||
      attr2.code.input != IOStructure::AUTO ||
      (attr1.code.workload != attr2.code.workload &&
       uint3() != attr2.code.workload)) {
    return {TransformStatus::SKIPPED, ""};
  }

  // First count of replaces that would happen to check whether rewrite is
  // needed.
  {
    EmptyInplaceRewrite counting_rewrite;
    TextPreprocessor preprocessor('$', true);
    preprocessor.AddRewrite(&counting_rewrite);
    std::string temp;
    if (!preprocessor.Rewrite(attr1.code.source_code, &temp).ok()) {
      return {TransformStatus::INVALID, ""};
    }
    // no rewrites in the source code. skip it.
    if (counting_rewrite.num_rewrites() == 0) {
      return {TransformStatus::SKIPPED, ""};
    }
  }
  if (!MergeCode(&attr2, &attr1).ok()) {
    return {TransformStatus::INVALID, "Unable to merge two nodes"};
  }
  TextPreprocessor preprocessor('$', true);
  InplaceCodeRewrite rewrite(attr2.code.source_code);
  preprocessor.AddRewrite(&rewrite);
  if (!preprocessor.Rewrite(attr1.code.source_code, &attr1.code.source_code)
           .ok()) {
    return {TransformStatus::INVALID, ""};
  }
  node1->operation.type += "+" + node2->operation.type;

  if (!RemoveFollowingNode(graph, node2, node1).ok()) {
    return {TransformStatus::INVALID,
            "Unable to remove node " + std::to_string(node2->id)};
  }
  return {TransformStatus::APPLIED, ""};
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
