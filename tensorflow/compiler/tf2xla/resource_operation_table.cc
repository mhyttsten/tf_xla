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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_operation_tableDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_operation_tableDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_operation_tableDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/resource_operation_table.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"

namespace tensorflow {
/*static*/ absl::string_view XlaResourceOpInfo::XlaResourceOpKindToString(
    XlaResourceOpKind op_kind) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_operation_tableDTcc mht_0(mht_0_v, 192, "", "./tensorflow/compiler/tf2xla/resource_operation_table.cc", "XlaResourceOpInfo::XlaResourceOpKindToString");

  switch (op_kind) {
    case XlaResourceOpKind::kRead:
      return "Read";
    case XlaResourceOpKind::kWrite:
      return "Write";
    case XlaResourceOpKind::kReadWrite:
      return "Modify";
  }
}

static absl::flat_hash_map<absl::string_view, XlaResourceOpInfo>*
CreateResourceOpInfoMap() {
  auto* result = new absl::flat_hash_map<absl::string_view, XlaResourceOpInfo>;

  auto add = [&](absl::string_view op, XlaResourceOpKind op_kind,
                 XlaResourceKind resource_kind) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op: \"" + std::string(op.data(), op.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_operation_tableDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/tf2xla/resource_operation_table.cc", "lambda");

    auto insert_result =
        result->insert({op, XlaResourceOpInfo(op_kind, resource_kind)});
    CHECK(insert_result.second);
  };

  auto kRead = XlaResourceOpKind::kRead;
  auto kWrite = XlaResourceOpKind::kWrite;
  auto kReadWrite = XlaResourceOpKind::kReadWrite;

  auto kVariable = XlaResourceKind::kVariable;
  auto kStack = XlaResourceKind::kStack;
  auto kTensorArray = XlaResourceKind::kTensorArray;

  // clang-format off
  add("AssignAddVariableOp"                  , kReadWrite, kVariable);
  add("AssignSubVariableOp"                  , kReadWrite, kVariable);
  add("AssignVariableOp"                     , kWrite,     kVariable);
  add("AssignVariableXlaConcatND"            , kWrite,     kVariable);
  add("CollectiveReduceV2"                   , kRead,      kVariable);
  add("ReadVariableOp"                       , kRead,      kVariable);
  add("ReadVariableXlaSplitND"               , kRead,      kVariable);
  add("ResourceApplyAdaMax"                  , kReadWrite, kVariable);
  add("ResourceApplyAdadelta"                , kReadWrite, kVariable);
  add("ResourceApplyAdagrad"                 , kReadWrite, kVariable);
  add("ResourceApplyAdagradV2"               , kReadWrite, kVariable),
  add("ResourceApplyAdagradDA"               , kReadWrite, kVariable);
  add("ResourceApplyAdam"                    , kReadWrite, kVariable);
  add("ResourceApplyAddSign"                 , kReadWrite, kVariable);
  add("ResourceApplyCenteredRMSProp"         , kReadWrite, kVariable);
  add("ResourceApplyFtrl"                    , kReadWrite, kVariable);
  add("ResourceApplyFtrlV2"                  , kReadWrite, kVariable);
  add("ResourceApplyGradientDescent"         , kReadWrite, kVariable);
  add("ResourceApplyMomentum"                , kReadWrite, kVariable);
  add("ResourceApplyKerasMomentum"           , kReadWrite, kVariable);
  add("ResourceApplyPowerSign"               , kReadWrite, kVariable);
  add("ResourceApplyProximalAdagrad"         , kReadWrite, kVariable);
  add("ResourceApplyProximalGradientDescent" , kReadWrite, kVariable);
  add("ResourceApplyRMSProp"                 , kReadWrite, kVariable);
  add("ResourceGather"                       , kRead,      kVariable);
  add("ResourceScatterAdd"                   , kReadWrite, kVariable);
  add("ResourceScatterDiv"                   , kReadWrite, kVariable);
  add("ResourceScatterMax"                   , kReadWrite, kVariable);
  add("ResourceScatterMin"                   , kReadWrite, kVariable);
  add("ResourceScatterMul"                   , kReadWrite, kVariable);
  add("ResourceScatterNdAdd"                 , kReadWrite, kVariable);
  add("ResourceScatterNdSub"                 , kReadWrite, kVariable);
  add("ResourceScatterNdUpdate"              , kReadWrite, kVariable);
  add("ResourceScatterSub"                   , kReadWrite, kVariable);
  add("ResourceScatterUpdate"                , kReadWrite, kVariable);
  add("ResourceStridedSliceAssign"           , kReadWrite, kVariable);
  add("RngReadAndSkip"                       , kReadWrite, kVariable);
  add("RngSkip"                              , kReadWrite, kVariable);
  add("StatefulStandardNormalV2"             , kReadWrite, kVariable);
  add("StatefulTruncatedNormal"              , kReadWrite, kVariable);
  add("StatefulUniform"                      , kReadWrite, kVariable);
  add("StatefulUniformFullInt"               , kReadWrite, kVariable);
  add("StatefulUniformInt"                   , kReadWrite, kVariable);
  add("VarIsInitializedOp"                   , kRead,      kVariable);
  add("VariableShape"                        , kRead,      kVariable);

  add("StackV2"                              , kWrite,     kStack);
  add("StackCloseV2"                         , kRead,      kStack);
  add("StackPopV2"                           , kReadWrite, kStack);
  add("StackPushV2"                          , kReadWrite, kStack);

  add("TensorArrayV3"                        , kWrite,     kTensorArray);
  add("TensorArrayConcatV3"                  , kRead,      kTensorArray);
  add("TensorArrayGatherV3"                  , kRead,      kTensorArray);
  add("TensorArrayScatterV3"                 , kWrite,     kTensorArray);
  add("TensorArrayGradV3"                    , kRead,      kTensorArray);
  add("TensorArrayCloseV3"                   , kRead,      kTensorArray);
  add("TensorArrayReadV3"                    , kRead,      kTensorArray);
  add("TensorArraySizeV3"                    , kRead,      kTensorArray);
  add("TensorArraySplitV3"                   , kWrite,     kTensorArray);
  add("TensorArrayWriteV3"                   , kWrite,     kTensorArray);
  // clang-format on

  return result;
}

static const absl::flat_hash_map<absl::string_view, XlaResourceOpInfo>&
GetStaticResourceOpInfoMap() {
  static absl::flat_hash_map<absl::string_view, XlaResourceOpInfo>*
      op_info_map = CreateResourceOpInfoMap();
  return *op_info_map;
}

const XlaResourceOpInfo* GetResourceOpInfoForOp(absl::string_view op) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op: \"" + std::string(op.data(), op.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_operation_tableDTcc mht_2(mht_2_v, 304, "", "./tensorflow/compiler/tf2xla/resource_operation_table.cc", "GetResourceOpInfoForOp");

  const absl::flat_hash_map<absl::string_view, XlaResourceOpInfo>& op_infos =
      GetStaticResourceOpInfoMap();
  auto it = op_infos.find(op);
  return it == op_infos.end() ? nullptr : &it->second;
}

namespace resource_op_table_internal {
std::vector<absl::string_view> GetKnownResourceOps() {
  std::vector<absl::string_view> result;
  for (const auto& p : GetStaticResourceOpInfoMap()) {
    result.push_back(p.first);
  }
  absl::c_sort(result);
  return result;
}
}  // namespace resource_op_table_internal
}  // namespace tensorflow
