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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc() {
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
#ifdef INTEL_MKL
#include <string>
#include <unordered_map>

#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

class MklEagerOpRewrite : public EagerOpRewrite {
 public:
  MklEagerOpRewrite(string name, string file, string line);
  struct MklEagerOp {
    string op_name;
    std::function<bool(EagerOperation*)> RewriteRule;
    std::function<Status(EagerOperation*, std::unique_ptr<EagerOperation>*)>
        CreateMklOp;
  };

 private:
  std::unordered_map<std::string, MklEagerOp> mkl_eager_ops_;

  // The entry point to execute the op rewrite.
  Status Run(EagerOperation* orig_op,
             std::unique_ptr<tensorflow::EagerOperation>* out_op);

  // Initializes the new op and sets up its inputs and attributes
  static Status SetupNewOp(EagerOperation* orig_op, const string mkl_op_name,
                           std::unique_ptr<EagerOperation>* new_mkl_op);

  // Generic rewrite that can be used for any mkl op that doesn't need
  // special processing.
  static Status CreateGenericMklOp(EagerOperation* orig_op,
                                   std::unique_ptr<EagerOperation>* mkl_op);

  // Rewrite rule for Conv2D, Conv2DBackpropInput and Conv2DBackpropFilter.
  static bool RewriteConv2D(EagerOperation* op);

  // Rewrite rule for FusedBatchNormV3 and FusedBatchNormGradV3
  static bool RewriteFusedBatchNormV3(EagerOperation* op);

  // Calls op-specific rewrite function to create new MKL op.
  Status RewriteToMklOp(EagerOperation* orig_op,
                        std::unique_ptr<EagerOperation>* mkl_op);

  // Check whether we can rewrite the op to MKL one or not.
  bool ShouldRewriteOp(EagerOperation* op);

  // Default rewrite rule to be used when rewrite should happen without any
  // restriction.
  static bool AlwaysRewrite(EagerOperation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc mht_0(mht_0_v, 237, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc", "AlwaysRewrite");
 return true; }

  // Check if kernel is registered for a particular op.
  bool IsKernelRegistered(string op_name, DataType dt);

  // Helper function to insert mkl_eager_ops to Map
  void InsertMKLEagerOps(MklEagerOp op);
};

REGISTER_REWRITE(EagerOpRewriteRegistry::POST_PLACEMENT, 10000,
                 MklEagerOpRewrite);

// Constructor
MklEagerOpRewrite::MklEagerOpRewrite(string name, string file, string line)
    : EagerOpRewrite(name, file, line) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   mht_1_v.push_back("file: \"" + file + "\"");
   mht_1_v.push_back("line: \"" + line + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc mht_1(mht_1_v, 257, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc", "MklEagerOpRewrite::MklEagerOpRewrite");

  InsertMKLEagerOps({"AvgPool", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"AvgPoolGrad", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"AvgPool3D", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"AvgPool3DGrad", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"BatchMatMul", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"BatchMatMulV2", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"Conv2D", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"Conv2DBackpropFilter", RewriteConv2D, CreateGenericMklOp});
  InsertMKLEagerOps({"Conv2DBackpropInput", RewriteConv2D, CreateGenericMklOp});
  InsertMKLEagerOps({"Conv3D", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"Conv3DBackpropFilterV2", RewriteConv2D, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"Conv3DBackpropInputV2", RewriteConv2D, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"DepthwiseConv2dNative", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"DepthwiseConv2dNativeBackpropFilter", RewriteConv2D,
                     CreateGenericMklOp});
  InsertMKLEagerOps({"DepthwiseConv2dNativeBackpropInput", RewriteConv2D,
                     CreateGenericMklOp});
  InsertMKLEagerOps({"Einsum", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"FusedBatchNorm", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"FusedBatchNormGrad", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"FusedBatchNormGradV2", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"FusedBatchNormGradV3", RewriteFusedBatchNormV3, CreateGenericMklOp});
  InsertMKLEagerOps({"FusedBatchNormV2", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps(
      {"FusedBatchNormV3", RewriteFusedBatchNormV3, CreateGenericMklOp});
  InsertMKLEagerOps({"MatMul", AlwaysRewrite, CreateGenericMklOp});
};

void MklEagerOpRewrite::InsertMKLEagerOps(MklEagerOp op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc mht_2(mht_2_v, 295, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc", "MklEagerOpRewrite::InsertMKLEagerOps");

  mkl_eager_ops_.insert(std::make_pair(op.op_name, op));
}

Status MklEagerOpRewrite::Run(
    EagerOperation* orig_op,
    std::unique_ptr<tensorflow::EagerOperation>* out_op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc mht_3(mht_3_v, 304, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc", "MklEagerOpRewrite::Run");

  if (ShouldRewriteOp(orig_op)) {
    TF_CHECK_OK(RewriteToMklOp(orig_op, out_op));
  }
  return Status::OK();
}

Status MklEagerOpRewrite::SetupNewOp(
    EagerOperation* orig_op, const string mkl_op_name,
    std::unique_ptr<EagerOperation>* new_mkl_op) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("mkl_op_name: \"" + mkl_op_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc mht_4(mht_4_v, 317, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc", "MklEagerOpRewrite::SetupNewOp");

  bool is_remote = false;
  new_mkl_op->reset(new tensorflow::EagerOperation(&orig_op->EagerContext()));
  TF_RETURN_IF_ERROR(new_mkl_op->get()->Reset(mkl_op_name.c_str(), nullptr,
                                              is_remote, nullptr));

  int num_inputs = orig_op->Inputs().size();
  // Add all inputs to the new op.
  for (int i = 0; i < num_inputs; ++i) {
    TF_RETURN_IF_ERROR((*new_mkl_op)->AddInput(orig_op->Inputs()[i]));
  }

  // Copy all attributes to the new op.
  const NodeDef& orig_ndef = orig_op->MutableAttrs()->BuildNodeDef();

  AttrSlice attr_list(orig_ndef);
  for (const auto& attr : attr_list) {
    (*new_mkl_op)->MutableAttrs()->Set(attr.first, attr.second);
  }

  if (!orig_op->EagerContext().RunEagerOpAsFunction()) {
    (*new_mkl_op)
        ->MutableAttrs()
        ->Set("_kernel", mkl_op_registry::kMklNameChangeOpLabel);
  }

  string device_name = orig_op->DeviceName();
  return (*new_mkl_op)->SetDeviceName(device_name.c_str());
}

Status MklEagerOpRewrite::CreateGenericMklOp(
    EagerOperation* orig_op, std::unique_ptr<EagerOperation>* mkl_op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc mht_5(mht_5_v, 351, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc", "MklEagerOpRewrite::CreateGenericMklOp");

  const string mkl_op_name =
      mkl_op_registry::GetMklNativeOpName(orig_op->Name());
  TF_CHECK_OK(SetupNewOp(orig_op, mkl_op_name, mkl_op));
  return Status::OK();
}

bool MklEagerOpRewrite::ShouldRewriteOp(EagerOperation* op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc mht_6(mht_6_v, 361, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc", "MklEagerOpRewrite::ShouldRewriteOp");

  // Don't rewrite the op if MKL use is disabled at runtime.
  if (!IsMKLEnabled()) {
    return false;
  }
  DataType data_type;
  if (op->Attrs().Get("T", &data_type) != Status::OK()) {
    return false;
  }
  // Only rewrite if op is to be run on CPU device.
  if (op->GetDeviceParsedName().type != "CPU") {
    return false;
  }
  // Check if we have registered MKL kernel for this op.
  bool kernel_found = IsKernelRegistered(op->Name(), data_type);
  if (!kernel_found) {
    return false;
  }

  // Find and call the op's rewrite rule that determines whether we need to
  // rewrite this op or not.
  auto it = mkl_eager_ops_.find(op->Name());
  if (it != mkl_eager_ops_.end()) {
    // Eager op found so verify Rewrite
    if (it->second.RewriteRule(op)) {
      return true;
    }
  }
  return false;
}

bool MklEagerOpRewrite::IsKernelRegistered(string op_name, DataType dt) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc mht_7(mht_7_v, 396, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc", "MklEagerOpRewrite::IsKernelRegistered");

  // Find if the eager op_name exists in mkl_eager_ops_ list.
  auto element = mkl_eager_ops_.find(op_name);
  if (element != mkl_eager_ops_.end()) {
    // Eager Op exists. So verify registry and return registered or not.
    return (mkl_op_registry::IsMklOp(
                mkl_op_registry::GetMklNativeOpName(op_name), dt, true) ||
            mkl_op_registry::IsMklOp(mkl_op_registry::GetMklOpName(op_name), dt,
                                     true));
  } else {
    return false;
  }
}

Status MklEagerOpRewrite::RewriteToMklOp(
    EagerOperation* orig_op, std::unique_ptr<EagerOperation>* mkl_op) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc mht_8(mht_8_v, 414, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc", "MklEagerOpRewrite::RewriteToMklOp");

  // TODO(intel-tf): mkl_eager_ops_ lookup can be reduced from twice
  // (once each in ShouldRewriteOp & RewriteToMklOp) to just once.
  TF_RETURN_IF_ERROR(
      mkl_eager_ops_[orig_op->Name()].CreateMklOp(orig_op, mkl_op));
  return Status::OK();
}

bool MklEagerOpRewrite::RewriteConv2D(EagerOperation* op) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc mht_9(mht_9_v, 425, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc", "MklEagerOpRewrite::RewriteConv2D");

  const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
  string padding;
  TF_CHECK_OK(GetNodeAttr(ndef, "padding", &padding));
  // Right now MKL Conv2D does not support explicit padding.
  return (padding != "EXPLICIT");
}

bool MklEagerOpRewrite::RewriteFusedBatchNormV3(EagerOperation* op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSmkl_eager_op_rewriteDTcc mht_10(mht_10_v, 436, "", "./tensorflow/core/common_runtime/eager/mkl_eager_op_rewrite.cc", "MklEagerOpRewrite::RewriteFusedBatchNormV3");

  const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
  if (Check5DFormat(ndef)) {
    VLOG(1) << "Eager Op Rewrite: FusedBatchNorm(Grad)V3 op currently does not "
            << "support 5D tensors.";
    return false;
  }
  return true;
}

}  // namespace tensorflow
#endif  // INTEL_MKL
