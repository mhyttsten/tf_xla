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
class MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_utilDTcc() {
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
#include "tensorflow/core/common_runtime/collective_util.h"

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace collective_util {

/*static*/
Status InitializeDeviceAndLocality(const DeviceMgr* dev_mgr,
                                   const string& device_name, Device** device,
                                   DeviceLocality* device_locality) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_utilDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/common_runtime/collective_util.cc", "InitializeDeviceAndLocality");

  if (!dev_mgr) {
    return errors::Internal("Required non-null dev_mgr ", dev_mgr,
                            " for InitializeDeviceAndLocality");
  }

  Status status = dev_mgr->LookupDevice(device_name, device);
  if (status.ok()) {
    CHECK(*device);
    *device_locality = (*device)->attributes().locality();
  } else {
    LOG(ERROR) << "Failed to find device " << device_name;
    for (auto d : dev_mgr->ListDevices()) {
      LOG(ERROR) << "Available devices " << d->name();
    }
  }
  return status;
}

/*static*/
string SubdivPermDebugString(const CollectiveParams& col_params) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_utilDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/common_runtime/collective_util.cc", "SubdivPermDebugString");

  const auto& subdiv_perms =
      col_params.instance.impl_details.subdiv_permutations;
  string buf;
  for (int sdi = 0; sdi < subdiv_perms.size(); ++sdi) {
    strings::StrAppend(&buf, "Subdiv ", sdi, " device order:\n");
    for (int di = 0; di < subdiv_perms[sdi].size(); ++di) {
      int idx = subdiv_perms[sdi][di];
      if (idx >= 0) {
        CHECK_GT(col_params.group.members.size(), idx);
        strings::StrAppend(&buf, col_params.group.members[idx].device.name(),
                           "\n");
      }
    }
    strings::StrAppend(&buf, " subdiv_offsets: ");
    for (auto o : col_params.instance.impl_details.subdiv_offsets)
      strings::StrAppend(&buf, o, " ");
    strings::StrAppend(&buf, " SubdivRank: ");
    for (auto d : col_params.subdiv_rank) strings::StrAppend(&buf, d, " ");
    if (col_params.instance.type == BROADCAST_COLLECTIVE) {
      strings::StrAppend(&buf, " subdiv_source_rank: ");
      for (auto src : col_params.instance.impl_details.subdiv_source_rank)
        strings::StrAppend(&buf, src, " ");
    }
    strings::StrAppend(&buf, "\n");
  }
  return buf;
}

SubContext::SubContext(OpKernelContext* ctx, OpKernelContext::Params* params,
                       OpKernel* op, Tensor* output, Tensor* input)
    : sub_params_(*params),
      sub_inputs_({TensorValue(output), TensorValue(input)}),
      sub_input_attr_({ctx->input_alloc_attr(0), ctx->input_alloc_attr(0)}) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_utilDTcc mht_2(mht_2_v, 263, "", "./tensorflow/core/common_runtime/collective_util.cc", "SubContext::SubContext");

  sub_params_.op_kernel = op;
  sub_params_.inputs = &sub_inputs_;
  sub_params_.input_alloc_attrs = &sub_input_attr_;
  sub_params_.op_device_context = ctx->op_device_context();
  sub_params_.eigen_gpu_device = nullptr;
  sub_params_.ensure_eigen_gpu_device();
  sub_params_.forward_from_array = &forward_from_;
  sub_ctx_.reset(new OpKernelContext(&sub_params_, 1));
}

Status ComputeBinOp(OpKernelContext* op_ctx, OpKernelContext::Params* params,
                    Device* device, OpKernel* op, Tensor* output,
                    Tensor* input) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_utilDTcc mht_3(mht_3_v, 279, "", "./tensorflow/core/common_runtime/collective_util.cc", "ComputeBinOp");

  // Prepare an OpKernelContext that is identical to that of the original Op
  // (i.e. the collective), except for the input output sizes and identities and
  // the Op itself.
  // TODO(ayushd, tucker): Is it possible to cache and reuse these objects?
  // They're mostly identical inside one device execution.
  std::unique_ptr<SubContext> sub_ctx(
      new SubContext(op_ctx, params, op, output, input));
  device->Compute(op, sub_ctx->sub_ctx_.get());
  return sub_ctx->sub_ctx_->status();
}

}  // namespace collective_util
}  // namespace tensorflow
