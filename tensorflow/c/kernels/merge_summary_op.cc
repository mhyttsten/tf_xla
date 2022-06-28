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
class MHTracer_DTPStensorflowPScPSkernelsPSmerge_summary_opDTcc {
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
   MHTracer_DTPStensorflowPScPSkernelsPSmerge_summary_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSkernelsPSmerge_summary_opDTcc() {
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
#include <memory>
#include <sstream>
#include <unordered_set>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tstring.h"

namespace {

// Operators used to create a std::unique_ptr for TF_Tensor and TF_Status
struct TFTensorDeleter {
  void operator()(TF_Tensor* tf_tensor) const { TF_DeleteTensor(tf_tensor); }
};

struct TFStatusDeleter {
  void operator()(TF_Status* tf_status) const { TF_DeleteStatus(tf_status); }
};

// Struct that wraps TF_Tensor and TF_Status to delete once out of scope
using Safe_TF_TensorPtr = std::unique_ptr<TF_Tensor, TFTensorDeleter>;
using Safe_TF_StatusPtr = std::unique_ptr<TF_Status, TFStatusDeleter>;

// dummy functions used for kernel registration
void* MergeSummaryOp_Create(TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSkernelsPSmerge_summary_opDTcc mht_0(mht_0_v, 215, "", "./tensorflow/c/kernels/merge_summary_op.cc", "MergeSummaryOp_Create");
 return nullptr; }

void MergeSummaryOp_Delete(void* kernel) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSkernelsPSmerge_summary_opDTcc mht_1(mht_1_v, 220, "", "./tensorflow/c/kernels/merge_summary_op.cc", "MergeSummaryOp_Delete");
}

void MergeSummaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSkernelsPSmerge_summary_opDTcc mht_2(mht_2_v, 225, "", "./tensorflow/c/kernels/merge_summary_op.cc", "MergeSummaryOp_Compute");

  tensorflow::Summary s;
  std::unordered_set<tensorflow::string> tags;
  Safe_TF_StatusPtr status(TF_NewStatus());
  for (int input_num = 0; input_num < TF_NumInputs(ctx); ++input_num) {
    TF_Tensor* input;
    TF_GetInput(ctx, input_num, &input, status.get());
    Safe_TF_TensorPtr safe_input_ptr(input);
    if (TF_GetCode(status.get()) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status.get());
      return;
    }
    auto tags_array =
        static_cast<tensorflow::tstring*>(TF_TensorData(safe_input_ptr.get()));
    for (int i = 0; i < TF_TensorElementCount(safe_input_ptr.get()); ++i) {
      const tensorflow::tstring& s_in = tags_array[i];
      tensorflow::Summary summary_in;
      if (!tensorflow::ParseProtoUnlimited(&summary_in, s_in)) {
        TF_SetStatus(status.get(), TF_INVALID_ARGUMENT,
                     "Could not parse one of the summary inputs");
        TF_OpKernelContext_Failure(ctx, status.get());
        return;
      }
      for (int v = 0; v < summary_in.value_size(); ++v) {
        // This tag is unused by the TensorSummary op, so no need to check for
        // duplicates.
        const tensorflow::string& tag = summary_in.value(v).tag();
        if ((!tag.empty()) && !tags.insert(tag).second) {
          std::ostringstream err;
          err << "Duplicate tag " << tag << " found in summary inputs ";
          TF_SetStatus(status.get(), TF_INVALID_ARGUMENT, err.str().c_str());
          TF_OpKernelContext_Failure(ctx, status.get());
          return;
        }
        *s.add_value() = summary_in.value(v);
      }
    }
  }
  Safe_TF_TensorPtr summary_tensor(TF_AllocateOutput(
      /*context=*/ctx, /*index=*/0, /*dtype=*/TF_ExpectedOutputDataType(ctx, 0),
      /*dims=*/nullptr, /*num_dims=*/0,
      /*len=*/sizeof(tensorflow::tstring), status.get()));
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  tensorflow::tstring* output_tstring = reinterpret_cast<tensorflow::tstring*>(
      TF_TensorData(summary_tensor.get()));
  CHECK(SerializeToTString(s, output_tstring));
}

void RegisterMergeSummaryOpKernel() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSkernelsPSmerge_summary_opDTcc mht_3(mht_3_v, 279, "", "./tensorflow/c/kernels/merge_summary_op.cc", "RegisterMergeSummaryOpKernel");

  TF_Status* status = TF_NewStatus();
  {
    auto* builder = TF_NewKernelBuilder(
        "MergeSummary", tensorflow::DEVICE_CPU, &MergeSummaryOp_Create,
        &MergeSummaryOp_Compute, &MergeSummaryOp_Delete);
    TF_RegisterKernelBuilder("MergeSummary", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering Merge Summmary kernel";
  }
  TF_DeleteStatus(status);
}

// A dummy static variable initialized by a lambda whose side-effect is to
// register the Histogram Summary kernel.
TF_ATTRIBUTE_UNUSED static bool IsMergeSummaryOpKernelRegistered = []() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSkernelsPSmerge_summary_opDTcc mht_4(mht_4_v, 297, "", "./tensorflow/c/kernels/merge_summary_op.cc", "lambda");

  if (SHOULD_REGISTER_OP_KERNEL("MergeSummary")) {
    RegisterMergeSummaryOpKernel();
  }
  return true;
}();

}  // namespace
