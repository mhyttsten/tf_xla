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
class MHTracer_DTPStensorflowPScorePSkernelsPSsummary_tensor_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_tensor_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsummary_tensor_opDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

template <typename T>
class SummaryTensorOpV2 : public OpKernel {
 public:
  explicit SummaryTensorOpV2(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_tensor_opDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/summary_tensor_op.cc", "SummaryTensorOpV2");
}

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_tensor_opDTcc mht_1(mht_1_v, 205, "", "./tensorflow/core/kernels/summary_tensor_op.cc", "Compute");

    const Tensor& tag = c->input(0);
    OP_REQUIRES(c, TensorShapeUtils::IsScalar(tag.shape()),
                errors::InvalidArgument("tag must be scalar"));
    const Tensor& tensor = c->input(1);
    const Tensor& serialized_summary_metadata_tensor = c->input(2);
    OP_REQUIRES(
        c,
        TensorShapeUtils::IsScalar(serialized_summary_metadata_tensor.shape()),
        errors::InvalidArgument("serialized_summary_metadata must be scalar"));

    Summary s;
    Summary::Value* v = s.add_value();
    v->set_tag(string(tag.scalar<tstring>()()));  // NOLINT

    if (tensor.dtype() == DT_STRING) {
      // tensor_util.makeNdarray doesn't work for strings in tensor_content
      tensor.AsProtoField(v->mutable_tensor());
    } else {
      tensor.AsProtoTensorContent(v->mutable_tensor());
    }

    ParseFromTString(serialized_summary_metadata_tensor.scalar<tstring>()(),
                     v->mutable_metadata());

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(SerializeToTString(s, &summary_tensor->scalar<tstring>()()));
  }
};

#define REGISTER(T)                                                      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("TensorSummaryV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SummaryTensorOpV2<T>);

TF_CALL_ALL_TYPES(REGISTER)

#undef REGISTER

// NOTE(chizeng): We are phasing out the use of SummaryTensorOp in favor of
// SummaryTensorOpV2. This is because SummaryTensorOpV2 allows the callers to
// pass a tag (more consistent with other summaries) as well as serialized
// summary metadata used by plugins (which lets TensorBoard determine which
// events are relevant to which plugins).
template <typename T>
class SummaryTensorOp : public OpKernel {
 public:
  explicit SummaryTensorOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_tensor_opDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/kernels/summary_tensor_op.cc", "SummaryTensorOp");
}

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_tensor_opDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/kernels/summary_tensor_op.cc", "Compute");

    const Tensor& tensor = c->input(0);

    Summary s;
    Summary::Value* v = s.add_value();
    v->set_node_name(c->op_kernel().name());

    if (tensor.dtype() == DT_STRING) {
      // tensor_util.makeNdarray doesn't work for strings in tensor_content
      tensor.AsProtoField(v->mutable_tensor());
    } else {
      tensor.AsProtoTensorContent(v->mutable_tensor());
    }

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(SerializeToTString(s, &summary_tensor->scalar<tstring>()()));
  }
};

#define REGISTER(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("TensorSummary").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SummaryTensorOp<T>);

TF_CALL_ALL_TYPES(REGISTER)

#undef REGISTER

}  // namespace tensorflow
