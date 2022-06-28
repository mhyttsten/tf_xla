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
class MHTracer_DTPStensorflowPScorePSkernelsPSsave_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsave_opDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/io_ops.cc
#include "tensorflow/core/kernels/save_restore_tensor.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace tensorflow {

class SaveOp : public OpKernel {
 public:
  explicit SaveOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_opDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/save_op.cc", "SaveOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_opDTcc mht_1(mht_1_v, 205, "", "./tensorflow/core/kernels/save_op.cc", "Compute");

    SaveTensors(context, &checkpoint::CreateTableTensorSliceBuilder, false);
  }
};

REGISTER_KERNEL_BUILDER(Name("Save").Device(DEVICE_CPU), SaveOp);

class SaveSlicesOp : public OpKernel {
 public:
  explicit SaveSlicesOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_opDTcc mht_2(mht_2_v, 217, "", "./tensorflow/core/kernels/save_op.cc", "SaveSlicesOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_opDTcc mht_3(mht_3_v, 222, "", "./tensorflow/core/kernels/save_op.cc", "Compute");

    SaveTensors(context, &checkpoint::CreateTableTensorSliceBuilder, true);
  }
};

REGISTER_KERNEL_BUILDER(Name("SaveSlices").Device(DEVICE_CPU), SaveSlicesOp);

class ShardedFilenameOp : public OpKernel {
 public:
  explicit ShardedFilenameOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_opDTcc mht_4(mht_4_v, 234, "", "./tensorflow/core/kernels/save_op.cc", "ShardedFilenameOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_opDTcc mht_5(mht_5_v, 239, "", "./tensorflow/core/kernels/save_op.cc", "Compute");

    static const char* input_names[3] = {"basename", "shard", "num_shards"};
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ctx->input(i).shape()),
                  errors::InvalidArgument(input_names[i],
                                          " must be a scalar, got shape ",
                                          ctx->input(i).shape().DebugString()));
    }
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    out->scalar<tstring>()() = strings::Printf(
        "%s-%05d-of-%05d", ctx->input(0).scalar<tstring>()().c_str(),
        ctx->input(1).scalar<int32>()(), ctx->input(2).scalar<int32>()());
  }
};

REGISTER_KERNEL_BUILDER(Name("ShardedFilename").Device(DEVICE_CPU),
                        ShardedFilenameOp);

class ShardedFilespecOp : public OpKernel {
 public:
  explicit ShardedFilespecOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_opDTcc mht_6(mht_6_v, 263, "", "./tensorflow/core/kernels/save_op.cc", "ShardedFilespecOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsave_opDTcc mht_7(mht_7_v, 268, "", "./tensorflow/core/kernels/save_op.cc", "Compute");

    static const char* input_names[2] = {"basename", "num_shards"};
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ctx->input(i).shape()),
                  errors::InvalidArgument(input_names[i],
                                          " must be a scalar, got shape ",
                                          ctx->input(i).shape().DebugString()));
    }
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    out->scalar<tstring>()() = strings::Printf(
        "%s-\?\?\?\?\?-of-%05d", ctx->input(0).scalar<tstring>()().c_str(),
        ctx->input(1).scalar<int32>()());
  }
};
REGISTER_KERNEL_BUILDER(Name("ShardedFilespec").Device(DEVICE_CPU),
                        ShardedFilespecOp);

}  // namespace tensorflow
