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
class MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc() {
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

#include "tensorflow/core/kernels/logging_ops.h"

#include <iostream>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/determinism.h"

namespace tensorflow {

namespace {

// If the following string is found at the beginning of an output stream, it
// will be interpreted as a file path.
const char kOutputStreamEscapeStr[] = "file://";

// A mutex that guards appending strings to files.
static mutex* file_mutex = new mutex();

// Appends the given data to the specified file. It will create the file if it
// doesn't already exist.
Status AppendStringToFile(const std::string& fname, StringPiece data,
                          Env* env) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/logging_ops.cc", "AppendStringToFile");

  // TODO(ckluk): If opening and closing on every log causes performance issues,
  // we can reimplement using reference counters.
  mutex_lock l(*file_mutex);
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewAppendableFile(fname, &file));
  Status a = file->Append(data);
  Status c = file->Close();
  return a.ok() ? c : a;
}

}  // namespace

AssertOp::AssertOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/kernels/logging_ops.cc", "AssertOp::AssertOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr("summarize", &summarize_));
}

void AssertOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/kernels/logging_ops.cc", "AssertOp::Compute");

  const Tensor& cond = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(cond.shape()),
              errors::InvalidArgument("In[0] should be a scalar: ",
                                      cond.shape().DebugString()));

  if (cond.scalar<bool>()()) {
    return;
  }
  string msg = "assertion failed: ";
  for (int i = 1; i < ctx->num_inputs(); ++i) {
    strings::StrAppend(&msg, "[", ctx->input(i).SummarizeValue(summarize_),
                       "]");
    if (i < ctx->num_inputs() - 1) strings::StrAppend(&msg, " ");
  }
  ctx->SetStatus(errors::InvalidArgument(msg));
}

REGISTER_KERNEL_BUILDER(Name("Assert")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("condition")
                            .HostMemory("data"),
                        AssertOp);

class PrintOp : public OpKernel {
 public:
  explicit PrintOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), call_counter_(0) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc mht_3(mht_3_v, 264, "", "./tensorflow/core/kernels/logging_ops.cc", "PrintOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("message", &message_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("first_n", &first_n_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("summarize", &summarize_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc mht_4(mht_4_v, 273, "", "./tensorflow/core/kernels/logging_ops.cc", "Compute");

    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    } else {
      ctx->set_output(0, ctx->input(0));
    }
    if (first_n_ >= 0) {
      mutex_lock l(mu_);
      if (call_counter_ >= first_n_) return;
      call_counter_++;
    }
    string msg;
    strings::StrAppend(&msg, message_);
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      strings::StrAppend(&msg, "[", ctx->input(i).SummarizeValue(summarize_),
                         "]");
    }
    std::cerr << msg << std::endl;
  }

 private:
  mutex mu_;
  int64_t call_counter_ TF_GUARDED_BY(mu_) = 0;
  int64_t first_n_ = 0;
  int32 summarize_ = 0;
  string message_;
};

REGISTER_KERNEL_BUILDER(Name("Print").Device(DEVICE_CPU), PrintOp);

class PrintV2Op : public OpKernel {
 public:
  explicit PrintV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc mht_5(mht_5_v, 308, "", "./tensorflow/core/kernels/logging_ops.cc", "PrintV2Op");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_stream", &output_stream_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("end", &end_));

    SetFilePathIfAny();
    if (!file_path_.empty()) return;

    auto output_stream_index =
        std::find(std::begin(valid_output_streams_),
                  std::end(valid_output_streams_), output_stream_);

    if (output_stream_index == std::end(valid_output_streams_)) {
      string error_msg = strings::StrCat(
          "Unknown output stream: ", output_stream_, ", Valid streams are:");
      for (auto valid_stream : valid_output_streams_) {
        strings::StrAppend(&error_msg, " ", valid_stream);
      }
      OP_REQUIRES(ctx, false, errors::InvalidArgument(error_msg));
    }
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc mht_6(mht_6_v, 332, "", "./tensorflow/core/kernels/logging_ops.cc", "Compute");

    const Tensor* input_;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(input_->shape()),
        errors::InvalidArgument("Input is expected to be scalar, but got ",
                                input_->shape()));
    const string& msg = input_->scalar<tstring>()();

    string ended_msg = strings::StrCat(msg, end_);

    if (!file_path_.empty()) {
      // Outputs to a file at the specified path.
      OP_REQUIRES_OK(ctx,
                     AppendStringToFile(file_path_, ended_msg, ctx->env()));
      return;
    }

    if (logging::LogToListeners(ended_msg, "")) {
      return;
    }

    if (output_stream_ == "stdout") {
      std::cout << ended_msg << std::flush;
    } else if (output_stream_ == "stderr") {
      std::cerr << ended_msg << std::flush;
    } else if (output_stream_ == "log(info)") {
      LOG(INFO) << ended_msg << std::flush;
    } else if (output_stream_ == "log(warning)") {
      LOG(WARNING) << ended_msg << std::flush;
    } else if (output_stream_ == "log(error)") {
      LOG(ERROR) << ended_msg << std::flush;
    } else {
      string error_msg = strings::StrCat(
          "Unknown output stream: ", output_stream_, ", Valid streams are:");
      for (auto valid_stream : valid_output_streams_) {
        strings::StrAppend(&error_msg, " ", valid_stream);
      }
      strings::StrAppend(&error_msg, ", or file://<filename>");
      OP_REQUIRES(ctx, false, errors::InvalidArgument(error_msg));
    }
  }

  const char* valid_output_streams_[5] = {"stdout", "stderr", "log(info)",
                                          "log(warning)", "log(error)"};

 private:
  string end_;
  // Either output_stream_ or file_path_ (but not both) will be non-empty.
  string output_stream_;
  string file_path_;

  // If output_stream_ is a file path, extracts it to file_path_ and clears
  // output_stream_; otherwise sets file_paths_ to "".
  void SetFilePathIfAny() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc mht_7(mht_7_v, 389, "", "./tensorflow/core/kernels/logging_ops.cc", "SetFilePathIfAny");

    if (absl::StartsWith(output_stream_, kOutputStreamEscapeStr)) {
      file_path_ = output_stream_.substr(strlen(kOutputStreamEscapeStr));
      output_stream_ = "";
    } else {
      file_path_ = "";
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("PrintV2").Device(DEVICE_CPU), PrintV2Op);

class TimestampOp : public OpKernel {
 public:
  explicit TimestampOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc mht_8(mht_8_v, 406, "", "./tensorflow/core/kernels/logging_ops.cc", "TimestampOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlogging_opsDTcc mht_9(mht_9_v, 411, "", "./tensorflow/core/kernels/logging_ops.cc", "Compute");

    OP_REQUIRES(context, !OpDeterminismRequired(),
                errors::FailedPrecondition(
                    "Timestamp cannot be called when determinism is enabled"));
    TensorShape output_shape;  // Default shape is 0 dim, 1 element
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    auto output_scalar = output_tensor->scalar<double>();
    double now_us = static_cast<double>(Env::Default()->NowMicros());
    double now_s = now_us / 1000000;
    output_scalar() = now_s;
  }
};

REGISTER_KERNEL_BUILDER(Name("Timestamp").Device(DEVICE_CPU), TimestampOp);

}  // end namespace tensorflow
