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
class MHTracer_DTPStensorflowPScorePSkernelsPSregex_replace_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSregex_replace_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSregex_replace_opDTcc() {
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

#include <string>

#include "re2/re2.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

// Execute the specified regex using the given context.
// Context requirements:
//  - "input" string Tensor at input_index=0
//  - "output" string Tensor at output_index=0
Status InternalCompute(const RE2& regex, const string& rewrite,
                       const bool replace_global, OpKernelContext* ctx) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("rewrite: \"" + rewrite + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSregex_replace_opDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/regex_replace_op.cc", "InternalCompute");

  const Tensor* input_tensor;
  TF_RETURN_IF_ERROR(ctx->input("input", &input_tensor));
  Tensor* output_tensor;
  std::unique_ptr<Tensor> maybe_forwarded =
      ctx->forward_input(0 /*input_index*/, 0 /*output_index*/,
                         tensorflow::DT_STRING, input_tensor->shape(),
                         ctx->input_memory_type(0), ctx->input_alloc_attr(0));
  if (maybe_forwarded) {
    output_tensor = maybe_forwarded.get();
    TF_RETURN_IF_ERROR(ctx->set_output("output", *output_tensor));
  } else {
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("output", input_tensor->shape(), &output_tensor));
    output_tensor->flat<tstring>() = input_tensor->flat<tstring>();
  }
  auto output_flat = output_tensor->flat<tstring>();
  for (size_t i = 0; i < output_flat.size(); ++i) {
    // TODO(dero): Mitigate copy; Global and GlobalReplace below currently only
    // accept std::string.
    string buf = output_flat(i);
    if (replace_global) {
      RE2::GlobalReplace(&buf, regex, rewrite);
    } else {
      RE2::Replace(&buf, regex, rewrite);
    }
    output_flat(i) = std::move(buf);
  }
  return Status::OK();
}
}  // namespace

class RegexReplaceOp : public OpKernel {
 public:
  explicit RegexReplaceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSregex_replace_opDTcc mht_1(mht_1_v, 242, "", "./tensorflow/core/kernels/regex_replace_op.cc", "RegexReplaceOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("replace_global", &replace_global_));
  }

  ~RegexReplaceOp() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSregex_replace_opDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/kernels/regex_replace_op.cc", "~RegexReplaceOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSregex_replace_opDTcc mht_3(mht_3_v, 254, "", "./tensorflow/core/kernels/regex_replace_op.cc", "Compute");

    const Tensor* pattern_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("pattern", &pattern_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(pattern_tensor->shape()),
                errors::InvalidArgument("Pattern must be scalar, but received ",
                                        pattern_tensor->shape().DebugString()));
    const string& pattern = pattern_tensor->scalar<tstring>()();
    std::shared_ptr<RE2> regex = CachedRE2(pattern);
    OP_REQUIRES(ctx, regex->ok(),
                errors::InvalidArgument("Invalid pattern: ", pattern,
                                        ", error: ", regex->error()));

    const Tensor* rewrite_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("rewrite", &rewrite_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rewrite_tensor->shape()),
                errors::InvalidArgument("Rewrite must be scalar, but received ",
                                        rewrite_tensor->shape().DebugString()));
    const string& rewrite = rewrite_tensor->scalar<tstring>()();
    OP_REQUIRES_OK(ctx, InternalCompute(*regex, rewrite, replace_global_, ctx));
  }

 private:
  std::shared_ptr<RE2> CachedRE2(const string& pattern) {
    {
      tf_shared_lock l(mu_);
      if (regex_ != nullptr && regex_->pattern() == pattern) {
        return regex_;
      }
    }
    // Construct the new RE2 object before acquiring the lock.
    auto regex = std::make_shared<RE2>(pattern);
    {
      mutex_lock l(mu_);
      // Swap instead of assigning so that we destruct the old
      // RE2 object (when necessary) after releasing the lock.
      regex_.swap(regex);
      return regex_;
    }
  }

  bool replace_global_;
  mutex mu_;
  std::shared_ptr<RE2> regex_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RegexReplaceOp);
};

REGISTER_KERNEL_BUILDER(Name("RegexReplace").Device(DEVICE_CPU),
                        RegexReplaceOp);

class StaticRegexReplaceOp : public OpKernel {
 public:
  explicit StaticRegexReplaceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSregex_replace_opDTcc mht_4(mht_4_v, 309, "", "./tensorflow/core/kernels/regex_replace_op.cc", "StaticRegexReplaceOp");

    string pattern;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pattern", &pattern));
    re_ = MakeUnique<RE2>(pattern);
    OP_REQUIRES(ctx, re_->ok(),
                errors::InvalidArgument("Invalid pattern: ", pattern,
                                        ", error: ", re_->error()));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rewrite", &rewrite_str_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("replace_global", &replace_global_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSregex_replace_opDTcc mht_5(mht_5_v, 323, "", "./tensorflow/core/kernels/regex_replace_op.cc", "Compute");

    OP_REQUIRES_OK(ctx,
                   InternalCompute(*re_, rewrite_str_, replace_global_, ctx));
  }

 private:
  std::unique_ptr<RE2> re_;
  string rewrite_str_;
  bool replace_global_;
};

REGISTER_KERNEL_BUILDER(Name("StaticRegexReplace").Device(DEVICE_CPU),
                        StaticRegexReplaceOp);

}  // namespace tensorflow
