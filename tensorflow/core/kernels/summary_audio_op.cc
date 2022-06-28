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
class MHTracer_DTPStensorflowPScorePSkernelsPSsummary_audio_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_audio_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsummary_audio_opDTcc() {
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

// Operators that deal with SummaryProtos (encoded as DT_STRING tensors) as
// inputs or outputs in various ways.

// See docs in ../ops/summary_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/wav/wav_io.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class SummaryAudioOp : public OpKernel {
 public:
  explicit SummaryAudioOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_audio_opDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/summary_audio_op.cc", "SummaryAudioOp");

    OP_REQUIRES_OK(context, context->GetAttr("max_outputs", &max_outputs_));
    OP_REQUIRES(context, max_outputs_ > 0,
                errors::InvalidArgument("max_outputs must be > 0"));
    has_sample_rate_attr_ =
        context->GetAttr("sample_rate", &sample_rate_attr_).ok();
  }

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_audio_opDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/kernels/summary_audio_op.cc", "Compute");

    const Tensor& tag = c->input(0);
    const Tensor& tensor = c->input(1);
    OP_REQUIRES(c, TensorShapeUtils::IsScalar(tag.shape()),
                errors::InvalidArgument("Tag must be a scalar"));
    OP_REQUIRES(c, tensor.dims() >= 2 && tensor.dims() <= 3,
                errors::InvalidArgument("Tensor must be 3-D or 2-D, got: ",
                                        tensor.shape().DebugString()));
    const string& base_tag = tag.scalar<tstring>()();

    float sample_rate = sample_rate_attr_;
    if (!has_sample_rate_attr_) {
      const Tensor& sample_rate_tensor = c->input(2);
      sample_rate = sample_rate_tensor.scalar<float>()();
    }
    OP_REQUIRES(c, sample_rate > 0.0f,
                errors::InvalidArgument("sample_rate must be > 0"));

    const int batch_size = tensor.dim_size(0);
    const int64_t length_frames = tensor.dim_size(1);
    const int64_t num_channels =
        tensor.dims() == 2 ? 1 : tensor.dim_size(tensor.dims() - 1);

    Summary s;
    const int N = std::min<int>(max_outputs_, batch_size);
    for (int i = 0; i < N; ++i) {
      Summary::Value* v = s.add_value();
      if (max_outputs_ > 1) {
        v->set_tag(strings::StrCat(base_tag, "/audio/", i));
      } else {
        v->set_tag(strings::StrCat(base_tag, "/audio"));
      }

      Summary::Audio* sa = v->mutable_audio();
      sa->set_sample_rate(sample_rate);
      sa->set_num_channels(num_channels);
      sa->set_length_frames(length_frames);
      sa->set_content_type("audio/wav");

      auto values =
          tensor.shaped<float, 3>({batch_size, length_frames, num_channels});
      const float* data =
          tensor.NumElements() == 0 ? nullptr : &values(i, 0, 0);

      size_t sample_rate_truncated = lrintf(sample_rate);
      if (sample_rate_truncated == 0) {
        sample_rate_truncated = 1;
      }
      OP_REQUIRES_OK(c, wav::EncodeAudioAsS16LEWav(
                            data, sample_rate_truncated, num_channels,
                            length_frames, sa->mutable_encoded_audio_string()));
    }

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(SerializeToTString(s, &summary_tensor->scalar<tstring>()()));
  }

 private:
  int max_outputs_;
  bool has_sample_rate_attr_;
  float sample_rate_attr_;
};

REGISTER_KERNEL_BUILDER(Name("AudioSummaryV2").Device(DEVICE_CPU),
                        SummaryAudioOp);

// Deprecated -- this op is registered with sample_rate as an attribute for
// backwards compatibility.
REGISTER_KERNEL_BUILDER(Name("AudioSummary").Device(DEVICE_CPU),
                        SummaryAudioOp);

}  // namespace tensorflow
