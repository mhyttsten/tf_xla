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
class MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/summary/schema.h"
#include "tensorflow/core/summary/summary_db_writer.h"
#include "tensorflow/core/summary/summary_file_writer.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("SummaryWriter").Device(DEVICE_CPU),
                        ResourceHandleOp<SummaryWriterInterface>);

class CreateSummaryFileWriterOp : public OpKernel {
 public:
  explicit CreateSummaryFileWriterOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/summary_kernels.cc", "CreateSummaryFileWriterOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("logdir", &tmp));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tmp->shape()),
                errors::InvalidArgument("logdir must be a scalar"));
    const string logdir = tmp->scalar<tstring>()();
    OP_REQUIRES_OK(ctx, ctx->input("max_queue", &tmp));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tmp->shape()),
                errors::InvalidArgument("max_queue must be a scalar"));
    const int32_t max_queue = tmp->scalar<int32>()();
    OP_REQUIRES_OK(ctx, ctx->input("flush_millis", &tmp));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tmp->shape()),
                errors::InvalidArgument("flush_millis must be a scalar"));
    const int32_t flush_millis = tmp->scalar<int32>()();
    OP_REQUIRES_OK(ctx, ctx->input("filename_suffix", &tmp));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tmp->shape()),
                errors::InvalidArgument("filename_suffix must be a scalar"));
    const string filename_suffix = tmp->scalar<tstring>()();

    core::RefCountPtr<SummaryWriterInterface> s;
    OP_REQUIRES_OK(ctx, LookupOrCreateResource<SummaryWriterInterface>(
                            ctx, HandleFromInput(ctx, 0), &s,
                            [max_queue, flush_millis, logdir, filename_suffix,
                             ctx](SummaryWriterInterface** s) {
                              return CreateSummaryFileWriter(
                                  max_queue, flush_millis, logdir,
                                  filename_suffix, ctx->env(), s);
                            }));
  }
};
REGISTER_KERNEL_BUILDER(Name("CreateSummaryFileWriter").Device(DEVICE_CPU),
                        CreateSummaryFileWriterOp);

class CreateSummaryDbWriterOp : public OpKernel {
 public:
  explicit CreateSummaryDbWriterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/kernels/summary_kernels.cc", "CreateSummaryDbWriterOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_3(mht_3_v, 253, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("db_uri", &tmp));
    const string db_uri = tmp->scalar<tstring>()();
    OP_REQUIRES_OK(ctx, ctx->input("experiment_name", &tmp));
    const string experiment_name = tmp->scalar<tstring>()();
    OP_REQUIRES_OK(ctx, ctx->input("run_name", &tmp));
    const string run_name = tmp->scalar<tstring>()();
    OP_REQUIRES_OK(ctx, ctx->input("user_name", &tmp));
    const string user_name = tmp->scalar<tstring>()();

    core::RefCountPtr<SummaryWriterInterface> s;
    OP_REQUIRES_OK(
        ctx,
        LookupOrCreateResource<SummaryWriterInterface>(
            ctx, HandleFromInput(ctx, 0), &s,
            [db_uri, experiment_name, run_name, user_name,
             ctx](SummaryWriterInterface** s) {
              Sqlite* db;
              TF_RETURN_IF_ERROR(Sqlite::Open(
                  db_uri, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, &db));
              core::ScopedUnref unref(db);
              TF_RETURN_IF_ERROR(SetupTensorboardSqliteDb(db));
              TF_RETURN_IF_ERROR(CreateSummaryDbWriter(
                  db, experiment_name, run_name, user_name, ctx->env(), s));
              return Status::OK();
            }));
  }
};
REGISTER_KERNEL_BUILDER(Name("CreateSummaryDbWriter").Device(DEVICE_CPU),
                        CreateSummaryDbWriterOp);

class FlushSummaryWriterOp : public OpKernel {
 public:
  explicit FlushSummaryWriterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_4(mht_4_v, 290, "", "./tensorflow/core/kernels/summary_kernels.cc", "FlushSummaryWriterOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_5(mht_5_v, 295, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    core::RefCountPtr<SummaryWriterInterface> s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    OP_REQUIRES_OK(ctx, s->Flush());
  }
};
REGISTER_KERNEL_BUILDER(Name("FlushSummaryWriter").Device(DEVICE_CPU),
                        FlushSummaryWriterOp);

class CloseSummaryWriterOp : public OpKernel {
 public:
  explicit CloseSummaryWriterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_6(mht_6_v, 309, "", "./tensorflow/core/kernels/summary_kernels.cc", "CloseSummaryWriterOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_7(mht_7_v, 314, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    OP_REQUIRES_OK(ctx, DeleteResource<SummaryWriterInterface>(
                            ctx, HandleFromInput(ctx, 0)));
  }
};
REGISTER_KERNEL_BUILDER(Name("CloseSummaryWriter").Device(DEVICE_CPU),
                        CloseSummaryWriterOp);

class WriteSummaryOp : public OpKernel {
 public:
  explicit WriteSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_8(mht_8_v, 327, "", "./tensorflow/core/kernels/summary_kernels.cc", "WriteSummaryOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_9(mht_9_v, 332, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    core::RefCountPtr<SummaryWriterInterface> s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("step", &tmp));
    const int64_t step = tmp->scalar<int64_t>()();
    OP_REQUIRES_OK(ctx, ctx->input("tag", &tmp));
    const string& tag = tmp->scalar<tstring>()();
    OP_REQUIRES_OK(ctx, ctx->input("summary_metadata", &tmp));
    const string& serialized_metadata = tmp->scalar<tstring>()();

    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("tensor", &t));

    OP_REQUIRES_OK(ctx, s->WriteTensor(step, *t, tag, serialized_metadata));
  }
};
REGISTER_KERNEL_BUILDER(Name("WriteSummary").Device(DEVICE_CPU),
                        WriteSummaryOp);

class WriteRawProtoSummaryOp : public OpKernel {
 public:
  explicit WriteRawProtoSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_10(mht_10_v, 357, "", "./tensorflow/core/kernels/summary_kernels.cc", "WriteRawProtoSummaryOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_11(mht_11_v, 362, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    core::RefCountPtr<SummaryWriterInterface> s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("step", &tmp));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tmp->shape()),
                errors::InvalidArgument("step must be scalar, got shape ",
                                        tmp->shape().DebugString()));
    const int64_t step = tmp->scalar<int64_t>()();
    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("tensor", &t));
    std::unique_ptr<Event> event{new Event};
    event->set_step(step);
    event->set_wall_time(static_cast<double>(ctx->env()->NowMicros()) / 1.0e6);
    // Each Summary proto contains just one repeated field "value" of Value
    // messages with the actual data, so repeated Merge() is equivalent to
    // concatenating all the Value entries together into a single Event.
    const auto summary_pbs = t->flat<tstring>();
    for (int i = 0; i < summary_pbs.size(); ++i) {
      if (!event->mutable_summary()->MergeFromString(summary_pbs(i))) {
        ctx->CtxFailureWithWarning(errors::DataLoss(
            "Bad tf.compat.v1.Summary binary proto tensor string at index ",
            i));
        return;
      }
    }
    OP_REQUIRES_OK(ctx, s->WriteEvent(std::move(event)));
  }
};
REGISTER_KERNEL_BUILDER(Name("WriteRawProtoSummary").Device(DEVICE_CPU),
                        WriteRawProtoSummaryOp);

class ImportEventOp : public OpKernel {
 public:
  explicit ImportEventOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_12(mht_12_v, 399, "", "./tensorflow/core/kernels/summary_kernels.cc", "ImportEventOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_13(mht_13_v, 404, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    core::RefCountPtr<SummaryWriterInterface> s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("event", &t));
    std::unique_ptr<Event> event{new Event};
    if (!ParseProtoUnlimited(event.get(), t->scalar<tstring>()())) {
      ctx->CtxFailureWithWarning(
          errors::DataLoss("Bad tf.Event binary proto tensor string"));
      return;
    }
    OP_REQUIRES_OK(ctx, s->WriteEvent(std::move(event)));
  }
};
REGISTER_KERNEL_BUILDER(Name("ImportEvent").Device(DEVICE_CPU), ImportEventOp);

class WriteScalarSummaryOp : public OpKernel {
 public:
  explicit WriteScalarSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_14(mht_14_v, 425, "", "./tensorflow/core/kernels/summary_kernels.cc", "WriteScalarSummaryOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_15(mht_15_v, 430, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    core::RefCountPtr<SummaryWriterInterface> s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("step", &tmp));
    const int64_t step = tmp->scalar<int64_t>()();
    OP_REQUIRES_OK(ctx, ctx->input("tag", &tmp));
    const string& tag = tmp->scalar<tstring>()();

    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("value", &t));

    OP_REQUIRES_OK(ctx, s->WriteScalar(step, *t, tag));
  }
};
REGISTER_KERNEL_BUILDER(Name("WriteScalarSummary").Device(DEVICE_CPU),
                        WriteScalarSummaryOp);

class WriteHistogramSummaryOp : public OpKernel {
 public:
  explicit WriteHistogramSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_16(mht_16_v, 453, "", "./tensorflow/core/kernels/summary_kernels.cc", "WriteHistogramSummaryOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_17(mht_17_v, 458, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    core::RefCountPtr<SummaryWriterInterface> s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("step", &tmp));
    const int64_t step = tmp->scalar<int64_t>()();
    OP_REQUIRES_OK(ctx, ctx->input("tag", &tmp));
    const string& tag = tmp->scalar<tstring>()();

    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("values", &t));

    OP_REQUIRES_OK(ctx, s->WriteHistogram(step, *t, tag));
  }
};
REGISTER_KERNEL_BUILDER(Name("WriteHistogramSummary").Device(DEVICE_CPU),
                        WriteHistogramSummaryOp);

class WriteImageSummaryOp : public OpKernel {
 public:
  explicit WriteImageSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_18(mht_18_v, 481, "", "./tensorflow/core/kernels/summary_kernels.cc", "WriteImageSummaryOp");

    int64_t max_images_tmp;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_images", &max_images_tmp));
    OP_REQUIRES(ctx, max_images_tmp < (1LL << 31),
                errors::InvalidArgument("max_images must be < 2^31"));
    max_images_ = static_cast<int32>(max_images_tmp);
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_19(mht_19_v, 492, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    core::RefCountPtr<SummaryWriterInterface> s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("step", &tmp));
    const int64_t step = tmp->scalar<int64_t>()();
    OP_REQUIRES_OK(ctx, ctx->input("tag", &tmp));
    const string& tag = tmp->scalar<tstring>()();
    const Tensor* bad_color;
    OP_REQUIRES_OK(ctx, ctx->input("bad_color", &bad_color));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(bad_color->shape()),
        errors::InvalidArgument("bad_color must be a vector, got shape ",
                                bad_color->shape().DebugString()));

    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("tensor", &t));

    OP_REQUIRES_OK(ctx, s->WriteImage(step, *t, tag, max_images_, *bad_color));
  }

 private:
  int32 max_images_;
};
REGISTER_KERNEL_BUILDER(Name("WriteImageSummary").Device(DEVICE_CPU),
                        WriteImageSummaryOp);

class WriteAudioSummaryOp : public OpKernel {
 public:
  explicit WriteAudioSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_20(mht_20_v, 524, "", "./tensorflow/core/kernels/summary_kernels.cc", "WriteAudioSummaryOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_outputs", &max_outputs_));
    OP_REQUIRES(ctx, max_outputs_ > 0,
                errors::InvalidArgument("max_outputs must be > 0"));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_21(mht_21_v, 533, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    core::RefCountPtr<SummaryWriterInterface> s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    const Tensor* tmp;
    OP_REQUIRES_OK(ctx, ctx->input("step", &tmp));
    const int64_t step = tmp->scalar<int64_t>()();
    OP_REQUIRES_OK(ctx, ctx->input("tag", &tmp));
    const string& tag = tmp->scalar<tstring>()();
    OP_REQUIRES_OK(ctx, ctx->input("sample_rate", &tmp));
    const float sample_rate = tmp->scalar<float>()();

    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("tensor", &t));

    OP_REQUIRES_OK(ctx,
                   s->WriteAudio(step, *t, tag, max_outputs_, sample_rate));
  }

 private:
  int max_outputs_;
};
REGISTER_KERNEL_BUILDER(Name("WriteAudioSummary").Device(DEVICE_CPU),
                        WriteAudioSummaryOp);

class WriteGraphSummaryOp : public OpKernel {
 public:
  explicit WriteGraphSummaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_22(mht_22_v, 562, "", "./tensorflow/core/kernels/summary_kernels.cc", "WriteGraphSummaryOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsummary_kernelsDTcc mht_23(mht_23_v, 567, "", "./tensorflow/core/kernels/summary_kernels.cc", "Compute");

    core::RefCountPtr<SummaryWriterInterface> s;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &s));
    const Tensor* t;
    OP_REQUIRES_OK(ctx, ctx->input("step", &t));
    const int64_t step = t->scalar<int64_t>()();
    OP_REQUIRES_OK(ctx, ctx->input("tensor", &t));
    std::unique_ptr<GraphDef> graph{new GraphDef};
    if (!ParseProtoUnlimited(graph.get(), t->scalar<tstring>()())) {
      ctx->CtxFailureWithWarning(
          errors::DataLoss("Bad tf.GraphDef binary proto tensor string"));
      return;
    }
    OP_REQUIRES_OK(ctx, s->WriteGraph(step, std::move(graph)));
  }
};
REGISTER_KERNEL_BUILDER(Name("WriteGraphSummary").Device(DEVICE_CPU),
                        WriteGraphSummaryOp);

}  // namespace tensorflow
