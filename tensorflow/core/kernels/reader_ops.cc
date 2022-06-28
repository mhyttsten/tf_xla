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
class MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc() {
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

// See docs in ../ops/io_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/reader_interface.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

class ReaderVerbSyncOpKernel : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/kernels/reader_ops.cc", "Compute");

    ReaderInterface* reader;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "reader_handle", &reader));
    ComputeWithReader(context, reader);
    reader->Unref();
  }

 protected:
  virtual void ComputeWithReader(OpKernelContext* context,
                                 ReaderInterface* reader) = 0;
};

class ReaderVerbAsyncOpKernel : public AsyncOpKernel {
 public:
  using AsyncOpKernel::AsyncOpKernel;

  explicit ReaderVerbAsyncOpKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context),
        thread_pool_(new thread::ThreadPool(
            context->env(), ThreadOptions(),
            strings::StrCat("reader_thread_", SanitizeThreadSuffix(name())),
            1 /* num_threads */, false /* low_latency_hint */)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/kernels/reader_ops.cc", "ReaderVerbAsyncOpKernel");
}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/kernels/reader_ops.cc", "ComputeAsync");

    ReaderInterface* reader;
    OP_REQUIRES_OK_ASYNC(
        context, GetResourceFromContext(context, "reader_handle", &reader),
        done);
    thread_pool_->Schedule([this, context, reader, done]() {
      ComputeWithReader(context, reader);
      reader->Unref();
      done();
    });
  }

 protected:
  virtual void ComputeWithReader(OpKernelContext* context,
                                 ReaderInterface* reader) = 0;

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

class ReaderReadOp : public ReaderVerbAsyncOpKernel {
 public:
  using ReaderVerbAsyncOpKernel::ReaderVerbAsyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc mht_3(mht_3_v, 259, "", "./tensorflow/core/kernels/reader_ops.cc", "ComputeWithReader");

    QueueInterface* queue;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "queue_handle", &queue));
    core::ScopedUnref unref_me(queue);
    Tensor* key = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("key", TensorShape({}), &key));
    Tensor* value = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("value", TensorShape({}), &value));

    auto key_scalar = key->scalar<tstring>();
    auto value_scalar = value->scalar<tstring>();
    tstring key_out, val_out;
    reader->Read(queue, &key_out, &val_out, context);
    key_scalar() = key_out;
    value_scalar() = val_out;
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderRead").Device(DEVICE_CPU), ReaderReadOp);
REGISTER_KERNEL_BUILDER(Name("ReaderReadV2").Device(DEVICE_CPU), ReaderReadOp);

class ReaderReadUpToOp : public ReaderVerbAsyncOpKernel {
 public:
  using ReaderVerbAsyncOpKernel::ReaderVerbAsyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc mht_4(mht_4_v, 291, "", "./tensorflow/core/kernels/reader_ops.cc", "ComputeWithReader");

    QueueInterface* queue;

    const Tensor* num_records_tensor;
    OP_REQUIRES_OK(context, context->input("num_records", &num_records_tensor));
    int64_t num_records = num_records_tensor->scalar<int64_t>()();

    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "queue_handle", &queue));
    core::ScopedUnref unref_me(queue);

    std::vector<tstring> keys_vec;
    keys_vec.reserve(num_records);
    std::vector<tstring> values_vec;
    values_vec.reserve(num_records);

    int64_t num_actually_read =
        reader->ReadUpTo(num_records, queue, &keys_vec, &values_vec, context);

    OP_REQUIRES(context, num_actually_read == keys_vec.size(),
                errors::InvalidArgument("num_actually_read != len(keys_vec"));

    OP_REQUIRES(context, num_actually_read == values_vec.size(),
                errors::InvalidArgument("num_actually_read != len(values_vec"));

    Tensor* keys = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       "keys", TensorShape({num_actually_read}), &keys));

    Tensor* values = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       "values", TensorShape({num_actually_read}), &values));

    auto keys_t = keys->vec<tstring>();
    auto values_t = values->vec<tstring>();
    for (int i = 0; i < num_actually_read; ++i) {
      keys_t(i) = std::move(keys_vec[i]);
      values_t(i) = std::move(values_vec[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderReadUpTo").Device(DEVICE_CPU),
                        ReaderReadUpToOp);
REGISTER_KERNEL_BUILDER(Name("ReaderReadUpToV2").Device(DEVICE_CPU),
                        ReaderReadUpToOp);

class ReaderNumRecordsProducedOp : public ReaderVerbSyncOpKernel {
 public:
  using ReaderVerbSyncOpKernel::ReaderVerbSyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc mht_5(mht_5_v, 348, "", "./tensorflow/core/kernels/reader_ops.cc", "ComputeWithReader");

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("records_produced",
                                                     TensorShape({}), &output));
    output->scalar<int64_t>()() = reader->NumRecordsProduced();
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderNumRecordsProduced").Device(DEVICE_CPU),
                        ReaderNumRecordsProducedOp);
REGISTER_KERNEL_BUILDER(Name("ReaderNumRecordsProducedV2").Device(DEVICE_CPU),
                        ReaderNumRecordsProducedOp);

class ReaderNumWorkUnitsCompletedOp : public ReaderVerbSyncOpKernel {
 public:
  using ReaderVerbSyncOpKernel::ReaderVerbSyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc mht_6(mht_6_v, 369, "", "./tensorflow/core/kernels/reader_ops.cc", "ComputeWithReader");

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("units_completed",
                                                     TensorShape({}), &output));
    output->scalar<int64_t>()() = reader->NumWorkUnitsCompleted();
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderNumWorkUnitsCompleted").Device(DEVICE_CPU),
                        ReaderNumWorkUnitsCompletedOp);
REGISTER_KERNEL_BUILDER(
    Name("ReaderNumWorkUnitsCompletedV2").Device(DEVICE_CPU),
    ReaderNumWorkUnitsCompletedOp);

class ReaderSerializeStateOp : public ReaderVerbSyncOpKernel {
 public:
  using ReaderVerbSyncOpKernel::ReaderVerbSyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc mht_7(mht_7_v, 391, "", "./tensorflow/core/kernels/reader_ops.cc", "ComputeWithReader");

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("state", TensorShape({}), &output));
    OP_REQUIRES_OK(context,
                   reader->SerializeState(&output->scalar<tstring>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderSerializeState").Device(DEVICE_CPU),
                        ReaderSerializeStateOp);
REGISTER_KERNEL_BUILDER(Name("ReaderSerializeStateV2").Device(DEVICE_CPU),
                        ReaderSerializeStateOp);

class ReaderRestoreStateOp : public ReaderVerbSyncOpKernel {
 public:
  using ReaderVerbSyncOpKernel::ReaderVerbSyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc mht_8(mht_8_v, 413, "", "./tensorflow/core/kernels/reader_ops.cc", "ComputeWithReader");

    const Tensor* tensor;
    OP_REQUIRES_OK(context, context->input("state", &tensor));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(tensor->shape()),
        errors::InvalidArgument("Reader state must be scalar, but had shape: ",
                                tensor->shape().DebugString()));
    OP_REQUIRES_OK(context, reader->RestoreState(tensor->scalar<tstring>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderRestoreState").Device(DEVICE_CPU),
                        ReaderRestoreStateOp);
REGISTER_KERNEL_BUILDER(Name("ReaderRestoreStateV2").Device(DEVICE_CPU),
                        ReaderRestoreStateOp);

class ReaderResetOp : public ReaderVerbSyncOpKernel {
 public:
  using ReaderVerbSyncOpKernel::ReaderVerbSyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreader_opsDTcc mht_9(mht_9_v, 437, "", "./tensorflow/core/kernels/reader_ops.cc", "ComputeWithReader");

    OP_REQUIRES_OK(context, reader->Reset());
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderReset").Device(DEVICE_CPU), ReaderResetOp);
REGISTER_KERNEL_BUILDER(Name("ReaderResetV2").Device(DEVICE_CPU),
                        ReaderResetOp);

}  // namespace tensorflow
