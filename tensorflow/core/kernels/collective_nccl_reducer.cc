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
class MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_reducerDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_reducerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_reducerDTcc() {
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
#include "tensorflow/core/kernels/collective_nccl_reducer.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/common_runtime/collective_util.h"
#include "tensorflow/core/nccl/nccl_manager.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

void NcclReducer::Run(StatusCallback done) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_reducerDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/kernels/collective_nccl_reducer.cc", "NcclReducer::Run");

  Tensor group_size;
  std::unique_ptr<Notification> group_size_ready;
  Status group_size_status;
  std::unique_ptr<Notification> nccl_done;
  if (col_params_->final_op) {
    group_size_ready = absl::make_unique<Notification>();
    // Create an on-device scalar value from group_size_.
    // TODO(ayushd, tucker): avoid this copy by either reusing across
    // invocations or providing the scalar to the kernel in host memory.
    Tensor group_size_val;
    switch (col_ctx_->output->dtype()) {
      case DT_HALF:
        group_size_val =
            Tensor(static_cast<Eigen::half>(col_params_->group.group_size));
        break;
      case DT_FLOAT:
        group_size_val =
            Tensor(static_cast<float>(col_params_->group.group_size));
        break;
      case DT_DOUBLE:
        group_size_val =
            Tensor(static_cast<double>(col_params_->group.group_size));
        break;
      case DT_INT32:
        group_size_val =
            Tensor(static_cast<int32>(col_params_->group.group_size));
        break;
      case DT_INT64:
        group_size_val =
            Tensor(static_cast<int64_t>(col_params_->group.group_size));
        break;
      default:
        done(errors::Internal("Unsupported type ",
                              DataTypeString(col_ctx_->output->dtype())));
        return;
    }
    group_size = Tensor(
        col_ctx_->device->GetAllocator(col_ctx_->op_ctx->input_alloc_attr(0)),
        col_ctx_->output->dtype(), TensorShape({}));
    DeviceContext* op_dev_ctx = col_ctx_->op_ctx->op_device_context();
    // Enqueue copy on gpu stream.
    Notification* copy_note = group_size_ready.get();
    op_dev_ctx->CopyCPUTensorToDevice(
        &group_size_val, col_ctx_->device, &group_size,
        [copy_note, &group_size_status](const Status& s) {
          group_size_status = s;
          copy_note->Notify();
        });
    nccl_done = absl::make_unique<Notification>();
  }

  Status nccl_status;
  // If no final_op, then the NCCL callback is just `done`.  Otherwise we notify
  // `nccl_done` so that we can then perform `final_op`.
  StatusCallback done_callback;
  if (col_params_->final_op) {
    Notification* nccl_note = nccl_done.get();
    done_callback = [nccl_note, &nccl_status](const Status& s) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_reducerDTcc mht_1(mht_1_v, 256, "", "./tensorflow/core/kernels/collective_nccl_reducer.cc", "lambda");

      nccl_status = s;
      nccl_note->Notify();
    };
  } else {
    done_callback = std::move(done);
  }
  // Hold a ref to col_params for the rest of this function.
  col_params_->Ref();
  core::ScopedUnref unref(col_params_);
  col_ctx_->nccl_communicator->Enqueue(col_ctx_, std::move(done_callback));

  // If no final_op, then this OpKernel is non-blocking.
  if (!col_params_->final_op) {
    return;
  }

  // Wait for nccl op and group_size copy to succeed, then do final_op.  This
  // kernel needs to wait for both notifications because they execute on
  // different GPU streams with no ordering guarantees between them.
  // TODO(b/80529858): make this entirely non-blocking by getting rid of the
  // waits below and calling final op from the nccl kernel's DoneCallback.
  {
    profiler::TraceMe activity("Nccl", profiler::TraceMeLevel::kInfo);
    nccl_done->WaitForNotification();
  }
  {
    profiler::TraceMe activity("GroupSizeCopy", profiler::TraceMeLevel::kInfo);
    group_size_ready->WaitForNotification();
  }
  Status final_status =
      group_size_status.ok() ? nccl_status : group_size_status;
  if (final_status.ok()) {
    final_status = collective_util::ComputeBinOp(
        col_ctx_->op_ctx, col_ctx_->op_params, col_ctx_->device,
        col_params_->final_op, col_ctx_->output, &group_size);
  }
  done(final_status);
}

REGISTER_COLLECTIVE(NcclReduce, NcclReducer);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
