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
class MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_init_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_init_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_init_opDTcc() {
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
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/lookup_table_init_op.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

using InitializerSerializer =
    lookup::InitializableLookupTable::InitializerSerializer;

// Kernel to initialize a look table given a key and value tensors.
// After this operation, the table becomes read-only.
class InitializeTableOp : public OpKernel {
 public:
  explicit InitializeTableOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_init_opDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/kernels/lookup_table_init_op.cc", "InitializeTableOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_init_opDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/kernels/lookup_table_init_op.cc", "Compute");

    mutex_lock l(mu_);
    lookup::InitializableLookupTable* table;
    OP_REQUIRES_OK(ctx,
                   GetInitializableLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& keys = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(keys.shape()),
        errors::InvalidArgument("Keys must be a vector, but received shape",
                                keys.shape().DebugString()));

    const Tensor& values = ctx->input(2);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(values.shape()),
        errors::InvalidArgument("Values must be a vector, but received shape",
                                values.shape().DebugString()));

    OP_REQUIRES(ctx, keys.NumElements() == values.NumElements(),
                errors::InvalidArgument(
                    "Keys and values must have the same size ",
                    keys.NumElements(), " vs ", values.NumElements()));

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }

 private:
  mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("InitializeTable").Device(DEVICE_CPU),
                        InitializeTableOp);
REGISTER_KERNEL_BUILDER(Name("InitializeTableV2").Device(DEVICE_CPU),
                        InitializeTableOp);

// Kernel to initialize a lookup table from a text file.
//
// After this operation, the table becomes read-only.
class InitializeTableFromTextFileOp : public OpKernel {
 public:
  explicit InitializeTableFromTextFileOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_init_opDTcc mht_2(mht_2_v, 282, "", "./tensorflow/core/kernels/lookup_table_init_op.cc", "InitializeTableFromTextFileOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_size", &vocab_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key_index", &key_index_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value_index", &value_index_));
    if (ctx->HasAttr("offset")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("offset", &offset_));
    }
    string delimiter;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("delimiter", &delimiter));
    OP_REQUIRES(ctx, delimiter.size() == 1,
                errors::InvalidArgument("delimiter should be only 1 char"));
    delimiter_ = delimiter[0];
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlookup_table_init_opDTcc mht_3(mht_3_v, 299, "", "./tensorflow/core/kernels/lookup_table_init_op.cc", "Compute");

    mutex_lock l(mu_);
    lookup::InitializableLookupTable* table;
    OP_REQUIRES_OK(ctx,
                   GetInitializableLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, DT_STRING};
    DataTypeVector expected_outputs = {};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& vocab_filename_tensor = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(vocab_filename_tensor.shape()),
        errors::InvalidArgument("filename should be a single string, but got ",
                                vocab_filename_tensor.shape().DebugString()));

    const string& vocab_filename = vocab_filename_tensor.scalar<tstring>()();
    OP_REQUIRES(ctx, !vocab_filename.empty(),
                errors::InvalidArgument("filename cannot be empty."));

    int64_t memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(
        ctx, lookup::InitializeTableFromTextFile(
                 vocab_filename, vocab_size_, delimiter_, key_index_,
                 value_index_, offset_, ctx->env(),
                 MakeInitializerSerializer(vocab_filename_tensor), table));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }

 private:
  std::unique_ptr<InitializerSerializer> MakeInitializerSerializer(
      Tensor vocab_filename) {
    return absl::make_unique<InitializerSerializer>(
        [vocab_filename, vocab_size = vocab_size_, delimiter = delimiter_,
         key_index = key_index_, value_index = value_index_,
         offset = offset_](GraphDefBuilder* builder, Node* table, Node** out) {
          Node* vocab_filename_node = ops::SourceOp(
              "Const", builder->opts()
                           .WithAttr("dtype", vocab_filename.dtype())
                           .WithAttr("value", vocab_filename));
          std::string delimiter_string(1, delimiter);
          Node* import_table = ops::BinaryOp(
              "InitializeTableFromTextFileV2", table, vocab_filename_node,
              builder->opts()
                  .WithAttr("vocab_size", vocab_size)
                  .WithAttr("key_index", key_index)
                  .WithAttr("value_index", value_index)
                  .WithAttr("offset", offset)
                  .WithAttr("delimiter", delimiter_string));
          *out = ops::UnaryOp("Identity", table,
                              builder->opts().WithControlInput(import_table));
          return Status::OK();
        });
  }

  mutex mu_;
  int64_t vocab_size_;
  char delimiter_;
  int64_t key_index_;
  int64_t value_index_;
  int64_t offset_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(InitializeTableFromTextFileOp);
};

REGISTER_KERNEL_BUILDER(Name("InitializeTableFromTextFile").Device(DEVICE_CPU),
                        InitializeTableFromTextFileOp);
REGISTER_KERNEL_BUILDER(
    Name("InitializeTableFromTextFileV2").Device(DEVICE_CPU),
    InitializeTableFromTextFileOp);
}  // namespace tensorflow
