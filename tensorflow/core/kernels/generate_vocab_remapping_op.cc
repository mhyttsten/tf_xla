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
class MHTracer_DTPStensorflowPScorePSkernelsPSgenerate_vocab_remapping_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSgenerate_vocab_remapping_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSgenerate_vocab_remapping_opDTcc() {
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

#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/lookup_table_init_op.h"
#include "tensorflow/core/kernels/lookup_table_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {
// lookup::InitializeTableFromTextFile requires a delimiter even though we use
// the entire line for vocabularies.
constexpr char kUnusedLookupDelim = '\t';
}  // namespace

// This Op generates a vocab remapping Tensor from an old and new vocabulary
// file that maps new ID's to old ID's.
class GenerateVocabRemappingOp : public OpKernel {
 public:
  explicit GenerateVocabRemappingOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgenerate_vocab_remapping_opDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/kernels/generate_vocab_remapping_op.cc", "GenerateVocabRemappingOp");

    OP_REQUIRES_OK(context,
                   context->GetAttr("new_vocab_offset", &new_vocab_offset_));
    OP_REQUIRES_OK(context, context->GetAttr("num_new_vocab", &num_new_vocab_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("old_vocab_size", &old_vocab_size_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgenerate_vocab_remapping_opDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/generate_vocab_remapping_op.cc", "Compute");

    const Tensor* new_vocab_file_tensor;
    OP_REQUIRES_OK(context,
                   context->input("new_vocab_file", &new_vocab_file_tensor));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(new_vocab_file_tensor->shape()),
                errors::InvalidArgument(
                    "new_vocab_file should be a single string, but got ",
                    new_vocab_file_tensor->shape().DebugString()));

    // Build a new ID->token lookup table.
    const string& new_vocab_filename =
        new_vocab_file_tensor->scalar<tstring>()();
    OP_REQUIRES(context, !new_vocab_filename.empty(),
                errors::InvalidArgument("new vocab filename cannot be empty."));
    lookup::HashTable<int64_t, tstring>* new_vocab_table =
        new lookup::HashTable<int64_t, tstring>(context, this);
    core::ScopedUnref unref_new(new_vocab_table);
    // Note: we pass -1 (unknown) for vocab_size, which is supposed to be the
    // total elements in file.  This is different from num_new_vocab_, which
    // accounts for partitioning.
    OP_REQUIRES_OK(context, lookup::InitializeTableFromTextFile(
                                new_vocab_filename,
                                -1,  // vocab_size
                                kUnusedLookupDelim,
                                -1,  // key_index, use the line number.
                                -2,  // value_index, use the whole line/token.
                                0,   // No offset.
                                context->env(), new_vocab_table));
    OP_REQUIRES(context,
                new_vocab_offset_ + num_new_vocab_ <= new_vocab_table->size(),
                errors::InvalidArgument("lookup table size must be larger than "
                                        "last new vocab entry's line"));

    const Tensor* old_vocab_file_tensor;
    OP_REQUIRES_OK(context,
                   context->input("old_vocab_file", &old_vocab_file_tensor));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(old_vocab_file_tensor->shape()),
                errors::InvalidArgument(
                    "old_vocab_file should be a single string, but got ",
                    old_vocab_file_tensor->shape().DebugString()));
    // Build a token->old ID lookup table.
    const string& old_vocab_filename =
        old_vocab_file_tensor->scalar<tstring>()();
    OP_REQUIRES(context, !old_vocab_filename.empty(),
                errors::InvalidArgument("new vocab filename cannot be empty."));
    lookup::HashTable<tstring, int64_t>* old_vocab_table =
        new lookup::HashTable<tstring, int64_t>(context, this);
    core::ScopedUnref unref_old(old_vocab_table);
    // Note: If old_vocab_size_ is -1 (unknown), we retrieve all elements in
    // file (see TextFileLineIterator).
    OP_REQUIRES_OK(context,
                   lookup::InitializeTableFromTextFile(
                       old_vocab_filename, old_vocab_size_, kUnusedLookupDelim,
                       -2,  // key_index, use the whole line/token.
                       -1,  // value_index, use the line number.
                       0,   // No offset.
                       context->env(), old_vocab_table));

    // Fill out new_ids = [new_vocab_offset, new_vocab_offset + 1, ...,
    //                     new_vocab_offset + num_new_vocab_]
    // The double look-up requires a few temporary Tensors.
    Tensor new_ids;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DT_INT64, TensorShape({num_new_vocab_}),
                                        &new_ids));
    auto new_ids_vec = new_ids.vec<int64_t>();
    // Note that we should always be able to find tokens for all new ID's, given
    // that the lookup table is constructed with the vocabulary file itself
    // (see the check on offset and table size post-initialization).
    Tensor default_token;
    OP_REQUIRES_OK(
        context, context->allocate_temp(
                     DT_STRING, TensorShape({num_new_vocab_}), &default_token));
    auto default_token_vec = default_token.vec<tstring>();
    default_token_vec.setConstant("" /* NOT_FOUND_TOKEN */);

    Tensor default_id;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DT_INT64, TensorShape({num_new_vocab_}),
                                        &default_id));
    auto default_id_vec = default_id.vec<int64_t>();
    default_id_vec.setConstant(-1 /* NOT_FOUND_ID */);

    for (int i = 0; i < num_new_vocab_; ++i) {
      new_ids_vec(i) = static_cast<int64_t>(i + new_vocab_offset_);
    }
    Tensor tokens;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DT_STRING, TensorShape({num_new_vocab_}), &tokens));
    Tensor* remapping;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       "remapping", TensorShape({num_new_vocab_}), &remapping));
    // In the corner case where num_new_vocab_ is 0 (we are dealing with an
    // OOV-only partition), we should not do this lookup.
    if (num_new_vocab_ != 0) {
      OP_REQUIRES_OK(context, new_vocab_table->Find(context, new_ids, &tokens,
                                                    default_token));
      OP_REQUIRES_OK(context, old_vocab_table->Find(context, tokens, remapping,
                                                    default_id));
    }
    // Iterate through remapping to calculate num_present.
    const auto remapping_vec = remapping->vec<int64_t>();
    int num_present = 0;
    for (int i = 0; i < num_new_vocab_; ++i) {
      if (remapping_vec(i) != -1 /* NOT_FOUND_ID */) {
        ++num_present;
      }
    }
    Tensor* num_present_t;
    OP_REQUIRES_OK(context,
                   context->allocate_output("num_present", TensorShape({}),
                                            &num_present_t));
    num_present_t->scalar<int>()() = num_present;
  }

 private:
  int new_vocab_offset_;
  int num_new_vocab_;
  int old_vocab_size_;
};

REGISTER_KERNEL_BUILDER(Name("GenerateVocabRemapping").Device(DEVICE_CPU),
                        GenerateVocabRemappingOp);

}  // namespace tensorflow
