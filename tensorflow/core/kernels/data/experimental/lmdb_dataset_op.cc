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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/lmdb_dataset_op.h"

#include <sys/stat.h>

#include "lmdb.h"  // NOLINT(build/include)
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const LMDBDatasetOp::kDatasetType;
/* static */ constexpr const char* const LMDBDatasetOp::kFileNames;
/* static */ constexpr const char* const LMDBDatasetOp::kOutputTypes;
/* static */ constexpr const char* const LMDBDatasetOp::kOutputShapes;

class LMDBDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const std::vector<string>& filenames)
      : DatasetBase(DatasetContext(ctx)), filenames_(filenames) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::LMDB")});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "output_dtypes");

    static DataTypeVector* dtypes = new DataTypeVector({DT_STRING, DT_STRING});
    return *dtypes;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "output_shapes");

    static std::vector<PartialTensorShape>* shapes =
        new std::vector<PartialTensorShape>({{}, {}});
    return *shapes;
  }

  string DebugString() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "DebugString");
 return "LMDBDatasetOp::Dataset"; }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_3(mht_3_v, 235, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "InputDatasets");

    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_4(mht_4_v, 242, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "CheckExternalState");
 return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_5(mht_5_v, 250, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "AsGraphDefInternal");

    Node* filenames = nullptr;
    TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {filenames}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_6(mht_6_v, 264, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "Iterator");
}

    ~Iterator() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_7(mht_7_v, 269, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "~Iterator");

      // Close any open database connections.
      ResetStreamsLocked();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_8(mht_8_v, 279, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "GetNextInternal");

      mutex_lock l(mu_);
      do {
        if (mdb_cursor_) {
          out_tensors->emplace_back(ctx->allocator({}), DT_STRING,
                                    TensorShape({}));
          Tensor& key_tensor = out_tensors->back();
          key_tensor.scalar<tstring>()() = string(
              static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);

          out_tensors->emplace_back(ctx->allocator({}), DT_STRING,
                                    TensorShape({}));
          Tensor& value_tensor = out_tensors->back();
          value_tensor.scalar<tstring>()() = string(
              static_cast<const char*>(mdb_value_.mv_data), mdb_value_.mv_size);

          int val;
          val = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT);
          if (val != MDB_SUCCESS && val != MDB_NOTFOUND) {
            return errors::InvalidArgument(mdb_strerror(val));
          }
          if (val == MDB_NOTFOUND) {
            ResetStreamsLocked();
            ++current_file_index_;
          }
          *end_of_sequence = false;
          return Status::OK();
        }
        if (current_file_index_ == dataset()->filenames_.size()) {
          *end_of_sequence = true;
          ResetStreamsLocked();
          return Status::OK();
        }

        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      } while (true);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_9(mht_9_v, 327, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "SaveInternal");

      return errors::Unimplemented(
          "Checkpointing is currently not supported for LMDBDataset.");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_10(mht_10_v, 336, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "RestoreInternal");

      return errors::Unimplemented(
          "Checkpointing is currently not supported for LMDBDataset.");
    }

   private:
    Status SetupStreamsLocked(Env* env) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_11(mht_11_v, 345, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "SetupStreamsLocked");

      if (current_file_index_ >= dataset()->filenames_.size()) {
        return errors::InvalidArgument(
            "current_file_index_:", current_file_index_,
            " >= filenames_.size():", dataset()->filenames_.size());
      }
      const string& filename = dataset()->filenames_[current_file_index_];

      int val = mdb_env_create(&mdb_env_);
      if (val != MDB_SUCCESS) {
        return errors::InvalidArgument(mdb_strerror(val));
      }
      int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;

      struct stat source_stat;
      if (stat(filename.c_str(), &source_stat) == 0 &&
          (source_stat.st_mode & S_IFREG)) {
        flags |= MDB_NOSUBDIR;
      }
      val = mdb_env_open(mdb_env_, filename.c_str(), flags, 0664);
      if (val != MDB_SUCCESS) {
        return errors::InvalidArgument(mdb_strerror(val));
      }
      val = mdb_txn_begin(mdb_env_, nullptr, MDB_RDONLY, &mdb_txn_);
      if (val != MDB_SUCCESS) {
        return errors::InvalidArgument(mdb_strerror(val));
      }
      val = mdb_dbi_open(mdb_txn_, nullptr, 0, &mdb_dbi_);
      if (val != MDB_SUCCESS) {
        return errors::InvalidArgument(mdb_strerror(val));
      }
      val = mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_);
      if (val != MDB_SUCCESS) {
        return errors::InvalidArgument(mdb_strerror(val));
      }
      val = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
      if (val != MDB_SUCCESS && val != MDB_NOTFOUND) {
        return errors::InvalidArgument(mdb_strerror(val));
      }
      if (val == MDB_NOTFOUND) {
        ResetStreamsLocked();
      }
      return Status::OK();
    }
    void ResetStreamsLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_12(mht_12_v, 392, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "ResetStreamsLocked");

      if (mdb_env_ != nullptr) {
        if (mdb_cursor_) {
          mdb_cursor_close(mdb_cursor_);
          mdb_cursor_ = nullptr;
        }
        mdb_dbi_close(mdb_env_, mdb_dbi_);
        mdb_txn_abort(mdb_txn_);
        mdb_env_close(mdb_env_);
        mdb_txn_ = nullptr;
        mdb_dbi_ = 0;
        mdb_env_ = nullptr;
      }
    }
    mutex mu_;
    size_t current_file_index_ TF_GUARDED_BY(mu_) = 0;
    MDB_env* mdb_env_ TF_GUARDED_BY(mu_) = nullptr;
    MDB_txn* mdb_txn_ TF_GUARDED_BY(mu_) = nullptr;
    MDB_dbi mdb_dbi_ TF_GUARDED_BY(mu_) = 0;
    MDB_cursor* mdb_cursor_ TF_GUARDED_BY(mu_) = nullptr;

    MDB_val mdb_key_ TF_GUARDED_BY(mu_);
    MDB_val mdb_value_ TF_GUARDED_BY(mu_);
  };

  const std::vector<string> filenames_;
};

void LMDBDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_opDTcc mht_13(mht_13_v, 423, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op.cc", "LMDBDatasetOp::MakeDataset");

  const Tensor* filenames_tensor;
  OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
  OP_REQUIRES(
      ctx, filenames_tensor->dims() <= 1,
      errors::InvalidArgument("`filenames` must be a scalar or a vector."));

  std::vector<string> filenames;
  filenames.reserve(filenames_tensor->NumElements());
  for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
    filenames.push_back(filenames_tensor->flat<tstring>()(i));
  }

  *output = new Dataset(ctx, filenames);
}

namespace {

REGISTER_KERNEL_BUILDER(Name("LMDBDataset").Device(DEVICE_CPU), LMDBDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalLMDBDataset").Device(DEVICE_CPU),
                        LMDBDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
