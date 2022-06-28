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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc() {
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
#include "tensorflow/core/kernels/data/cache_dataset_ops.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/cache_ops.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level description of
// the following op.

/* static */ constexpr const char* const CacheDatasetOp::kDatasetType;
/* static */ constexpr const char* const CacheDatasetOp::kInputDataset;
/* static */ constexpr const char* const CacheDatasetOp::kFileName;
/* static */ constexpr const char* const CacheDatasetOp::kOutputTypes;
/* static */ constexpr const char* const CacheDatasetOp::kOutputShapes;

namespace {

constexpr char kKeyStrFormat[] = "%%%zuzu_%%%zuzu";
constexpr char kPaddingSizeStrFormat[] = "%zu";
constexpr char kFileDatasetPrefix[] = "File";
constexpr char kMode[] = "Mode";
constexpr char kLockFileSuffix[] = ".lockfile";
constexpr char kIterationCompleted[] = "iteration_completed";
constexpr char kCurIndex[] = "cur_index";
constexpr char kShardId[] = "shard_id";
constexpr char kCreatedAt[] = "Created at";
constexpr char kMemoryDatasetPrefix[] = "Memory";
constexpr char kMemoryCache[] = "MemoryCache";
constexpr char kCacheCompleted[] = "cache_completed";
constexpr char kIndex[] = "index";
constexpr char kImpl[] = "Impl";
constexpr char kCacheDataset[] = "CacheDataset";
constexpr char kIncompleteCacheErrorMessage[] =
    "The calling iterator did not fully read the dataset being cached. In "
    "order to avoid unexpected truncation of the dataset, the partially cached "
    "contents of the dataset  will be discarded. This can happen if you have "
    "an input pipeline similar to `dataset.cache().take(k).repeat()`. You "
    "should use `dataset.take(k).cache().repeat()` instead.";
}  // namespace

class PartialCache {
 public:
  explicit PartialCache(const DatasetBase* dataset) : input_(dataset) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_0(mht_0_v, 246, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "PartialCache");
}

  // Extends the temporary cache up to a given index and then updates
  // out_tensors with the element at that index.
  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_1(mht_1_v, 254, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "Get");

    if (!iter_resource_) {
      TF_ASSIGN_OR_RETURN(iter_resource_,
                          GetIteratorResourceFromDataset(ctx, input_));
      TF_RETURN_IF_ERROR(iter_resource_->SetIteratorFromDataset(ctx, input_));
    }
    if (index >= cache_.size()) {
      TF_RETURN_IF_ERROR(ExtendTempCacheToIndex(index, ctx));
    }
    *out_tensors = cache_.at(index);
    return Status::OK();
  }

  // Returns the data which has been cached up to this point.
  std::vector<std::vector<Tensor>> GetCacheData() { return cache_; }

 private:
  Status ExtendTempCacheToIndex(int64 index, OpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_2(mht_2_v, 274, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "ExtendTempCacheToIndex");

    bool end_of_sequence;
    while (cache_.size() <= index) {
      std::vector<Tensor> out_tensors;
      TF_RETURN_IF_ERROR(
          iter_resource_->GetNext(ctx, &out_tensors, &end_of_sequence));
      if (end_of_sequence) {
        return tensorflow::errors::OutOfRange("Index out of range [0, ",
                                              cache_.size(), "):", index);
      }
      cache_.push_back(out_tensors);
    }
    return Status::OK();
  }

  StatusOr<core::RefCountPtr<IteratorResource>> GetIteratorResourceFromDataset(
      OpKernelContext* ctx, const DatasetBase* dataset) {
    FunctionLibraryRuntime* flr;
    std::unique_ptr<DeviceMgr> device_mgr(nullptr);
    std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
    std::unique_ptr<ProcessFunctionLibraryRuntime> plfr(nullptr);
    TF_RETURN_IF_ERROR(
        ctx->function_library()->Clone(&flib_def, &plfr, &flr, true));

    core::RefCountPtr<IteratorResource> iter_resource(new IteratorResource(
        ctx->env(), dataset->output_dtypes(), dataset->output_shapes(),
        std::move(device_mgr), std::move(flib_def), std::move(plfr), flr));
    return iter_resource;
  }

  const DatasetBase* input_;  // Not owned.
  core::RefCountPtr<IteratorResource> iter_resource_;
  std::vector<std::vector<Tensor>> cache_;
};

class CacheDatasetOp::FileDatasetBase : public DatasetBase {
 public:
  FileDatasetBase(OpKernelContext* ctx, const DatasetBase* input,
                  string filename, Env* env)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        filename_(std::move(filename)),
        env_(env),
        num_tensors_(input->output_dtypes().size()),
        tensor_index_padding_size_(StringPaddingSize(num_tensors_)),
        item_index_padding_size_(StringPaddingSize(kMaxItems)),
        tensor_format_string_(strings::Printf(kKeyStrFormat,
                                              item_index_padding_size_,
                                              tensor_index_padding_size_)) {
    input_->Ref();
    DCHECK_EQ(item_index_padding_size_, 7);
  }

  ~FileDatasetBase() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_3(mht_3_v, 330, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "~FileDatasetBase");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.dataset_prefix = kFileDatasetPrefix;
    return absl::make_unique<FileIterator>(FileIterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_4(mht_4_v, 343, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "output_dtypes");

    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_5(mht_5_v, 350, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "output_shapes");

    return input_->output_shapes();
  }

  string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_6(mht_6_v, 357, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "DebugString");

    name_utils::DatasetDebugStringParams params;
    params.dataset_prefix = kFileDatasetPrefix;
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_7(mht_7_v, 366, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "CardinalityInternal");
 return input_->Cardinality(); }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_8(mht_8_v, 371, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_9(mht_9_v, 379, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

 protected:
  const DatasetBase* const input_;
  const tstring filename_;

 private:
  static size_t StringPaddingSize(size_t num_tensors) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_10(mht_10_v, 391, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "StringPaddingSize");

    return strings::Printf(kPaddingSizeStrFormat, num_tensors - 1).size();
  }

  string FormatName(size_t item_index, size_t tensor_index) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_11(mht_11_v, 398, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "FormatName");

    return strings::Printf(tensor_format_string_.c_str(), item_index,
                           tensor_index);
  }

  class FileIterator : public DatasetIterator<FileDatasetBase> {
   public:
    explicit FileIterator(const Params& params)
        : DatasetIterator<FileDatasetBase>(params) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_12(mht_12_v, 409, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "FileIterator");

      if (params.dataset->env_
              ->FileExists(MetaFilename(params.dataset->filename_))
              .ok()) {
        mode_ = Mode::read;
      } else {
        mode_ = Mode::write;
      }
    }

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_13(mht_13_v, 422, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "Initialize");

      mutex_lock l(mu_);
      return InitializeIterator(ctx);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_14(mht_14_v, 432, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "GetNextInternal");

      mutex_lock l(mu_);
      return iterator_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_15(mht_15_v, 448, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "SaveInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kMode), mode_));
      return SaveInput(ctx, writer, iterator_);
    }
    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_16(mht_16_v, 457, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "RestoreInternal");

      mutex_lock l(mu_);
      {
        int64_t temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kMode), &temp));
        mode_ = static_cast<Mode>(temp);
      }
      if (mode_ == Mode::write &&
          dataset()
              ->env_->FileExists(MetaFilename(dataset()->filename_))
              .ok()) {
        // This could happen if the cache was completely written after the
        // checkpoint was saved.
        LOG(WARNING)
            << "It looks like the cache was already completely written("
            << MetaFilename(dataset()->filename_)
            << ") after the last checkpoint was saved. Attempting to read "
            << "the cache instead of continuing to write. If this is a "
            << "mistake, please remove the above file and try running again.";
        mode_ = Mode::read;
      }
      TF_RETURN_IF_ERROR(InitializeIterator(ctx));
      return RestoreInput(ctx, reader, iterator_);
    }

   private:
    // FileWriterIterator passes through and caches items from the input
    // FileDatasetBase.
    //
    // This iterator is used when the cache directory is not found on disk. It
    // creates the cache directory, and passes on the underlying iterator's
    // elements.
    //
    // Caching is performed by writing the input tensors to disk using the
    // `BundleWriter`. Note that the cache gets fully flushed to disk only
    // after the input iterator has been fully exhausted. If the program
    // exits, before completion of an epoch, the cached state would be lost.
    // To ensure that the partial cache persists across sessions, one should
    // checkpoint the input pipeline. On each call to `SaveInternal` the
    // partial cache gets flushed to disk in files with prefix
    // <filename>_<shard_id> where shard_id is unique for each checkpoint.
    // When all elements have been produced, these shards get coalesced.
    class FileWriterIterator : public DatasetIterator<FileDatasetBase> {
     public:
      explicit FileWriterIterator(const Params& params)
          : DatasetIterator<FileDatasetBase>(params),
            cur_index_(0),
            shard_id_(0),
            filename_(
                strings::StrCat(params.dataset->filename_, "_", shard_id_)),
            lockfile_(strings::StrCat(filename_, kLockFileSuffix)),
            lockfile_created_(false),
            iteration_completed_(false) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_17(mht_17_v, 512, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "FileWriterIterator");
}

      ~FileWriterIterator() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_18(mht_18_v, 517, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "~FileWriterIterator");

        if (!dataset()->env_->FileExists(MetaFilename(filename_)).ok()) {
          LOG(WARNING) << kIncompleteCacheErrorMessage;
          std::vector<string> cache_files;
          Status s = dataset()->env_->GetMatchingPaths(
              strings::StrCat(filename_, "*"), &cache_files);
          if (!s.ok()) {
            LOG(WARNING) << "Failed to get matching files on " << filename_
                         << "* : " << s.ToString();
          }
          for (const string& path : cache_files) {
            s = dataset()->env_->DeleteFile(path);
            if (!s.ok()) {
              LOG(WARNING) << "Failed to delete " << path << " : "
                           << s.ToString();
            }
          }
        }
      }

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_19(mht_19_v, 540, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "Initialize");

        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_20(mht_20_v, 550, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "GetNextInternal");

        mutex_lock l(mu_);
        *end_of_sequence = false;
        TF_RETURN_IF_ERROR(EnsureLockFileExists(end_of_sequence));
        if (*end_of_sequence) {
          return Status::OK();
        }
        TF_RETURN_IF_ERROR(writer_->status());
        if (cur_index_ >= kMaxItems) {
          // As a courtesy, close the [truncated] cache file.
          Status s = Finish();
          if (!s.ok()) {
            LOG(ERROR) << s;
          }
          return errors::InvalidArgument(
              "Upstream iterator is producing more than ", kMaxItems,
              " items, which is more than the cache limit.");
        }

        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        if (*end_of_sequence && out_tensors->empty()) {
          TF_RETURN_IF_ERROR(Finish());
          cur_index_++;
          return Status::OK();
        }
        if (out_tensors->size() != dataset()->num_tensors_) {
          return errors::Internal(
              "Upstream iterator returned invalid number of tensors. "
              "Expected ",
              dataset()->num_tensors_, " got: ", out_tensors->size());
        }
        size_t tensor_index = 0;
        for (const Tensor& t : *out_tensors) {
          DCHECK_LT(tensor_index, dataset()->num_tensors_);
          string key = dataset()->FormatName(cur_index_, tensor_index++);
          TF_RETURN_IF_ERROR(writer_->Add(key, t));
        }
        if (*end_of_sequence) {
          TF_RETURN_IF_ERROR(Finish());
        }
        cur_index_++;
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_21(mht_21_v, 606, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "SaveInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kCurIndex), cur_index_));

        if (iteration_completed_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kIterationCompleted), ""));
          return Status::OK();
        }

        // lockfile is created on the first call to GetNextInternal. The
        // absence of a lockfile means that GetNextInternal was not called
        // and hence nothing was written to cache. So we don't need to worry
        // about flushing the current shard. This ensures that we never write
        // empty shards.
        if (lockfile_created_) {
          // Flush the current bundle.
          TF_RETURN_IF_ERROR(writer_->Finish());

          // Note: We do not delete the lockfile here. We keep lockfiles of
          // all shards around until the entire cache has been written to
          // prevent concurrent iterators from corrupting any of the shards.

          // Start caching to a new shard.
          shard_id_++;
          filename_ = strings::StrCat(dataset()->filename_, "_", shard_id_);
          lockfile_ = strings::StrCat(filename_, kLockFileSuffix);
          lockfile_created_ = false;
        }
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kShardId), shard_id_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_22(mht_22_v, 645, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "RestoreInternal");

        mutex_lock l(mu_);
        int64_t temp;
        // TODO(b/78048575): Update this when saving size_t tensors directly
        // is supported.
        {
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurIndex), &temp));
          cur_index_ = static_cast<size_t>(temp);
          if (cur_index_ != temp) {
            return errors::Internal("Invalid value for cur_index ", temp);
          }
        }

        if (reader->Contains(full_name(kIterationCompleted))) {
          iteration_completed_ = true;
          return Status::OK();
        }

        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));

        // TODO(b/78048575): Update this when saving size_t tensors directly
        // is supported.
        {
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kShardId), &temp));
          shard_id_ = static_cast<size_t>(temp);
          if (shard_id_ != temp) {
            return errors::Internal("Invalid value for shard_id ", temp);
          }
        }
        filename_ = strings::StrCat(dataset()->filename_, "_", shard_id_);
        lockfile_ = strings::StrCat(filename_, kLockFileSuffix);
        writer_ = absl::make_unique<BundleWriter>(dataset()->env_, filename_);
        return Status::OK();
      }

     private:
      Status EnsureLockFileExists(bool* end_of_sequence)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_23(mht_23_v, 685, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "EnsureLockFileExists");

        if (iteration_completed_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        if (lockfile_created_) {
          return Status::OK();
        }

        // Perform rudimentary locking to help catch concurrent writes to the
        // same cache files.

        // 1. Check that a checkpoint for the shard has not already been
        // written.
        if (dataset()->env_->FileExists(MetaFilename(filename_)).ok()) {
          return errors::AlreadyExists("Existing cache files found: \n",
                                       MetaFilename(filename_), "\n",
                                       DataFilename(filename_, 0, 1), "\n",
                                       "To continue delete the above files.");
        }

        // 2. Check that there isn't a concurrent iterator that is writing
        // to cache.
        if (dataset()->env_->FileExists(lockfile_).ok()) {
          // Attempt to read the contents of the lockfile.
          char contents_scratch[151] = {0};  // Initialize all to 0.
          StringPiece contents;
          std::unique_ptr<RandomAccessFile> file;
          if (dataset()->env_->NewRandomAccessFile(lockfile_, &file).ok()) {
            file->Read(0, 150, &contents, contents_scratch).IgnoreError();
          }
          return errors::AlreadyExists(
              "There appears to be a concurrent caching iterator running - "
              "cache lockfile already exists ('",
              lockfile_,
              "'). If you are sure no other running TF computations are "
              "using this cache prefix, delete the lockfile and "
              "re-initialize the iterator. Lockfile contents: ",
              contents);
        }
        // Create the file, and write some basic contents.
        std::unique_ptr<WritableFile> lockfile;
        TF_RETURN_IF_ERROR(
            dataset()->env_->NewWritableFile(lockfile_, &lockfile));
        TF_RETURN_IF_ERROR(lockfile->Append(
            strings::StrCat(kCreatedAt, ": ", EnvTime::NowSeconds())));

        // At this point we know that
        // 1. There is no conflicting checkpoint with prefix `filename_`.
        // 2. There is no concurrent session that is trying to write a ckpt
        //    to filename.
        // So it is safe to create a BundleWriter here. Note that it is
        // unsafe to initialize the BundleWriter anywhere the above
        // conditions are not met since BundleWriter's constructor creates
        // new temp files which can delete the temp files created by a
        // BundleWriter in another Session.
        writer_ = absl::make_unique<BundleWriter>(dataset()->env_, filename_);
        lockfile_created_ = true;
        return Status::OK();
      }

      Status Finish() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_24(mht_24_v, 749, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "Finish");

        iteration_completed_ = true;
        // Flush the current bundle.
        TF_RETURN_IF_ERROR(writer_->Finish());
        // Merge all the bundles.
        // Currently there are `shard_id_ + 1` bundles, one for each
        // checkpoint. Each bundle has prefix <filename>_<id> where `id` is an
        // integer starting at 0 and incremented by 1 for each new checkpoint.
        // We merge all these bundles into a bundle with prefix <filename> so
        // that the next call to `MakeIterator` can build a
        // `FileReaderIterator`.
        {
          std::vector<tstring> prefixes;
          prefixes.reserve(shard_id_ + 1);
          for (size_t i = 0; i <= shard_id_; ++i) {
            prefixes.emplace_back(
                strings::StrCat(dataset()->filename_, "_", i));
          }
          TF_RETURN_IF_ERROR(
              MergeBundles(dataset()->env_, prefixes, dataset()->filename_));
        }
        // Delete all lockfiles.
        for (size_t i = 0; i <= shard_id_; ++i) {
          TF_RETURN_IF_ERROR(dataset()->env_->DeleteFile(
              strings::StrCat(dataset()->filename_, "_", i, kLockFileSuffix)));
        }
        return Status::OK();
      }

      mutex mu_;
      size_t cur_index_ TF_GUARDED_BY(mu_);
      // Index of the current shard. This gets incremented whenever a new
      // cache shard is saved.
      size_t shard_id_ TF_GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      // The current prefix for the cache file. This is equal to
      // `StrCat(dataset()->filename_, "_", shard_id_)`.
      string filename_;
      std::unique_ptr<BundleWriter> writer_ TF_GUARDED_BY(mu_);
      string lockfile_ TF_GUARDED_BY(mu_);
      bool lockfile_created_ TF_GUARDED_BY(mu_);
      bool iteration_completed_ TF_GUARDED_BY(mu_);
    };  // FileWriterIterator

    class FileReaderIterator : public DatasetIterator<FileDatasetBase> {
     public:
      explicit FileReaderIterator(const Params& params)
          : DatasetIterator<FileDatasetBase>(params),
            cur_index_(0),
            reader_(dataset()->env_, dataset()->filename_),
            iterator_restored_(false) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_25(mht_25_v, 802, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "FileReaderIterator");
}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_26(mht_26_v, 809, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "GetNextInternal");

        mutex_lock l(mu_);
        *end_of_sequence = false;
        TF_RETURN_IF_ERROR(reader_.status());
        if (!reader_.Valid()) {
          *end_of_sequence = true;
          return Status::OK();
        }
        out_tensors->clear();
        out_tensors->resize(dataset()->num_tensors_);

        for (size_t i = 0; i < dataset()->num_tensors_; ++i) {
          // When the iterator is restored from the checkpoint, `reader_` is
          // already pointing at `key` so we do not need to skip the header
          // entry.
          if (!iterator_restored_) {
            reader_.Next();  // The first entry in the table is a header.
          } else {
            iterator_restored_ = false;
          }
          if (!reader_.Valid()) {
            out_tensors->clear();
            *end_of_sequence = true;
            return Status::OK();
          }
          StringPiece key = reader_.key();
          DCHECK_EQ(key, dataset()->FormatName(cur_index_, i));
          TF_RETURN_IF_ERROR(reader_.ReadCurrent(&(*out_tensors)[i]));
          TF_RETURN_IF_ERROR(reader_.status());
        }
        cur_index_++;
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_27(mht_27_v, 854, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "SaveInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kCurIndex), cur_index_));
        return Status::OK();
      }

      Status RestoreInternal(
          IteratorContext* ctx,
          IteratorStateReader* iterator_state_reader) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_28(mht_28_v, 866, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "RestoreInternal");

        mutex_lock l(mu_);
        {
          // TODO(b/78048575): Update this when saving size_t tensors directly
          // is supported.
          int64_t temp;
          TF_RETURN_IF_ERROR(
              iterator_state_reader->ReadScalar(full_name(kCurIndex), &temp));
          cur_index_ = static_cast<size_t>(temp);
          if (cur_index_ != temp) {
            return errors::Internal("Invalid value for cur_index ", temp);
          }
        }
        if (!reader_.Valid()) {
          return errors::Internal("Error initializing BundleReader.");
        }
        reader_.Seek(dataset()->FormatName(cur_index_, 0));
        iterator_restored_ = true;
        return Status::OK();
      }

     private:
      mutex mu_;
      size_t cur_index_ TF_GUARDED_BY(mu_);
      BundleReader reader_ TF_GUARDED_BY(mu_);
      bool iterator_restored_ TF_GUARDED_BY(mu_);
    };  // FileReaderIterator

    Status InitializeIterator(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_29(mht_29_v, 898, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "InitializeIterator");

      // We intentionally use the same prefix for both `FileReaderIterator` and
      // `FileWriterIterator`. Since at any time there will be at most one of
      // them alive, there should be no conflicts. This allows both iterators to
      // use a common key for `cur_index`. We leverage this in the corner case
      // when this iterator is restored from an old checkpoint in `write` mode
      // and the cache has been completely flushed to disk since then. In that
      // case we simply build a `FileReaderIterator` and seek to the
      // `cur_index`.
      switch (mode_) {
        case Mode::read:
          iterator_ =
              absl::make_unique<FileReaderIterator>(FileReaderIterator::Params{
                  dataset(), strings::StrCat(prefix(), kImpl)});
          break;
        case Mode::write:
          iterator_ =
              absl::make_unique<FileWriterIterator>(FileWriterIterator::Params{
                  dataset(), strings::StrCat(prefix(), kImpl)});
      }
      TF_RETURN_IF_ERROR(iterator_->InitializeBase(ctx, this));
      return iterator_->Initialize(ctx);
    }

    mutex mu_;
    enum Mode { read, write };
    Mode mode_ TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> iterator_ TF_GUARDED_BY(mu_);
  };  // FileIterator

  Env* const env_;
  const size_t num_tensors_;
  const size_t tensor_index_padding_size_;
  static constexpr size_t kMaxItems = 10000000;  // 10 million
  const size_t item_index_padding_size_;
  const string tensor_format_string_;
};  // FileDatasetBase

class CacheDatasetOp::FileDataset : public CacheDatasetOp::FileDatasetBase {
 public:
  using FileDatasetBase::FileDatasetBase;

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_30(mht_30_v, 946, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "AsGraphDefInternal");

    Node* input_graph = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph));
    Node* filename = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(filename_, &filename));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph, filename}, output));
    return Status::OK();
  }
};

class CacheDatasetOp::FileDatasetV2 : public CacheDatasetOp::FileDatasetBase {
 public:
  explicit FileDatasetV2(OpKernelContext* ctx, const DatasetBase* input,
                         string filename, Env* env,
                         const Tensor& resource_handle)
      : FileDatasetBase(ctx, input, filename, env),
        resource_handle_(resource_handle) {}

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_31(mht_31_v, 970, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "AsGraphDefInternal");

    Node* input_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* filename_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(filename_, &filename_node));
    Node* resource_handle_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddTensor(resource_handle_, &resource_handle_node));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_node, filename_node, resource_handle_node}, output));
    return Status::OK();
  }

 private:
  const Tensor resource_handle_;
};

class CacheDatasetOp::MemoryDatasetBase : public DatasetBase {
 public:
  explicit MemoryDatasetBase(OpKernelContext* ctx, const DatasetBase* input,
                             std::shared_ptr<MemoryCache> cache)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        cache_(std::move(cache)) {
    input_->Ref();
  }

  ~MemoryDatasetBase() override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_32(mht_32_v, 999, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "~MemoryDatasetBase");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.dataset_prefix = kMemoryDatasetPrefix;
    return absl::make_unique<MemoryIterator>(
        MemoryIterator::Params{
            this, name_utils::IteratorPrefix(kDatasetType, prefix, params)},
        cache_.get());
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_33(mht_33_v, 1014, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "output_dtypes");

    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_34(mht_34_v, 1021, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "output_shapes");

    return input_->output_shapes();
  }

  string DebugString() const override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_35(mht_35_v, 1028, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "DebugString");

    name_utils::DatasetDebugStringParams params;
    params.dataset_prefix = kMemoryDatasetPrefix;
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_36(mht_36_v, 1037, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "CardinalityInternal");

    return input_->Cardinality();
  };

  int64_t CardinalityInternal(CardinalityOptions options) const override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_37(mht_37_v, 1044, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "CardinalityInternal");

    return input_->Cardinality(options);
  };

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_38(mht_38_v, 1052, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "Get");

    mutex_lock l(mu_);

    CardinalityOptions options;
    options.set_compute_level(CardinalityOptions::CARDINALITY_COMPUTE_LOW);
    int64_t cardinality = Cardinality(options);

    if (cardinality != kUnknownCardinality &&
        cardinality != kInfiniteCardinality && index >= cardinality) {
      return errors::OutOfRange("Index out of range [0, ", cardinality,
                                "):", index);
    }
    if (!partial_cache_) {
      partial_cache_ = absl::make_unique<PartialCache>(input_);
    }
    return partial_cache_->Get(ctx, index, out_tensors);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_39(mht_39_v, 1073, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_40(mht_40_v, 1081, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

 protected:
  class MemoryIterator : public DatasetIterator<MemoryDatasetBase> {
   public:
    explicit MemoryIterator(const Params& params, MemoryCache* cache)
        : DatasetIterator<MemoryDatasetBase>(params), cache_(cache) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_41(mht_41_v, 1092, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "MemoryIterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_42(mht_42_v, 1097, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "Initialize");

      mutex_lock l(mu_);
      return InitializeIterator(ctx);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_43(mht_43_v, 1107, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "GetNextInternal");

      mutex_lock l(mu_);
      return iterator_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_44(mht_44_v, 1123, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "SaveInternal");

      mutex_lock l(mu_);
      if (cache_->IsCompleted()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCacheCompleted), ""));
        TF_RETURN_IF_ERROR(
            WriteElementsToCheckpoint(writer, prefix(), cache_->data()));
      }
      return SaveInput(ctx, writer, iterator_);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_45(mht_45_v, 1137, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "RestoreInternal");

      mutex_lock l(mu_);
      iterator_.reset();
      cache_->Reset();
      if (reader->Contains(full_name(kCacheCompleted))) {
        std::vector<std::vector<Tensor>> temp_cache;
        TF_RETURN_IF_ERROR(
            ReadElementsFromCheckpoint(ctx, reader, prefix(), &temp_cache));
        cache_->Complete(std::move(temp_cache));
      }
      TF_RETURN_IF_ERROR(InitializeIterator(ctx));
      return RestoreInput(ctx, reader, iterator_);
    }

   private:
    class MemoryWriterIterator : public DatasetIterator<MemoryDatasetBase> {
     public:
      explicit MemoryWriterIterator(const Params& params, MemoryCache* cache)
          : DatasetIterator<MemoryDatasetBase>(params), cache_(cache) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_46(mht_46_v, 1158, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "MemoryWriterIterator");
}

      ~MemoryWriterIterator() override {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_47(mht_47_v, 1163, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "~MemoryWriterIterator");

        mutex_lock l(mu_);
        if (!temp_cache_.empty() && !cache_->IsCompleted()) {
          LOG(WARNING) << kIncompleteCacheErrorMessage;
          cache_->Reset();
        }
      }

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_48(mht_48_v, 1174, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "Initialize");

        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_49(mht_49_v, 1184, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "GetNextInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        if (*end_of_sequence) {
          if (!cache_->IsCompleted()) {
            VLOG(2) << "Finalizing the cache because EOF has been reached.";
            cache_->Complete(std::move(temp_cache_));
          }
          return Status::OK();
        }
        RecordBufferEnqueue(ctx, *out_tensors);
        temp_cache_.emplace_back(*out_tensors);
        if (temp_cache_.size() == dataset()->input_->Cardinality()) {
          VLOG(2) << "Finalizing the cache because its size matches the "
                     "expected input cardinality.";
          cache_->Complete(std::move(temp_cache_));
        }
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_50(mht_50_v, 1216, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "SaveInternal");

        mutex_lock l(mu_);
        if (!cache_->IsCompleted()) {
          TF_RETURN_IF_ERROR(
              WriteElementsToCheckpoint(writer, prefix(), temp_cache_));
        }
        return SaveInput(ctx, writer, input_impl_);
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_51(mht_51_v, 1229, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "RestoreInternal");

        mutex_lock l(mu_);
        if (!reader->Contains(full_name(kCacheCompleted))) {
          TF_RETURN_IF_ERROR(
              ReadElementsFromCheckpoint(ctx, reader, prefix(), &temp_cache_));
        }
        return RestoreInput(ctx, reader, input_impl_);
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      MemoryCache* const cache_ TF_GUARDED_BY(mu_);  // not owned.
      std::vector<std::vector<Tensor>> temp_cache_ TF_GUARDED_BY(mu_);
    };  // MemoryWriterIterator

    class MemoryReaderIterator : public DatasetIterator<MemoryDatasetBase> {
     public:
      explicit MemoryReaderIterator(const Params& params, MemoryCache* cache)
          : DatasetIterator<MemoryDatasetBase>(params),
            cache_(cache),
            index_(0) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_52(mht_52_v, 1253, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "MemoryReaderIterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_53(mht_53_v, 1258, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "Initialize");

        // The memory allocated for the cache is owned by the parent
        // dataset but performance modeling uses the iterator abstraction and
        // thus we record the memory allocated for the cache here. The caveat
        // is that this is incorrect if there are concurrent instances of this
        // iterator.
        tf_shared_lock l(mu_);
        for (size_t i = 0; i < cache_->size(); ++i) {
          RecordBufferEnqueue(ctx, cache_->at(i));
        }
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_54(mht_54_v, 1276, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "GetNextInternal");

        mutex_lock l(mu_);
        if (index_ < cache_->size()) {
          const std::vector<Tensor>& cache_tensors = cache_->at(index_);
          out_tensors->insert(out_tensors->begin(), cache_tensors.begin(),
                              cache_tensors.end());
          index_++;
          *end_of_sequence = false;
          return Status::OK();
        } else {
          *end_of_sequence = true;
          return Status::OK();
        }
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_55(mht_55_v, 1302, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "SaveInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kIndex), index_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_56(mht_56_v, 1312, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "RestoreInternal");

        mutex_lock l(mu_);
        {
          // kIndex will not be set if we are restoring from a checkpoint
          // written by a MemoryWriterIterator that has completed its cache.
          int64_t temp = cache_->size();
          if (reader->Contains(full_name(kIndex))) {
            TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kIndex), &temp));
          }
          index_ = static_cast<size_t>(temp);
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      MemoryCache* const cache_ TF_GUARDED_BY(mu_);  // not owned.
      size_t index_ TF_GUARDED_BY(mu_);
    };  // MemoryReaderIterator

    Status InitializeIterator(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_57(mht_57_v, 1336, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "InitializeIterator");

      if (cache_->IsCompleted()) {
        iterator_ = absl::make_unique<MemoryReaderIterator>(
            MemoryReaderIterator::Params{dataset(),
                                         strings::StrCat(prefix(), kImpl)},
            cache_);
      } else {
        iterator_ = absl::make_unique<MemoryWriterIterator>(
            MemoryWriterIterator::Params{dataset(),
                                         strings::StrCat(prefix(), kImpl)},
            cache_);
      }
      TF_RETURN_IF_ERROR(iterator_->InitializeBase(ctx, this));
      return iterator_->Initialize(ctx);
    }

    mutex mu_;
    MemoryCache* cache_ TF_GUARDED_BY(mu_);  // not owned.
    std::unique_ptr<IteratorBase> iterator_ TF_GUARDED_BY(mu_);
  };  // MemoryIterator

  mutable mutex mu_;
  const DatasetBase* const input_;
  const std::shared_ptr<MemoryCache> cache_;
  mutable std::unique_ptr<PartialCache> partial_cache_ TF_GUARDED_BY(mu_);
};  // MemoryDatasetBase

// This version of memory dataset has an exclusive ownership of the memory cache
// resource. It supports sharing of the cache across different iterations of the
// `repeat` transformation but not across different iterators.
class CacheDatasetOp::MemoryDataset : public CacheDatasetOp::MemoryDatasetBase {
 public:
  MemoryDataset(OpKernelContext* ctx, const DatasetBase* input,
                MemoryCacheManager* manager, ResourceHandle&& resource_handle)
      : MemoryDatasetBase(ctx, input, manager->get()),
        manager_(manager),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()) {}

  ~MemoryDataset() override {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_58(mht_58_v, 1378, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "~MemoryDataset");

    manager_->Unref();
    Status s = resource_mgr_->Delete<MemoryCacheManager>(
        resource_handle_.container(), resource_handle_.name());
    if (!s.ok()) {
      LOG(WARNING) << "Failed to delete cache resource: " << s.ToString();
    }
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_59(mht_59_v, 1393, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "AsGraphDefInternal");

    Node* input_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* filename_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(tstring(""), &filename_node));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_node, filename_node}, output));
    return Status::OK();
  }

 private:
  MemoryCacheManager* const manager_;  // Owned.
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
};

// This version of memory dataset has a shared ownership of the memory cache
// resource. It supports sharing of the cache across different iterations of
// the `repeat` transformation and also across different iterators.
class CacheDatasetOp::MemoryDatasetV2
    : public CacheDatasetOp::MemoryDatasetBase {
 public:
  MemoryDatasetV2(OpKernelContext* ctx, const DatasetBase* input,
                  MemoryCacheManager* manager, ResourceHandle&& resource_handle,
                  bool owns_resource)
      : MemoryDatasetBase(ctx, input, manager->get()),
        manager_(manager),
        owns_resource_(owns_resource),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()) {}

  ~MemoryDatasetV2() override {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_60(mht_60_v, 1427, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "~MemoryDatasetV2");

    manager_->Unref();
    if (owns_resource_) {
      Status s = resource_mgr_->Delete<MemoryCacheManager>(
          resource_handle_.container(), resource_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete cache resource: " << s.ToString();
      }
    }
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_61(mht_61_v, 1444, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "AsGraphDefInternal");

    Node* input_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* filename_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(tstring(""), &filename_node));
    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = resource_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_node, filename_node, resource_handle_node}, output));
    return Status::OK();
  }

 private:
  MemoryCacheManager* const manager_;  // Owned.
  const bool owns_resource_;
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
};

CacheDatasetOp::CacheDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx),
      op_version_(ctx->def().op() == kCacheDataset ? 1 : 2) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_62(mht_62_v, 1470, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "CacheDatasetOp::CacheDatasetOp");
}

void CacheDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_opsDTcc mht_63(mht_63_v, 1476, "", "./tensorflow/core/kernels/data/cache_dataset_ops.cc", "CacheDatasetOp::MakeDataset");

  // Parse out the filenames tensor.
  tstring filename;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, kFileName, &filename));
  if (filename.empty()) {
    static std::atomic<int64_t> resource_id_counter(0);
    const string& container = ctx->resource_manager()->default_container();
    auto name = strings::StrCat(ctx->op_kernel().name(), "/", kMemoryCache, "_",
                                resource_id_counter.fetch_add(1));
    if (op_version_ == 2) {
      bool owns_resource = false;
      MemoryCacheManager* manager = nullptr;
      auto handle = HandleFromInput(ctx, 2);
      Status s = ctx->resource_manager()->Lookup<MemoryCacheManager>(
          handle.container(), handle.name(), &manager);
      if (errors::IsNotFound(s)) {
        owns_resource = true;
        OP_REQUIRES_OK(
            ctx,
            ctx->resource_manager()->LookupOrCreate<MemoryCacheManager>(
                container, name, &manager, [](MemoryCacheManager** manager) {
                  *manager = new MemoryCacheManager();
                  return Status::OK();
                }));
        handle = MakeResourceHandle<MemoryCacheManager>(ctx, container, name);
      } else {
        OP_REQUIRES_OK(ctx, s);
      }
      // Ownership of manager is transferred onto `MemoryDatasetV2`.
      *output = new MemoryDatasetV2(ctx, input, manager, std::move(handle),
                                    owns_resource);
    } else {
      MemoryCacheManager* manager;
      OP_REQUIRES_OK(
          ctx, ctx->resource_manager()->LookupOrCreate<MemoryCacheManager>(
                   container, name, &manager, [](MemoryCacheManager** manager) {
                     *manager = new MemoryCacheManager();
                     return Status::OK();
                   }));
      auto handle =
          MakeResourceHandle<MemoryCacheManager>(ctx, container, name);
      // Ownership of manager is transferred onto `MemoryDataset`.
      *output = new MemoryDataset(ctx, input, manager, std::move(handle));
    }
  } else {
    if (op_version_ == 2) {
      *output =
          new FileDatasetV2(ctx, input, filename, ctx->env(), ctx->input(2));
    } else {
      *output = new FileDataset(ctx, input, filename, ctx->env());
    }
  }
}

namespace {
REGISTER_KERNEL_BUILDER(Name("CacheDataset").Device(DEVICE_CPU),
                        CacheDatasetOp);
REGISTER_KERNEL_BUILDER(Name("CacheDatasetV2").Device(DEVICE_CPU),
                        CacheDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
