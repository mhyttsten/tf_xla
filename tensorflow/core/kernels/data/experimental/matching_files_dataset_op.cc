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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc() {
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
#include <queue>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class MatchingFilesDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "MakeDataset");

    const Tensor* patterns_t;
    OP_REQUIRES_OK(ctx, ctx->input("patterns", &patterns_t));
    const auto patterns = patterns_t->flat<tstring>();
    size_t num_patterns = static_cast<size_t>(patterns.size());
    std::vector<tstring> pattern_strs;
    pattern_strs.reserve(num_patterns);

    for (size_t i = 0; i < num_patterns; i++) {
      pattern_strs.push_back(patterns(i));
    }

    *output = new Dataset(ctx, std::move(pattern_strs));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<tstring> patterns)
        : DatasetBase(DatasetContext(ctx)), patterns_(std::move(patterns)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "Dataset");
}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::MatchingFiles")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "output_dtypes");

      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_3(mht_3_v, 253, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "output_shapes");

      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "DebugString");

      return "MatchingFilesDatasetOp::Dataset";
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_5(mht_5_v, 270, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "InputDatasets");

      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_6(mht_6_v, 277, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "CheckExternalState");
 return Status::OK(); }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_7(mht_7_v, 285, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "AsGraphDefInternal");

      Node* patterns_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(patterns_, &patterns_node));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {patterns_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_8(mht_8_v, 299, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "Iterator");
}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_9(mht_9_v, 306, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "GetNextInternal");

        mutex_lock l(mu_);
        FileSystem* fs;

        TF_RETURN_IF_ERROR(ctx->env()->GetFileSystemForFile(
            dataset()->patterns_[(current_pattern_index_ > 0)
                                     ? current_pattern_index_ - 1
                                     : 0],
            &fs));

        while (!filepath_queue_.empty() ||
               current_pattern_index_ < dataset()->patterns_.size()) {
          // All the elements in the heap will be the matched filenames or the
          // potential directories.
          if (!filepath_queue_.empty()) {
            PathStatus current_path = filepath_queue_.top();
            filepath_queue_.pop();

            if (!current_path.second) {
              Tensor filepath_tensor(ctx->allocator({}), DT_STRING, {});

              // Replace the forward slash with the backslash for Windows path
              if (isWindows_) {
                std::replace(current_path.first.begin(),
                             current_path.first.end(), '/', '\\');
              }

              filepath_tensor.scalar<tstring>()() =
                  std::move(current_path.first);
              out_tensors->emplace_back(std::move(filepath_tensor));
              *end_of_sequence = false;
              hasMatch_ = true;
              return Status::OK();
            }

            // In this case, current_path is a directory. Then continue the
            // search.
            TF_RETURN_IF_ERROR(
                UpdateIterator(ctx, fs, current_path.first, current_pattern_));
          } else {
            // search a new pattern
            current_pattern_ = dataset()->patterns_[current_pattern_index_];
            StringPiece current_pattern_view = StringPiece(current_pattern_);

            // Windows paths contain backslashes and Windows APIs accept forward
            // and backslashes equivalently, so we convert the pattern to use
            // forward slashes exclusively. The backslash is used as the
            // indicator of Windows paths. Note that this is not ideal, since
            // the API expects backslash as an escape character, but no code
            // appears to rely on this behavior
            if (current_pattern_view.find('\\') != std::string::npos) {
              isWindows_ = true;
              std::replace(&current_pattern_[0],
                           &current_pattern_[0] + current_pattern_.size(), '\\',
                           '/');
            } else {
              isWindows_ = false;
            }

            StringPiece fixed_prefix = current_pattern_view.substr(
                0, current_pattern_view.find_first_of("*?[\\"));
            string current_dir(io::Dirname(fixed_prefix));

            // If current_dir is empty then we need to fix up fixed_prefix and
            // current_pattern_ to include . as the top level directory.
            if (current_dir.empty()) {
              current_dir = ".";
              current_pattern_ = io::JoinPath(current_dir, current_pattern_);
            }

            TF_RETURN_IF_ERROR(
                UpdateIterator(ctx, fs, current_dir, current_pattern_));
            ++current_pattern_index_;
          }
        }

        *end_of_sequence = true;
        if (hasMatch_) {
          return Status::OK();
        } else {
          return errors::NotFound("Don't find any matched files");
        }
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeSourceNode(std::move(args));
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_10(mht_10_v, 400, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "SaveInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name("current_pattern_index"), current_pattern_index_));

        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_pattern"),
                                               current_pattern_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("hasMatch"), hasMatch_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("isWindows"), isWindows_));

        if (!filepath_queue_.empty()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("queue_size"),
                                                 filepath_queue_.size()));
          int i = 0;
          while (!filepath_queue_.empty()) {
            TF_RETURN_IF_ERROR(
                writer->WriteScalar(full_name(strings::StrCat("path_", i)),
                                    filepath_queue_.top().first));
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("path_status_", i)),
                filepath_queue_.top().second));
            filepath_queue_.pop();
            i++;
          }
        }

        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_11(mht_11_v, 435, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        int64_t current_pattern_index;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name("current_pattern_index"), &current_pattern_index));
        current_pattern_index_ = size_t(current_pattern_index);

        tstring current_pattern_tstr;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_pattern"),
                                              &current_pattern_tstr));
        current_pattern_ = current_pattern_tstr;

        int64_t hasMatch;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("hasMatch"), &hasMatch));
        hasMatch_ = static_cast<bool>(hasMatch);

        int64_t isWindows;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("isWindows"), &isWindows));
        isWindows_ = static_cast<bool>(isWindows);

        if (reader->Contains(full_name("queue_size"))) {
          int64_t queue_size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("queue_size"), &queue_size));
          for (int i = 0; i < queue_size; i++) {
            tstring path;
            int64_t path_status;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("path_", i)), &path));
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("path_status_", i)), &path_status));
            filepath_queue_.push(
                PathStatus(path, static_cast<bool>(path_status)));
          }
        }

        return Status::OK();
      }

     private:
      Status UpdateIterator(IteratorContext* ctx, FileSystem* fs,
                            const string& dir, const string& eval_pattern)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("dir: \"" + dir + "\"");
   mht_12_v.push_back("eval_pattern: \"" + eval_pattern + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_12(mht_12_v, 484, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "UpdateIterator");

        StringPiece fixed_prefix =
            StringPiece(eval_pattern)
                .substr(0, eval_pattern.find_first_of("*?[\\"));

        filepath_queue_.push(PathStatus(dir, true));
        Status ret;  // Status to return

        // DFS to find the first element in the iterator.
        while (!filepath_queue_.empty()) {
          const PathStatus current_path = filepath_queue_.top();

          // All the files in the heap are matched with the pattern, so finish
          // the search if current_path is a file.
          if (!current_path.second) {
            return Status::OK();
          }

          filepath_queue_.pop();

          // If current_path is a directory, search its children.
          const string& current_dir = current_path.first;
          std::vector<string> children;
          ret.Update(fs->GetChildren(current_dir, &children));

          // Handle the error cases: 1) continue the search if the status is
          // NOT_FOUND; 2) return the non-ok status immediately if it is not
          // NOT_FOUND.
          if (ret.code() == error::NOT_FOUND) {
            continue;
          } else if (!ret.ok()) {
            return ret;
          }

          // children_dir_status holds is_dir status for children. It can have
          // three possible values: OK for true; FAILED_PRECONDITION for false;
          // CANCELLED if we don't calculate IsDirectory (we might do that
          // because there isn't any point in exploring that child path).
          std::vector<Status> children_dir_status;
          children_dir_status.resize(children.size());

          // This IsDirectory call can be expensive for some FS. Parallelizing
          // it.
          auto is_directory_fn = [fs, current_dir, &children, &fixed_prefix,
                                  &children_dir_status](int i) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSmatching_files_dataset_opDTcc mht_13(mht_13_v, 531, "", "./tensorflow/core/kernels/data/experimental/matching_files_dataset_op.cc", "lambda");

            const string child_path = io::JoinPath(current_dir, children[i]);
            // In case the child_path doesn't start with the fixed_prefix, then
            // we don't need to explore this path.
            if (!absl::StartsWith(child_path, fixed_prefix)) {
              children_dir_status[i] =
                  errors::Cancelled("Operation not needed");
            } else {
              children_dir_status[i] = fs->IsDirectory(child_path);
            }
          };

          BlockingCounter counter(children.size());
          for (int i = 0; i < children.size(); i++) {
            (*ctx->runner())([&is_directory_fn, &counter, i] {
              is_directory_fn(i);
              counter.DecrementCount();
            });
          }
          counter.Wait();

          for (int i = 0; i < children.size(); i++) {
            const string& child_dir_path =
                io::JoinPath(current_dir, children[i]);
            const Status& child_dir_status = children_dir_status[i];

            // If the IsDirectory call was cancelled we bail.
            if (child_dir_status.code() == tensorflow::error::CANCELLED) {
              continue;
            }

            if (child_dir_status.ok()) {
              // push the child dir for next search
              filepath_queue_.push(PathStatus(child_dir_path, true));
            } else {
              // This case will be a file: if the file matches the pattern, push
              // it to the heap; otherwise, ignore it.
              if (ctx->env()->MatchPath(child_dir_path, eval_pattern)) {
                filepath_queue_.push(PathStatus(child_dir_path, false));
              }
            }
          }
        }
        return ret;
      }

      mutex mu_;
      // True means the path is a directory; False means the path is a filename.
      typedef std::pair<string, bool> PathStatus;
      std::priority_queue<PathStatus, std::vector<PathStatus>,
                          std::greater<PathStatus>>
          filepath_queue_ TF_GUARDED_BY(mu_);
      size_t current_pattern_index_ TF_GUARDED_BY(mu_) = 0;
      tstring current_pattern_ TF_GUARDED_BY(mu_);
      bool hasMatch_ TF_GUARDED_BY(mu_) = false;
      bool isWindows_ TF_GUARDED_BY(mu_) = false;
    };

    const std::vector<tstring> patterns_;
  };
};

REGISTER_KERNEL_BUILDER(Name("MatchingFilesDataset").Device(DEVICE_CPU),
                        MatchingFilesDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalMatchingFilesDataset").Device(DEVICE_CPU),
    MatchingFilesDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
