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
class MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/split_utils.h"

#include <functional>
#include <string>
#include <utility>

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace {
constexpr char kNumToSkip[] = "num_to_skip";
constexpr char kSplitProvider[] = "split_provider";
constexpr char kSlash[] = "/";
constexpr char kIndex[] = "index";
}  // namespace

IndexSplitProvider::IndexSplitProvider(int64_t n) : i_(0), n_(n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/data/split_utils.cc", "IndexSplitProvider::IndexSplitProvider");
}

Status IndexSplitProvider::GetNext(Tensor* split, bool* end_of_splits) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/data/split_utils.cc", "IndexSplitProvider::GetNext");

  mutex_lock l(mu_);
  if (i_ >= n_) {
    *end_of_splits = true;
    return Status::OK();
  }
  *end_of_splits = false;
  *split = Tensor(DT_INT64, TensorShape{});
  split->scalar<int64_t>()() = i_++;
  return Status::OK();
}

Status IndexSplitProvider::Reset() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc mht_2(mht_2_v, 221, "", "./tensorflow/core/data/split_utils.cc", "IndexSplitProvider::Reset");

  mutex_lock l(mu_);
  i_ = 0;
  return Status::OK();
}

Status IndexSplitProvider::Save(
    std::function<std::string(std::string)> full_name,
    IteratorStateWriter* writer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc mht_3(mht_3_v, 232, "", "./tensorflow/core/data/split_utils.cc", "IndexSplitProvider::Save");

  mutex_lock l(mu_);
  return writer->WriteScalar(full_name(kIndex), i_);
}

Status IndexSplitProvider::Restore(
    std::function<std::string(std::string)> full_name,
    IteratorStateReader* reader) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc mht_4(mht_4_v, 242, "", "./tensorflow/core/data/split_utils.cc", "IndexSplitProvider::Restore");

  mutex_lock l(mu_);
  return reader->ReadScalar(full_name(kIndex), &i_);
}

ShardingSplitProvider::ShardingSplitProvider(
    int64_t num_shards, int64_t shard_index,
    std::shared_ptr<SplitProvider> split_provider)
    : num_shards_(num_shards),
      shard_index_(shard_index),
      split_provider_(split_provider),
      num_to_skip_(shard_index_) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc mht_5(mht_5_v, 256, "", "./tensorflow/core/data/split_utils.cc", "ShardingSplitProvider::ShardingSplitProvider");
}

Status ShardingSplitProvider::GetNext(Tensor* split, bool* end_of_splits) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc mht_6(mht_6_v, 261, "", "./tensorflow/core/data/split_utils.cc", "ShardingSplitProvider::GetNext");

  mutex_lock l(mu_);
  while (num_to_skip_ > 0) {
    TF_RETURN_IF_ERROR(split_provider_->GetNext(split, end_of_splits));
    if (*end_of_splits) {
      return Status::OK();
    }
    num_to_skip_--;
  }
  num_to_skip_ = num_shards_ - 1;
  TF_RETURN_IF_ERROR(split_provider_->GetNext(split, end_of_splits));
  return Status::OK();
}

Status ShardingSplitProvider::Reset() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc mht_7(mht_7_v, 278, "", "./tensorflow/core/data/split_utils.cc", "ShardingSplitProvider::Reset");

  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(split_provider_->Reset());
  num_to_skip_ = shard_index_;
  return Status::OK();
}

Status ShardingSplitProvider::Save(
    std::function<std::string(std::string)> full_name,
    IteratorStateWriter* writer) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc mht_8(mht_8_v, 290, "", "./tensorflow/core/data/split_utils.cc", "ShardingSplitProvider::Save");

  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(split_provider_->Save(
      [&](const std::string& key) {
        return full_name(absl::StrCat(kSplitProvider, kSlash, key));
      },
      writer));
  return writer->WriteScalar(full_name(kNumToSkip), num_to_skip_);
}

Status ShardingSplitProvider::Restore(
    std::function<std::string(std::string)> full_name,
    IteratorStateReader* reader) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSsplit_utilsDTcc mht_9(mht_9_v, 305, "", "./tensorflow/core/data/split_utils.cc", "ShardingSplitProvider::Restore");

  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(split_provider_->Restore(
      [&](const std::string& key) {
        return full_name(absl::StrCat(kSplitProvider, kSlash, key));
      },
      reader));
  TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNumToSkip), &num_to_skip_));
  return Status::OK();
}

StatusOr<std::shared_ptr<SplitProvider>> GetSingleSplitProvider(
    IteratorContext* ctx, const DatasetBase* dataset) {
  if (ctx->split_providers().size() != 1) {
    return errors::FailedPrecondition(
        "Failed to get single split provider for dataset ",
        dataset->DebugString(), ". Found ", ctx->split_providers().size(),
        " split providers");
  }
  return ctx->split_providers()[0];
}

StatusOr<std::vector<std::unique_ptr<SplitProvider>>> GetSplitProviders(
    const DatasetBase* dataset) {
  std::vector<std::unique_ptr<SplitProvider>> result;
  std::vector<const DatasetBase*> inputs;
  TF_RETURN_IF_ERROR(dataset->InputDatasets(&inputs));
  for (const auto& input : inputs) {
    std::vector<std::unique_ptr<SplitProvider>> providers;
    TF_RETURN_IF_ERROR(input->MakeSplitProviders(&providers));
    for (auto& provider : providers) {
      result.push_back(std::move(provider));
    }
  }
  return result;
}

StatusOr<std::vector<IteratorContext>> CreateInputIteratorContexts(
    IteratorContext* ctx, const DatasetBase* dataset) {
  std::vector<const DatasetBase*> inputs;
  TF_RETURN_IF_ERROR(dataset->InputDatasets(&inputs));
  std::vector<IteratorContext> result;
  if (ctx->split_providers().empty()) {
    for (int i = 0; i < inputs.size(); ++i) {
      result.emplace_back(ctx);
    }
    return result;
  }
  int64_t num_sources = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->num_sources() < 0) {
      return errors::FailedPrecondition(
          "Failed to determine the number of sources for dataset of type ",
          inputs[i]->type_string());
    }
    num_sources += inputs[i]->num_sources();
  }
  if (num_sources != ctx->split_providers().size()) {
    return errors::FailedPrecondition(
        "Attempted to feed ", ctx->split_providers().size(),
        " split providers into a dataset with ", num_sources, " sources");
  }
  int64_t split_provider_index = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    IteratorContext::Params params(ctx);
    params.split_providers.clear();
    for (int j = 0; j < inputs[i]->num_sources(); ++j) {
      params.split_providers.push_back(
          ctx->split_providers()[split_provider_index + j]);
    }
    split_provider_index += inputs[i]->num_sources();
    result.emplace_back(std::move(params));
  }
  return result;
}

}  // namespace data
}  // namespace tensorflow
