/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_MULTI_TRAINER_CACHE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_MULTI_TRAINER_CACHE_H_
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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cacheDTh {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cacheDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cacheDTh() {
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


#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/logging_utils.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// Sliding-window cache shared across concurrent trainers. Readers call `Get` to
// read elements they haven't read. After a trainer reads an element, it remains
// in the cache and the data is shared with other trainers. This is useful for
// datasets involving expensive computation, and multiple models use the same
// data for training. For example, for hyperparameter tuning.
//
// The cache progresses when a trainer that has consumed all elements in the
// cache requests additional data. It has a bounded size. Elements are garbage
// collected when the cache becomes full. Consequently, trainers read from a
// sliding window through the dataset and may not read the full dataset.
//
// The `MultiTrainerCache` class is thread-safe.
//
// Example usage:
//
//   // `InfiniteRange` returns 1, 2, 3, ... in the `GetNext` calls.
//   class InfiniteRange : public CachableSequence<int64_t> {
//    public:
//     StatusOr<int64_t> GetNext() override {
//       return next_++;
//     }
//
//     size_t GetElementSizeBytes(const int64_t& element) const override {
//       return sizeof(element);
//     }
//
//    private:
//     int64_t next_ = 1;
//   };
//
//   MultiTrainerCache<int64_t> cache(
//       /*max_cache_size_bytes=*/10 * (size_t{1} << 30),  // 10GB
//       absl::make_unique<InfiniteRange>());
//
//   std::shared_ptr<int64_t> next;
//   TF_ASSIGN_OR_RETURN(next, cache.Get("Trainer 1"));  // Returns 1
//   TF_ASSIGN_OR_RETURN(next, cache.Get("Trainer 2"));  // Returns 1
//   TF_ASSIGN_OR_RETURN(next, cache.Get("Trainer 1"));  // Returns 2
//   TF_ASSIGN_OR_RETURN(next, cache.Get("Trainer 2"));  // Returns 2

// To use the cache, the user needs to define a `CachableSequence` to generate
// an infinite sequence of data. It should implement a `GetNext` method to
// produce elements, and a `GetElementSizeBytes` method to estimate the element
// size in bytes.
template <class ElementType>
class CachableSequence {
 public:
  virtual ~CachableSequence() = default;

  // Returns the next element to be cached.
  virtual StatusOr<ElementType> GetNext() = 0;

  // Returns the estimated size of the element in bytes.
  virtual size_t GetElementSizeBytes(const ElementType&) const = 0;
};

// Sliding-window cache shared across concurrent trainers.
template <class ElementType>
class MultiTrainerCache {
 public:
  // Creates a `MultiTrainerCache` with `max_cache_size_bytes` of memory budget.
  // The cache should be able to hold at least one element, i.e.:
  // REQUIRES: max_cache_size_bytes >= max(get_element_size(*))
  explicit MultiTrainerCache(
      size_t max_cache_size_bytes,
      std::unique_ptr<CachableSequence<ElementType>> cachable_sequence);
  virtual ~MultiTrainerCache() = default;

  // Gets the next element for `trainer`. A `trainer_id` identifies the trainer
  // reading from the cache. If one trainer has read data, the data is shared
  // with other trainers.
  StatusOr<std::shared_ptr<const ElementType>> Get(
      const std::string& trainer_id);

  // Cancels the cache with `status` and notifies the readers. After cancelling,
  // all `Get` calls will return `status`.
  // REQUIRES: !status.ok()
  void Cancel(Status status);

  // Returns true if the cache has been cancelled.
  bool IsCancelled() const;

 private:
  // Returns true if element is ready for `trainer_id`. An element is ready if
  // other trainers have read the data and the data remains in the cache. If the
  // data is not ready, one of the trainers need to extend the cache.
  bool IsElementReady(const std::string& trainer_id);

  // Returns the absolute element index relative to the dataset (not relative to
  // the cached elements).
  size_t GetElementIndex(const std::string& trainer_id);

  // Returns the next element for `trainer_id`.
  StatusOr<std::shared_ptr<const ElementType>> GetElement(
      const std::string& trainer_id);

  // Reads a new element and writes it into the cache.
  Status ExtendCache();

  // Frees old elements to keep the cache size below `max_cache_size_bytes_`.
  // `new_element_size_bytes` is the size of the new element being inserted.
  void FreeSpace(size_t new_element_size_bytes);

  // Maximum cache size in bytes.
  const size_t max_cache_size_bytes_;

  // The element sequence over which the sliding window cache operates.
  std::unique_ptr<CachableSequence<ElementType>> cachable_sequence_;

  mutable mutex mu_;
  mutable condition_variable cv_;

  // If `status_` is non-OK, the cache is cancelled, and all method calls will
  // return this status.
  Status status_ TF_GUARDED_BY(mu_) = Status::OK();

  // `cache_` stores the cached elements.
  std::deque<std::shared_ptr<const ElementType>> cache_ TF_GUARDED_BY(mu_);
  size_t cache_size_bytes_ TF_GUARDED_BY(mu_) = 0;
  size_t cache_start_index_ TF_GUARDED_BY(mu_) = 0;

  // True if one thread is extending the cache.
  bool extending_cache_ TF_GUARDED_BY(mu_) = false;

  // Maps trainer IDs to element indices. The indices are absolute indices
  // within the dataset. The actual index to use with `cache_` would be
  // `trainer_to_element_index_map_[trainer_id] - cache_start_index_`.
  absl::flat_hash_map<std::string, size_t> trainer_to_element_index_map_
      TF_GUARDED_BY(mu_);
};

template <class ElementType>
MultiTrainerCache<ElementType>::MultiTrainerCache(
    size_t max_cache_size_bytes,
    std::unique_ptr<CachableSequence<ElementType>> cachable_sequence)
    : max_cache_size_bytes_(max_cache_size_bytes),
      cachable_sequence_(std::move(cachable_sequence)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cacheDTh mht_0(mht_0_v, 344, "", "./tensorflow/core/data/service/multi_trainer_cache.h", "MultiTrainerCache<ElementType>::MultiTrainerCache");

  DCHECK_GT(max_cache_size_bytes, 0)
      << "MultiTrainerCache size must be greater than 0.";
  VLOG(2) << "Initialized tf.data service multi-trainer cache with "
          << FormatBytes(max_cache_size_bytes) << " of memory.";
}

template <class ElementType>
StatusOr<std::shared_ptr<const ElementType>>
MultiTrainerCache<ElementType>::Get(const std::string& trainer_id)
    TF_LOCKS_EXCLUDED(mu_) {
  if (trainer_id.empty()) {
    return errors::Internal(
        "tf.data service multi-trainer cache trainer ID must be non-empty.");
  }

  bool should_extend_cache = false;
  while (true) {
    {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(status_);
      if (IsElementReady(trainer_id)) {
        metrics::RecordTFDataServiceMultiTrainerCacheQuery(
            /*cache_hit=*/!should_extend_cache);
        return GetElement(trainer_id);
      }

      // Extends the cache or waits for another thread to extend the cache. When
      // concurrent trainers wait for the next element, only one of them should
      // extend the cache.
      if (extending_cache_) {
        should_extend_cache = false;
        cv_.wait(l);
      } else {
        should_extend_cache = true;
        extending_cache_ = true;
      }
    }

    if (should_extend_cache) {
      Status s = ExtendCache();
      mutex_lock l(mu_);
      extending_cache_ = false;
      cv_.notify_all();
      TF_RETURN_IF_ERROR(s);
    }
  }
}

template <class ElementType>
bool MultiTrainerCache<ElementType>::IsElementReady(
    const std::string& trainer_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("trainer_id: \"" + trainer_id + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cacheDTh mht_1(mht_1_v, 399, "", "./tensorflow/core/data/service/multi_trainer_cache.h", "MultiTrainerCache<ElementType>::IsElementReady");

  return GetElementIndex(trainer_id) < cache_start_index_ + cache_.size();
}

template <class ElementType>
StatusOr<std::shared_ptr<const ElementType>>
MultiTrainerCache<ElementType>::GetElement(const std::string& trainer_id)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("trainer_id: \"" + trainer_id + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cacheDTh mht_2(mht_2_v, 410, "", "./tensorflow/core/data/service/multi_trainer_cache.h", "MultiTrainerCache<ElementType>::GetElement");

  size_t element_index = GetElementIndex(trainer_id);
  if (element_index >= std::numeric_limits<size_t>::max()) {
    return errors::Internal(
        "tf.data service caching element index exceeds integer limit. Got ",
        element_index);
  }

  std::shared_ptr<const ElementType> result =
      cache_[element_index - cache_start_index_];
  trainer_to_element_index_map_[trainer_id] = element_index + 1;
  return result;
}

template <class ElementType>
size_t MultiTrainerCache<ElementType>::GetElementIndex(
    const std::string& trainer_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("trainer_id: \"" + trainer_id + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cacheDTh mht_3(mht_3_v, 430, "", "./tensorflow/core/data/service/multi_trainer_cache.h", "MultiTrainerCache<ElementType>::GetElementIndex");

  size_t element_index = trainer_to_element_index_map_[trainer_id];
  if (element_index < cache_start_index_) {
    element_index = cache_start_index_;
  }
  return element_index;
}

template <class ElementType>
Status MultiTrainerCache<ElementType>::ExtendCache() TF_LOCKS_EXCLUDED(mu_) {
  TF_ASSIGN_OR_RETURN(ElementType element, cachable_sequence_->GetNext());
  size_t new_element_size_bytes =
      cachable_sequence_->GetElementSizeBytes(element);
  if (new_element_size_bytes > max_cache_size_bytes_) {
    return errors::InvalidArgument(
        "tf.data service element size is larger than cache size in bytes. Got ",
        "element size: ", new_element_size_bytes,
        " and cache size: ", max_cache_size_bytes_);
  }

  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(status_);
  FreeSpace(new_element_size_bytes);
  cache_.push_back(std::make_shared<ElementType>(std::move(element)));
  cache_size_bytes_ += new_element_size_bytes;
  return Status::OK();
}

template <class ElementType>
void MultiTrainerCache<ElementType>::FreeSpace(size_t new_element_size_bytes)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSmulti_trainer_cacheDTh mht_4(mht_4_v, 463, "", "./tensorflow/core/data/service/multi_trainer_cache.h", "MultiTrainerCache<ElementType>::FreeSpace");

  size_t num_elements_discarded = 0;
  while (!cache_.empty() &&
         cache_size_bytes_ + new_element_size_bytes > max_cache_size_bytes_) {
    size_t free_bytes =
        cachable_sequence_->GetElementSizeBytes(*cache_.front());
    cache_.pop_front();
    cache_size_bytes_ -= free_bytes;
    ++cache_start_index_;
    ++num_elements_discarded;
  }

  VLOG(3) << "Freed " << num_elements_discarded << " element(s) from "
          << "tf.data service multi-trainer cache. Memory usage: "
          << FormatBytes(cache_size_bytes_) << ".";
}

template <class ElementType>
void MultiTrainerCache<ElementType>::Cancel(Status status)
    TF_LOCKS_EXCLUDED(mu_) {
  DCHECK(!status.ok())
      << "Cancelling MultiTrainerCache requires a non-OK status. Got "
      << status;
  VLOG(2) << "Cancel tf.data service multi-trainer cache with status "
          << status;
  mutex_lock l(mu_);
  status_ = std::move(status);
  cv_.notify_all();
}

template <class ElementType>
bool MultiTrainerCache<ElementType>::IsCancelled() const
    TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return !status_.ok();
}
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_MULTI_CLIENT_CACHE_H_
