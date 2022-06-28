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
class MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc() {
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

#include <cstddef>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace {

// Partial Ordering Comparator for Tensor keys containing scalar int64's
struct KeyTensorLess {
  bool operator()(const Tensor& lhs, const Tensor& rhs) const {
    return std::less<int64_t>{}(lhs.scalar<int64_t>()(),
                                rhs.scalar<int64_t>()());
  }
};

// Key Equality operator for Tensor keys containing scalar int64's
struct KeyTensorEqual {
  bool operator()(const Tensor& lhs, const Tensor& rhs) const {
    return std::equal_to<int64_t>{}(lhs.scalar<int64_t>()(),
                                    rhs.scalar<int64_t>()());
  }
};

// Hash for Tensor keys containing scalar int64's
struct KeyTensorHash {
  std::size_t operator()(const Tensor& key) const {
    return std::hash<int64_t>{}(key.scalar<int64_t>()());
  }
};

// Primary template.
template <bool Ordered, typename Data>
struct MapTraits;

// Partial specialization for ordered.
template <typename Data>
struct MapTraits<true, Data> {
  using KeyType = Tensor;
  using DataType = Data;
  using MapType = std::map<KeyType, Data, KeyTensorLess>;
};

// Partial specialization for unordered.
template <typename Data>
struct MapTraits<false, Data> {
  using KeyType = Tensor;
  using DataType = Data;
  using MapType =
      std::unordered_map<KeyType, Data, KeyTensorHash, KeyTensorEqual>;
};

// Wrapper around map/unordered_map.
template <bool Ordered>
class StagingMap : public ResourceBase {
 public:
  // Public typedefs
  using Tuple = std::vector<Tensor>;
  using OptionalTensor = gtl::optional<Tensor>;
  using OptionalTuple = std::vector<OptionalTensor>;

  using MapType = typename MapTraits<Ordered, OptionalTuple>::MapType;
  using KeyType = typename MapTraits<Ordered, OptionalTuple>::KeyType;

  using IncompleteType = typename MapTraits<false, OptionalTuple>::MapType;

 private:
  // Private variables
  DataTypeVector dtypes_ TF_GUARDED_BY(mu_);
  std::size_t capacity_ TF_GUARDED_BY(mu_);
  std::size_t memory_limit_ TF_GUARDED_BY(mu_);
  std::size_t current_bytes_ TF_GUARDED_BY(mu_);
  tensorflow::mutex mu_;
  tensorflow::condition_variable not_empty_;
  tensorflow::condition_variable full_;
  IncompleteType incomplete_ TF_GUARDED_BY(mu_);
  MapType map_ TF_GUARDED_BY(mu_);

 private:
  // private methods

  // If map is configured for bounded capacity, notify
  // waiting inserters that space is now available
  void notify_inserters_if_bounded() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_0(mht_0_v, 281, "", "./tensorflow/core/kernels/map_stage_op.cc", "notify_inserters_if_bounded");

    if (has_capacity() || has_memory_limit()) {
      // Notify all inserters. The removal of an element
      // may make memory available for many inserters
      // to insert new elements
      full_.notify_all();
    }
  }

  // Notify all removers waiting to extract values
  // that data is now available
  void notify_removers() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_1(mht_1_v, 295, "", "./tensorflow/core/kernels/map_stage_op.cc", "notify_removers");

    // Notify all removers. This is because they are
    // waiting for specific keys to appear in the map
    // so we don't know which one to wake up.
    not_empty_.notify_all();
  }

  bool has_capacity() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return capacity_ > 0;
  }

  bool has_memory_limit() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return memory_limit_ > 0;
  }

  bool would_exceed_memory_limit(std::size_t bytes) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return has_memory_limit() && bytes + current_bytes_ > memory_limit_;
  }

  bool is_capacity_full() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return has_capacity() && map_.size() >= capacity_;
  }

  // Get number of bytes in the tuple
  std::size_t get_tuple_bytes(const Tuple& tuple) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_2(mht_2_v, 323, "", "./tensorflow/core/kernels/map_stage_op.cc", "get_tuple_bytes");

    return std::accumulate(tuple.begin(), tuple.end(),
                           static_cast<std::size_t>(0),
                           [](const std::size_t& lhs, const Tensor& rhs) {
                             return lhs + rhs.TotalBytes();
                           });
  }

  // Get number of bytes in the incomplete tuple
  std::size_t get_tuple_bytes(const OptionalTuple& tuple) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_3(mht_3_v, 335, "", "./tensorflow/core/kernels/map_stage_op.cc", "get_tuple_bytes");

    return std::accumulate(
        tuple.begin(), tuple.end(), static_cast<std::size_t>(0),
        [](const std::size_t& lhs, const OptionalTensor& rhs) {
          return (lhs + rhs.has_value()) ? rhs.value().TotalBytes() : 0;
        });
  }

  // Check that the index is within bounds
  Status check_index(const Tensor& key, std::size_t index)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_4(mht_4_v, 348, "", "./tensorflow/core/kernels/map_stage_op.cc", "check_index");

    if (index >= dtypes_.size()) {
      return Status(errors::InvalidArgument(
          "Index '", index, "' for key '", key.scalar<int64_t>()(),
          "' was out of bounds '", dtypes_.size(), "'."));
    }

    return Status::OK();
  }

  Status copy_or_move_tensors(OptionalTuple* map_tuple, const Tensor& key,
                              const Tensor& indices, Tuple* output,
                              bool copy = false)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_5(mht_5_v, 364, "", "./tensorflow/core/kernels/map_stage_op.cc", "copy_or_move_tensors");

    auto findices = indices.flat<int>();

    // Return values at specified indices
    for (std::size_t i = 0; i < findices.dimension(0); ++i) {
      std::size_t index = findices(i);

      TF_RETURN_IF_ERROR(check_index(key, index));

      // Insist on a value present at the specified index
      if (!(*map_tuple)[index].has_value()) {
        return Status(errors::InvalidArgument(
            "Tensor at index '", index, "' for key '", key.scalar<int64_t>()(),
            "' has already been removed."));
      }

      // Copy the contained tensor and
      // remove from the OptionalTuple
      output->push_back((*map_tuple)[index].value());

      // Clear out the entry if we're not copying (moving)
      if (!copy) {
        (*map_tuple)[index].reset();
      }
    }

    return Status::OK();
  }

  // Check that the optional value at the specified index
  // is uninitialized
  Status check_index_uninitialized(const Tensor& key, std::size_t index,
                                   const OptionalTuple& tuple)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_6(mht_6_v, 400, "", "./tensorflow/core/kernels/map_stage_op.cc", "check_index_uninitialized");

    if (tuple[index].has_value()) {
      return errors::InvalidArgument("The tensor for index '", index,
                                     "' for key '", key.scalar<int64_t>()(),
                                     "' was already initialized '",
                                     dtypes_.size(), "'.");
    }

    return Status::OK();
  }

  // Check that the indices are strictly ordered
  Status check_index_ordering(const Tensor& indices) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_7(mht_7_v, 415, "", "./tensorflow/core/kernels/map_stage_op.cc", "check_index_ordering");

    if (indices.NumElements() == 0) {
      return errors::InvalidArgument("Indices are empty");
    }

    auto findices = indices.flat<int>();

    for (std::size_t i = 0; i < findices.dimension(0) - 1; ++i) {
      if (findices(i) < findices(i + 1)) {
        continue;
      }

      return errors::InvalidArgument("Indices are not strictly ordered");
    }

    return Status::OK();
  }

  // Check bytes are within memory limits memory limits
  Status check_memory_limit(std::size_t bytes)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_8(mht_8_v, 438, "", "./tensorflow/core/kernels/map_stage_op.cc", "check_memory_limit");

    if (has_memory_limit() && bytes > memory_limit_) {
      return errors::ResourceExhausted(
          "Attempted to insert tensors with combined size of '", bytes,
          "' bytes into Staging Area with a memory limit of '", memory_limit_,
          "'.");
    }

    return Status::OK();
  }

  // Insert incomplete data into the Barrier
  Status put_incomplete(const KeyType& key, const Tensor& indices,
                        OptionalTuple* tuple, tensorflow::mutex_lock* lock)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_9(mht_9_v, 455, "", "./tensorflow/core/kernels/map_stage_op.cc", "put_incomplete");

    auto findices = indices.flat<int>();

    // Search for the key in our incomplete set
    auto it = incomplete_.find(key);

    // Check that the tuple fits within the memory limit
    std::size_t tuple_bytes = get_tuple_bytes(*tuple);
    TF_RETURN_IF_ERROR(check_memory_limit(tuple_bytes));

    // Wait until we don't exceed the memory limit
    while (would_exceed_memory_limit(tuple_bytes)) {
      full_.wait(*lock);
    }

    // This key isn't present in the incomplete set
    // Create OptionalTuple and insert
    if (it == incomplete_.end()) {
      OptionalTuple empty(dtypes_.size());

      // Initialize empty tuple with given dta
      for (std::size_t i = 0; i < findices.dimension(0); ++i) {
        std::size_t index = findices(i);
        TF_RETURN_IF_ERROR(check_index(key, index));

        // Assign tuple at this index
        empty[index] = std::move((*tuple)[i]);
      }

      // Insert into incomplete map
      incomplete_.insert({key, std::move(empty)});

      // Increment size
      current_bytes_ += tuple_bytes;
    }
    // Found an entry in the incomplete index
    // Update with given data and insert complete entries
    // into the main map
    else {
      // Reference existing incomplete tuple
      OptionalTuple& present = it->second;

      // Assign given data
      for (std::size_t i = 0; i < findices.dimension(0); ++i) {
        std::size_t index = findices(i);
        TF_RETURN_IF_ERROR(check_index(key, index));
        TF_RETURN_IF_ERROR(check_index_uninitialized(key, index, present));

        // Assign tuple at this index
        present[index] = std::move((*tuple)[i]);
      }

      // Increment size
      current_bytes_ += tuple_bytes;

      // Do we have values at all tuple elements?
      bool complete =
          std::all_of(present.begin(), present.end(),
                      [](const OptionalTensor& v) { return v.has_value(); });

      // If so, put the tuple in the actual map
      if (complete) {
        OptionalTuple insert_tuple = std::move(it->second);

        // Remove from incomplete
        incomplete_.erase(it);

        TF_RETURN_IF_ERROR(put_complete(key, &insert_tuple));
      }
    }

    return Status::OK();
  }

  // Does the insertion into the actual staging area
  Status put_complete(const KeyType& key, OptionalTuple* tuple)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_10(mht_10_v, 534, "", "./tensorflow/core/kernels/map_stage_op.cc", "put_complete");

    // Insert key and tuples into the map
    map_.insert({key, std::move(*tuple)});

    notify_removers();

    return Status::OK();
  }

 public:
  // public methods
  explicit StagingMap(const DataTypeVector& dtypes, std::size_t capacity,
                      std::size_t memory_limit)
      : dtypes_(dtypes),
        capacity_(capacity),
        memory_limit_(memory_limit),
        current_bytes_(0) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_11(mht_11_v, 553, "", "./tensorflow/core/kernels/map_stage_op.cc", "StagingMap");
}

  Status put(KeyType* key, const Tensor* indices, OptionalTuple* tuple) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_12(mht_12_v, 558, "", "./tensorflow/core/kernels/map_stage_op.cc", "put");

    tensorflow::mutex_lock lock(mu_);

    // Sanity check the indices
    TF_RETURN_IF_ERROR(check_index_ordering(*indices));

    // Handle incomplete inserts
    if (indices->NumElements() != dtypes_.size()) {
      return put_incomplete(*key, *indices, tuple, &lock);
    }

    std::size_t tuple_bytes = get_tuple_bytes(*tuple);
    // Check that tuple_bytes fits within the memory limit
    TF_RETURN_IF_ERROR(check_memory_limit(tuple_bytes));

    // Wait until there's space for insertion.
    while (would_exceed_memory_limit(tuple_bytes) || is_capacity_full()) {
      full_.wait(lock);
    }

    // Do the put operation
    TF_RETURN_IF_ERROR(put_complete(*key, tuple));

    // Update the current size
    current_bytes_ += tuple_bytes;

    return Status::OK();
  }

  Status get(const KeyType* key, const Tensor* indices, Tuple* tuple) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_13(mht_13_v, 590, "", "./tensorflow/core/kernels/map_stage_op.cc", "get");

    tensorflow::mutex_lock lock(mu_);

    // Sanity check the indices
    TF_RETURN_IF_ERROR(check_index_ordering(*indices));

    typename MapType::iterator it;

    // Wait until the element with the requested key is present
    while ((it = map_.find(*key)) == map_.end()) {
      not_empty_.wait(lock);
    }

    TF_RETURN_IF_ERROR(
        copy_or_move_tensors(&it->second, *key, *indices, tuple, true));

    // Update bytes in the Staging Area
    current_bytes_ -= get_tuple_bytes(*tuple);

    return Status::OK();
  }

  Status pop(const KeyType* key, const Tensor* indices, Tuple* tuple) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_14(mht_14_v, 615, "", "./tensorflow/core/kernels/map_stage_op.cc", "pop");

    tensorflow::mutex_lock lock(mu_);

    // Sanity check the indices
    TF_RETURN_IF_ERROR(check_index_ordering(*indices));

    typename MapType::iterator it;

    // Wait until the element with the requested key is present
    while ((it = map_.find(*key)) == map_.end()) {
      not_empty_.wait(lock);
    }

    TF_RETURN_IF_ERROR(
        copy_or_move_tensors(&it->second, *key, *indices, tuple));

    // Remove entry if all the values have been consumed
    if (!std::any_of(
            it->second.begin(), it->second.end(),
            [](const OptionalTensor& tensor) { return tensor.has_value(); })) {
      map_.erase(it);
    }

    // Update bytes in the Staging Area
    current_bytes_ -= get_tuple_bytes(*tuple);

    notify_inserters_if_bounded();

    return Status::OK();
  }

  Status popitem(KeyType* key, const Tensor* indices, Tuple* tuple) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_15(mht_15_v, 649, "", "./tensorflow/core/kernels/map_stage_op.cc", "popitem");

    tensorflow::mutex_lock lock(mu_);

    // Sanity check the indices
    TF_RETURN_IF_ERROR(check_index_ordering(*indices));

    // Wait until map is not empty
    while (this->map_.empty()) {
      not_empty_.wait(lock);
    }

    // Move from the first element and erase it

    auto it = map_.begin();

    TF_RETURN_IF_ERROR(
        copy_or_move_tensors(&it->second, *key, *indices, tuple));

    *key = it->first;

    // Remove entry if all the values have been consumed
    if (!std::any_of(
            it->second.begin(), it->second.end(),
            [](const OptionalTensor& tensor) { return tensor.has_value(); })) {
      map_.erase(it);
    }

    // Update bytes in the Staging Area
    current_bytes_ -= get_tuple_bytes(*tuple);

    notify_inserters_if_bounded();

    return Status::OK();
  }

  Status clear() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_16(mht_16_v, 687, "", "./tensorflow/core/kernels/map_stage_op.cc", "clear");

    tensorflow::mutex_lock lock(mu_);
    map_.clear();
    incomplete_.clear();
    current_bytes_ = 0;

    notify_inserters_if_bounded();

    return Status::OK();
  }

  std::size_t incomplete_size() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_17(mht_17_v, 701, "", "./tensorflow/core/kernels/map_stage_op.cc", "incomplete_size");

    tensorflow::mutex_lock lock(mu_);
    return incomplete_.size();
  }

  std::size_t size() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_18(mht_18_v, 709, "", "./tensorflow/core/kernels/map_stage_op.cc", "size");

    tensorflow::mutex_lock lock(mu_);
    return map_.size();
  }

  string DebugString() const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_19(mht_19_v, 717, "", "./tensorflow/core/kernels/map_stage_op.cc", "DebugString");
 return "StagingMap"; }
};

template <bool Ordered>
Status GetStagingMap(OpKernelContext* ctx, const NodeDef& ndef,
                     StagingMap<Ordered>** map) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_20(mht_20_v, 725, "", "./tensorflow/core/kernels/map_stage_op.cc", "GetStagingMap");

  auto rm = ctx->resource_manager();
  ContainerInfo cinfo;

  // Lambda for creating the Staging Area
  auto create_fn = [&ndef](StagingMap<Ordered>** ret) -> Status {
    DataTypeVector dtypes;
    int64_t capacity;
    int64_t memory_limit;
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "dtypes", &dtypes));
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "capacity", &capacity));
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "memory_limit", &memory_limit));
    *ret = new StagingMap<Ordered>(dtypes, capacity, memory_limit);
    return Status::OK();
  };

  TF_RETURN_IF_ERROR(cinfo.Init(rm, ndef, true /* use name() */));
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<StagingMap<Ordered>>(
      cinfo.container(), cinfo.name(), map, create_fn));
  return Status::OK();
}

template <bool Ordered>
class MapStageOp : public OpKernel {
 public:
  explicit MapStageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_21(mht_21_v, 753, "", "./tensorflow/core/kernels/map_stage_op.cc", "MapStageOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_22(mht_22_v, 758, "", "./tensorflow/core/kernels/map_stage_op.cc", "Compute");

    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::OptionalTuple tuple;

    const Tensor* key_tensor;
    const Tensor* indices_tensor;
    OpInputList values_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
    OP_REQUIRES_OK(ctx, ctx->input_list("values", &values_tensor));
    OP_REQUIRES(ctx, key_tensor->NumElements() > 0,
                errors::InvalidArgument("key must not be empty"));

    OP_REQUIRES(ctx, key_tensor->NumElements() == 1,
                errors::InvalidArgument(
                    "key must be an int64 scalar, got tensor with shape: ",
                    key_tensor->shape()));

    // Create copy for insertion into Staging Area
    Tensor key(*key_tensor);

    // Create the tuple to store
    for (std::size_t i = 0; i < values_tensor.size(); ++i) {
      tuple.push_back(values_tensor[i]);
    }

    // Store the tuple in the map
    OP_REQUIRES_OK(ctx, map->put(&key, indices_tensor, &tuple));
  }
};

REGISTER_KERNEL_BUILDER(Name("MapStage").Device(DEVICE_CPU), MapStageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapStage").Device(DEVICE_CPU),
                        MapStageOp<true>);

REGISTER_KERNEL_BUILDER(Name("MapStage")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapStageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapStage")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapStageOp<true>);

template <bool Ordered>
class MapUnstageOp : public OpKernel {
 public:
  explicit MapUnstageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_23(mht_23_v, 813, "", "./tensorflow/core/kernels/map_stage_op.cc", "MapUnstageOp");
}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_24(mht_24_v, 820, "", "./tensorflow/core/kernels/map_stage_op.cc", "Compute");

    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor* key_tensor;
    const Tensor* indices_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
    OP_REQUIRES_OK(ctx, map->pop(key_tensor, indices_tensor, &tuple));

    OP_REQUIRES(
        ctx, tuple.size() == indices_tensor->NumElements(),
        errors::InvalidArgument("output/indices size mismatch: ", tuple.size(),
                                " vs. ", indices_tensor->NumElements()));

    for (std::size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapUnstage").Device(DEVICE_CPU),
                        MapUnstageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstage").Device(DEVICE_CPU),
                        MapUnstageOp<true>);

REGISTER_KERNEL_BUILDER(Name("MapUnstage")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapUnstageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstage")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapUnstageOp<true>);

template <bool Ordered>
class MapPeekOp : public OpKernel {
 public:
  explicit MapPeekOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_25(mht_25_v, 866, "", "./tensorflow/core/kernels/map_stage_op.cc", "MapPeekOp");
}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_26(mht_26_v, 873, "", "./tensorflow/core/kernels/map_stage_op.cc", "Compute");

    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor* key_tensor;
    const Tensor* indices_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
    OP_REQUIRES_OK(ctx, map->get(key_tensor, indices_tensor, &tuple));

    OP_REQUIRES(
        ctx, tuple.size() == indices_tensor->NumElements(),
        errors::InvalidArgument("output/indices size mismatch: ", tuple.size(),
                                " vs. ", indices_tensor->NumElements()));

    for (std::size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapPeek").Device(DEVICE_CPU), MapPeekOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapPeek").Device(DEVICE_CPU),
                        MapPeekOp<true>);

REGISTER_KERNEL_BUILDER(
    Name("MapPeek").HostMemory("key").HostMemory("indices").Device(
        DEVICE_DEFAULT),
    MapPeekOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapPeek")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapPeekOp<true>);

template <bool Ordered>
class MapUnstageNoKeyOp : public OpKernel {
 public:
  explicit MapUnstageNoKeyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_27(mht_27_v, 917, "", "./tensorflow/core/kernels/map_stage_op.cc", "MapUnstageNoKeyOp");
}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_28(mht_28_v, 924, "", "./tensorflow/core/kernels/map_stage_op.cc", "Compute");

    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    // Pop a random (key, value) off the map
    typename StagingMap<Ordered>::KeyType key;
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor* indices_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
    OP_REQUIRES_OK(ctx, map->popitem(&key, indices_tensor, &tuple));

    // Allocate a key tensor and assign the key as the first output
    ctx->set_output(0, key);

    // Set the rest of the outputs to the tuple Tensors
    OP_REQUIRES(
        ctx, tuple.size() == indices_tensor->NumElements(),
        errors::InvalidArgument("output/indices size mismatch: ", tuple.size(),
                                " vs. ", indices_tensor->NumElements()));

    for (std::size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i + 1, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapUnstageNoKey").Device(DEVICE_CPU),
                        MapUnstageNoKeyOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstageNoKey").Device(DEVICE_CPU),
                        MapUnstageNoKeyOp<true>);

REGISTER_KERNEL_BUILDER(Name("MapUnstageNoKey")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapUnstageNoKeyOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstageNoKey")
                            .HostMemory("key")
                            .HostMemory("indices")
                            .Device(DEVICE_DEFAULT),
                        MapUnstageNoKeyOp<true>);

template <bool Ordered>
class MapSizeOp : public OpKernel {
 public:
  explicit MapSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_29(mht_29_v, 975, "", "./tensorflow/core/kernels/map_stage_op.cc", "MapSizeOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_30(mht_30_v, 980, "", "./tensorflow/core/kernels/map_stage_op.cc", "Compute");

    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    // Allocate size output tensor
    Tensor* size = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &size));

    // Set it to the actual size
    size->scalar<int32>().setConstant(map->size());
  }
};

REGISTER_KERNEL_BUILDER(Name("MapSize").Device(DEVICE_CPU), MapSizeOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapSize").Device(DEVICE_CPU),
                        MapSizeOp<true>);

REGISTER_KERNEL_BUILDER(
    Name("MapSize").Device(DEVICE_DEFAULT).HostMemory("size"),
    MapSizeOp<false>);
REGISTER_KERNEL_BUILDER(
    Name("OrderedMapSize").Device(DEVICE_DEFAULT).HostMemory("size"),
    MapSizeOp<true>);

template <bool Ordered>
class MapIncompleteSizeOp : public OpKernel {
 public:
  explicit MapIncompleteSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_31(mht_31_v, 1011, "", "./tensorflow/core/kernels/map_stage_op.cc", "MapIncompleteSizeOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_32(mht_32_v, 1016, "", "./tensorflow/core/kernels/map_stage_op.cc", "Compute");

    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    // Allocate size output tensor
    Tensor* size = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &size));

    // Set it to the actual size
    size->scalar<int32>().setConstant(map->incomplete_size());
  }
};

REGISTER_KERNEL_BUILDER(Name("MapIncompleteSize").Device(DEVICE_CPU),
                        MapIncompleteSizeOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapIncompleteSize").Device(DEVICE_CPU),
                        MapIncompleteSizeOp<true>);

REGISTER_KERNEL_BUILDER(
    Name("MapIncompleteSize").Device(DEVICE_DEFAULT).HostMemory("size"),
    MapIncompleteSizeOp<false>);
REGISTER_KERNEL_BUILDER(
    Name("OrderedMapIncompleteSize").Device(DEVICE_DEFAULT).HostMemory("size"),
    MapIncompleteSizeOp<true>);

template <bool Ordered>
class MapClearOp : public OpKernel {
 public:
  explicit MapClearOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_33(mht_33_v, 1048, "", "./tensorflow/core/kernels/map_stage_op.cc", "MapClearOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_stage_opDTcc mht_34(mht_34_v, 1053, "", "./tensorflow/core/kernels/map_stage_op.cc", "Compute");

    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    OP_REQUIRES_OK(ctx, map->clear());
  }
};

REGISTER_KERNEL_BUILDER(Name("MapClear").Device(DEVICE_CPU), MapClearOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapClear").Device(DEVICE_CPU),
                        MapClearOp<true>);

REGISTER_KERNEL_BUILDER(Name("MapClear").Device(DEVICE_DEFAULT),
                        MapClearOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapClear").Device(DEVICE_DEFAULT),
                        MapClearOp<true>);

}  // namespace
}  // namespace tensorflow
