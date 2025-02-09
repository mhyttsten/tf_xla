/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// A tensor bundle is a set of immutable persistent files storing a set of named
// tensors.  It is designed for checkpointing TensorFlow tensors.
//
// The paths of the managed files share a common prefix; e.g., with the prefix:
//   /fs/model/train/ckpt-step/ckpt
//
// the bundle may contain a metadata file, and sharded data files:
//   /fs/model/train/ckpt-step/
//       ckpt.index
//       ckpt.data-00000-of-00020
//       ckpt.data-00001-of-00020
//       ...
//       ckpt.data-00019-of-00020
//
// The ".index" file is a string-string immutable table
// (tensorflow::table::Table).  Each key is a name of a tensor and its value is
// a serialized BundleEntryProto.  Each BundleEntryProto describes the metadata
// of a tensor: which of the "data" files contains the content of a tensor, the
// offset into that file, checksum, some auxiliary data, etc.
//
// A tensor bundle can be accessed randomly using a BundleReader.  Usage:
//
//   BundleReader reader(env, "/fs/model/train/ckpt-step/ckpt");
//   reader.Lookup("name", &tensor);
//
// A tensor bundle can be built using BundleWriter.  Each BundleWriter builds a
// single data file bundle.  Multiple bundles can then be merged by
// MergeBundles() without reading and writing large chunk of data: it reads the
// metadata files and outputs a single merged metadata.  Typical usage:
//
//   worker 0:
//     BundleWriter writer(env, "/fs/model/train/ckpt-step/tmp/worker0-step");
//     writer.Add(...);  // Adds the tensors on this worker.
//     writer.Finish();  // Flushes.
//   worker 1:
//     BundleWriter writer(env, "/fs/model/train/ckpt-step/tmp/worker1-step");
//     writer.Add(...);
//     writer.Finish();
//   worker 2:
//     MergeBundles(env,
//       {"/fs/model/train/ckpt-step/tmp/worker0-step",
//        "/fs/model/train/ckpt-step/tmp/worker1-step"},
//       "/fs/model/train/ckpt-step/ckpt" /* merged prefix */);
//

#ifndef TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_TENSOR_BUNDLE_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_TENSOR_BUNDLE_H_
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
class MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh() {
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


#include <map>
#include <string>
#include <unordered_map>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/io/cache.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/table.h"
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/tensor_bundle.pb.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/core/util/tensor_slice_set.h"

namespace tensorflow {

class FileOutputBuffer;

// Versioning of the tensor bundle format.
// Follows the same rules as 3p/tf/core/public/version.h.
//
// History:
// 0. Any tensor bundles produced before this field was added.
// 1. Added this field (2016-09-14).
extern const int kTensorBundleMinProducer;
extern const int kTensorBundleMinConsumer;
extern const int kTensorBundleVersion;

// The empty string, hence always the first key in the metadata table.  Its
// corresponding value is a BundleHeaderProto.
extern const char* const kHeaderEntryKey;

// Builds a string-string table of tensor names to BundleEntryProto (metadata).
//
// On construction, attempts to create a directory given by the dirname of
// "prefix", so "status()" must be checked before calling any member functions.
//
// All threads accessing the same BundleWriter must synchronize.
class BundleWriter {
 public:
  struct Options {
    Options() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh mht_0(mht_0_v, 284, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle.h", "Options");
}
    // Alignment, in bytes, for tensor data.
    // Must be >= 1. The default size of 1 densely packs tensors.
    int data_alignment{1};
  };
  BundleWriter(Env* env, StringPiece prefix,
               const Options& options = Options());

  // Adds the tensor "val" under key "key".
  // Across calls "key" must be unique but can be added in any order.
  Status Add(StringPiece key, const Tensor& val);

  // Partitioned variables support.
  // A slice of a full tensor is stored in two entries in the metadata table:
  //
  //   full_tensor_key   -> BundleEntryProto, describing all stored slices
  //                        of this full tensor.  Does not append to the data
  //                        file.
  //   encoded slice key -> BundleEntryProto, describing one particular slice.
  //                        Appends values of this slice to the data file.
  //
  // Slices of a full tensor can be added in any order.
  //
  // If a full tensor has slices placed on N devices and N BundleWriter's are
  // concurrently used, the caller must use MergeBundles() to ensure that a
  // consistent entry for "full_tensor_key" is produced.
  //
  // Returns an error if the same slice is added the second time.
  Status AddSlice(StringPiece full_tensor_key,
                  const TensorShape& full_tensor_shape,
                  const TensorSlice& slice_spec, const Tensor& slice_tensor);

  // Finishes the writer and flushes.
  Status Finish() TF_MUST_USE_RESULT;

  Status status() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh mht_1(mht_1_v, 322, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle.h", "status");
 return status_; }

 private:
  Env* const env_;  // Not owned.
  const Options options_;
  const string prefix_;
  string metadata_path_;
  string data_path_;
  bool use_temp_file_;
  std::unique_ptr<FileOutputBuffer> out_;
  int64_t size_;  // Number of bytes written into out_.
  std::map<string, BundleEntryProto> entries_;
  Status status_;

  TF_DISALLOW_COPY_AND_ASSIGN(BundleWriter);
};

// Merges a set of bundles (given their prefixes) into a single bundle with the
// given "merged_prefix".  The merged metadata is guaranteed to be consistent.
//
// If there are N bundles in "prefixes", during the merge the data files will be
// renamed to contain a proper sharded file spec, with num_shards set to the sum
// of num_shards across the N input bundles.
//
// The caller should only rely on the metadata file of the merged bundle to
// query information about a tensor.  In particular, this function does not
// guarantee not to re-order the input data files.
//
// Once merged, makes a best effort to delete the old metadata files.
// Returns OK iff all bundles are successfully merged.
Status MergeBundles(Env* env, gtl::ArraySlice<tstring> prefixes,
                    StringPiece merged_prefix);

// On construction, silently attempts to read the metadata associated with
// "prefix".  If caller intends to call any function afterwards, "status()"
// must be checked.
// All threads accessing the same BundleReader must synchronize.
class BundleReader {
 public:
  BundleReader(Env* const env, StringPiece prefix);
  ~BundleReader();

  // Is ok() iff the reader construction is successful (completed the read of
  // the metadata).
  Status status() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh mht_2(mht_2_v, 369, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle.h", "status");
 return status_; }

  // Queries whether the bundle contains an entry keyed by "key".  Calls Seek()
  // internally, so this call invalidates the reader's current position.
  // REQUIRES: status().ok()
  bool Contains(StringPiece key);

  // Sorts a `container` of tensors to read such that when `Seek(key)` is called
  // on the elements of the sorted container, the underlying file access is
  // sequential. Sorting can greatly improve overall read speed.
  //
  // `get_key` should be a functon that when passed an element in `container`,
  // returns the `key` of the tensor.
  //
  // REQUIRES: status().ok()
  template <class T>
  Status SortForSequentialAccess(std::vector<T>& container,
                                 absl::FunctionRef<string(const T&)> get_key);

  // Looks up the dtype and the shape of the tensor keyed by "key".
  // REQUIRES: status().ok()
  Status LookupDtypeAndShape(StringPiece key, DataType* dtype,
                             TensorShape* shape) TF_MUST_USE_RESULT;

  // Looks up the shape of the tensor keyed by "key".
  // Clears "shape" if not found.
  // REQUIRES: status().ok()
  Status LookupTensorShape(StringPiece key,
                           TensorShape* shape) TF_MUST_USE_RESULT;

  // Looks up the tensor keyed by "key".  If "key" refers to a partitioned
  // tensor, attempts to look up the full contents using all stored slices.
  //
  // Caller must make sure "val" has the same shape and dtype as the
  // corresponding contents, so that its buffer can be filled without needing
  // extra allocation.  These can be queried via "LookupDtypeAndShape()".
  //
  // On error, "val" may contain nonsense data.  Returns a NotFound error if
  // tensor keyed by "key" does not exist in this bundle.
  //
  // Validates the stored crc32c checksum against the restored bytes.
  // REQUIRES: status().ok()
  Status Lookup(StringPiece key, Tensor* val) TF_MUST_USE_RESULT;

  // Looks up the tensor pointed to by the internal iterator.
  //
  // On error, "val" may contain nonsense data.
  //
  // Validates the stored crc32c checksum against the restored bytes.
  // REQUIRES: status().ok() && Valid()
  Status ReadCurrent(Tensor* val) TF_MUST_USE_RESULT;

  // Looks up the slices of the tensor keyed by "key".  On OK, "slices"
  // is non-empty if and only if the tensor is a partitioned tensor.
  //
  // Warning - there is no guaranteed ordering for the returned slices, so
  // a slice with a larger start index in some dimension could come before
  // another slice with a smaller start index in the same dimension.
  // REQUIRES: status().ok()
  Status LookupTensorSlices(StringPiece key, std::vector<TensorSlice>* slices)
      TF_MUST_USE_RESULT;

  // Looks up a specific slice of a partitioned tensor.
  // It is only required that the stored slices cover the requested slice,
  // namely "slice_spec" is a subset of the union of the stored slices.
  // REQUIRES: status().ok()
  Status LookupSlice(StringPiece full_tensor_key, const TensorSlice& slice_spec,
                     Tensor* val) TF_MUST_USE_RESULT;

  // Seeks to the first position in the bundle whose key is no less than "key".
  // REQUIRES: status().ok()
  void Seek(StringPiece key) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh mht_3(mht_3_v, 443, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle.h", "Seek");
 return iter_->Seek(key); }
  // Moves to the next position in the bundle.
  // REQUIRES: status().ok()
  void Next() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh mht_4(mht_4_v, 449, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle.h", "Next");
 iter_->Next(); }
  // Returns true iff the reader is positioned to a key/val pair.
  // REQUIRES: status().ok()
  bool Valid() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh mht_5(mht_5_v, 455, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle.h", "Valid");
 return iter_->Valid(); }

  // Returns the key at the current position.
  // REQUIRES: status().ok() && Valid()
  StringPiece key() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh mht_6(mht_6_v, 462, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle.h", "key");
 return iter_->key(); }
  // Returns the raw value at the current position.
  // REQUIRES: status().ok() && Valid()
  StringPiece value() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh mht_7(mht_7_v, 468, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle.h", "value");
 return iter_->value(); }

  string DebugString();

 private:
  // Seeks for "key" and reads the metadata proto.
  // On non-OK return, clears "entry" for the caller.
  // REQUIRES: status().ok()
  Status GetBundleEntryProto(StringPiece key,
                             BundleEntryProto* entry) TF_MUST_USE_RESULT;

  // Reads the tensor value described by the metadata proto "entry".
  // Usage for "val" follows the comment of "Lookup()".
  Status GetValue(const BundleEntryProto& entry,
                  Tensor* val) TF_MUST_USE_RESULT;

  // Reads the slice described by "slice_spec".  The corresponding full tensor
  // has key "ful_tensor_key" and metadata proto "full_tensor_entry".
  // REQUIRES: full_tensor_entry.slices_size() > 0
  Status GetSliceValue(StringPiece full_tensor_key,
                       const BundleEntryProto& full_tensor_entry,
                       const TensorSlice& slice_spec,
                       Tensor* val) TF_MUST_USE_RESULT;

  Env* env_;  // Not owned.
  const string prefix_;

  Status status_;
  RandomAccessFile* metadata_;  // Owned.
  table::Table* table_;
  table::Cache* index_cache_;
  table::Iterator* iter_;
  // Owned the InputBuffer objects and their underlying RandomAccessFile's.
  std::unordered_map<int32, io::InputBuffer*> data_;

  // Maps each partitioned tensor's key to its stored slices (represented in a
  // TensorSliceSet).  Populated on-demand.
  std::unordered_map<string, checkpoint::TensorSliceSet*> tensor_slices_;

  // Expected number of data file shards in the bundle.  Extracted by reading
  // the header entry in the metadata table.
  int num_shards_;

  // Flag that this class sets to true when the endianness of the target bundle
  // differs from that of the current system's processor architecture.
  bool need_to_swap_bytes_;

  friend class TensorBundleAlignmentTest;  // For testing data alignment.

  TF_DISALLOW_COPY_AND_ASSIGN(BundleReader);
};

// A buffering wrapper for a WritableFile.  Useful if the caller wishes to issue
// small writes to a file (e.g. writing out a list of small varints).
// External synchronization must be used in the presence of concurrent callers.
class FileOutputBuffer {
 public:
  FileOutputBuffer(WritableFile* file, size_t buffer_size);
  ~FileOutputBuffer();

  // Buffered append.
  Status Append(StringPiece data);

  // Returns the running crc32c checksum of all currently appended bytes.
  uint32 crc32c() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh mht_8(mht_8_v, 535, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle.h", "crc32c");
 return crc32c_; }
  // Clears the running crc32c checksum.
  void clear_crc32c() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh mht_9(mht_9_v, 540, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle.h", "clear_crc32c");
 crc32c_ = 0; }

  // Appends the buffered data, then closes the underlying file.
  Status Close();

 private:
  // Appends the buffered data to the underlying file. Does NOT flush the file.
  Status FlushBuffer(bool closing);

  WritableFile* file_;  // Owned.

  // buffer_ptr_[0, position_) holds the buffered data not yet appended to the
  // underlying file.
  size_t position_;
  const size_t buffer_size_;
  char* buffer_ptr_;

  // Checksum of all appended bytes since construction or last clear_crc32c().
  uint32 crc32c_ = 0;
};

template <class T>
Status BundleReader::SortForSequentialAccess(
    std::vector<T>& container, absl::FunctionRef<string(const T&)> get_key) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundleDTh mht_10(mht_10_v, 566, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle.h", "BundleReader::SortForSequentialAccess");

  struct FileOffset {
    int32_t shard_id;
    int64_t offset;
  };
  absl::flat_hash_map<string, FileOffset> file_offsets;
  for (const T& element : container) {
    BundleEntryProto entry;
    TF_RETURN_IF_ERROR(GetBundleEntryProto(get_key(element), &entry));
    file_offsets[get_key(element)] = {entry.shard_id(), entry.offset()};
  }
  absl::c_sort(container, [&get_key, &file_offsets](const T& a, const T& b) {
    const FileOffset& file_offset_a = file_offsets[get_key(a)];
    const FileOffset& file_offset_b = file_offsets[get_key(b)];
    if (file_offset_a.shard_id == file_offset_b.shard_id) {
      return file_offset_a.offset < file_offset_b.offset;
    } else {
      return file_offset_a.shard_id < file_offset_b.shard_id;
    }
  });
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_TENSOR_BUNDLE_H_
