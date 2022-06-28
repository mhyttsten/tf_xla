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

#ifndef TENSORFLOW_CORE_DATA_SNAPSHOT_UTILS_H_
#define TENSORFLOW_CORE_DATA_SNAPSHOT_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTh() {
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


#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

class GraphDef;

namespace data {

namespace experimental {

class SnapshotMetadataRecord;
class SnapshotTensorMetadata;

}  // namespace experimental

namespace snapshot_util {

constexpr char kMetadataFilename[] = "snapshot.metadata";

constexpr char kModeAuto[] = "auto";
constexpr char kModeWrite[] = "write";
constexpr char kModeRead[] = "read";
constexpr char kModePassthrough[] = "passthrough";
constexpr char kShardDirectorySuffix[] = ".shard";

enum Mode { READER = 0, WRITER = 1, PASSTHROUGH = 2 };

// Returns the name of the "hash" directory for the given base path and hash ID.
std::string HashDirectory(const std::string& path, uint64 hash);

// Returns the name of the "run" directory for the given base path and run ID.
std::string RunDirectory(const std::string& hash_directory, uint64 run_id);
std::string RunDirectory(const std::string& hash_directory,
                         const std::string& run_id);

// Returns the name of the "shard" directory for the given base path and shard
// ID.
std::string ShardDirectory(const std::string& run_directory, int64_t shard_id);

// Returns the checkpoint file name for the given directory and checkpoint ID.
std::string GetCheckpointFileName(const std::string& shard_directory,
                                  const uint64 checkpoint_id);

// This is a interface class that exposes snapshot writing functionality.
class Writer {
 public:
  // Creates a new writer object.
  static Status Create(Env* env, const std::string& filename,
                       const std::string& compression_type, int version,
                       const DataTypeVector& dtypes,
                       std::unique_ptr<Writer>* out_writer);

  // Writes a vector of tensors to the snapshot writer file.
  virtual Status WriteTensors(const std::vector<Tensor>& tensors) = 0;

  // Flushes any in-memory buffers to disk.
  virtual Status Sync() = 0;

  // Closes and finalizes the snapshot file. All calls to any other method will
  // be invalid after this call.
  virtual Status Close() = 0;

  virtual ~Writer() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTh mht_0(mht_0_v, 261, "", "./tensorflow/core/data/snapshot_utils.h", "~Writer");
}

 protected:
  virtual Status Initialize(tensorflow::Env* env) = 0;
};

// Writes snapshots with the standard TFRecord file format.
class TFRecordWriter : public Writer {
 public:
  TFRecordWriter(const std::string& filename,
                 const std::string& compression_type);

  Status WriteTensors(const std::vector<Tensor>& tensors) override;

  Status Sync() override;

  Status Close() override;

  ~TFRecordWriter() override;

 protected:
  Status Initialize(tensorflow::Env* env) override;

 private:
  const std::string filename_;
  const std::string compression_type_;

  std::unique_ptr<WritableFile> dest_;
  std::unique_ptr<io::RecordWriter> record_writer_;
};

// Writes snapshot with a custom (legacy) file format.
class CustomWriter : public Writer {
 public:
  static constexpr const size_t kHeaderSize = sizeof(uint64);

  static constexpr const char* const kClassName = "SnapshotWriter";
  static constexpr const char* const kWriteStringPiece = "WriteStringPiece";
  static constexpr const char* const kWriteCord = "WriteCord";
  static constexpr const char* const kSeparator = "::";

  CustomWriter(const std::string& filename, const std::string& compression_type,
               const DataTypeVector& dtypes);

  Status WriteTensors(const std::vector<Tensor>& tensors) override;

  Status Sync() override;

  Status Close() override;

  ~CustomWriter() override;

 protected:
  Status Initialize(tensorflow::Env* env) override;

 private:
  Status WriteRecord(const StringPiece& data);

#if defined(TF_CORD_SUPPORT)
  Status WriteRecord(const absl::Cord& data);
#endif  // TF_CORD_SUPPORT

  std::unique_ptr<WritableFile> dest_;
  const std::string filename_;
  const std::string compression_type_;
  const DataTypeVector dtypes_;
  // We hold zlib_dest_ because we may create a ZlibOutputBuffer and put that
  // in dest_ if we want compression. ZlibOutputBuffer doesn't own the original
  // dest_ and so we need somewhere to store the original one.
  std::unique_ptr<WritableFile> zlib_underlying_dest_;
  std::vector<bool> simple_tensor_mask_;  // true for simple, false for complex.
  int num_simple_ = 0;
  int num_complex_ = 0;
};

// Interface class for reading snapshot files previous written with Writer.
class Reader {
 public:
  // Op kernel that creates an instance of `Reader::Dataset` needed to support
  // serialization and deserialization of `Reader::Dataset`.
  class DatasetOp : public DatasetOpKernel {
   public:
    explicit DatasetOp(OpKernelConstruction* ctx);

   protected:
    void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

   private:
    DataTypeVector output_types_;
    std::vector<PartialTensorShape> output_shapes_;
    std::string compression_;
    int64_t version_;
  };

  // Op kernel that creates an instance of `Reader::NestedDataset` needed to
  // support serialization and deserialization of `Reader::NestedDataset`.
  class NestedDatasetOp : public DatasetOpKernel {
   public:
    explicit NestedDatasetOp(OpKernelConstruction* ctx);

   protected:
    void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

   private:
    DataTypeVector output_types_;
    std::vector<PartialTensorShape> output_shapes_;
  };

  // Creates a new Reader object that reads data from `filename`. Note that
  // the `version`, `compression_type`, and `dtypes` arguments passed into
  // `Writer` and `Reader` must be the same for the reading to succeed.
  static Status Create(Env* env, const std::string& filename,
                       const string& compression_type, int version,
                       const DataTypeVector& dtypes,
                       std::unique_ptr<Reader>* out_reader);

  // Returns a nested dataset for a set of given snapshot file names.
  //
  // This function takes a vector of snapshot files, and returns a nested
  // dataset. Each element within the nested dataset is itself a dataset, and
  // contains all the elements written out to each individual snapshot file.
  static Status MakeNestedDataset(Env* env,
                                  const std::vector<std::string>& shard_dirs,
                                  const string& compression_type, int version,
                                  const DataTypeVector& dtypes,
                                  const std::vector<PartialTensorShape>& shapes,
                                  const int64_t start_index,
                                  DatasetBase** output);

  // Reads a vector of Tensors from the snapshot file.
  virtual Status ReadTensors(std::vector<Tensor>* read_tensors) = 0;

  // Skips `num_records`. Equivalent to calling `ReadTensors` `num_records`
  // times then discarding the results.
  virtual Status SkipRecords(int64_t num_records);

  virtual ~Reader() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTh mht_1(mht_1_v, 400, "", "./tensorflow/core/data/snapshot_utils.h", "~Reader");
}

 protected:
  virtual Status Initialize(Env* env) = 0;

  class Dataset;
  class NestedDataset;
};

// Reads snapshots previously written with `TFRecordWriter`.
class TFRecordReader : public Reader {
 public:
  TFRecordReader(const std::string& filename, const string& compression_type,
                 const DataTypeVector& dtypes);

  Status ReadTensors(std::vector<Tensor>* read_tensors) override;

  ~TFRecordReader() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTh mht_2(mht_2_v, 420, "", "./tensorflow/core/data/snapshot_utils.h", "~TFRecordReader");
}

 protected:
  Status Initialize(Env* env) override;

 private:
  std::string filename_;
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::RecordReader> record_reader_;
  uint64 offset_;

  const string compression_type_;
  const DataTypeVector dtypes_;
};

// Reads snapshots previously written with `CustomWriter`.
class CustomReader : public Reader {
 public:
  // The reader input buffer size is deliberately large because the input reader
  // will throw an error if the compressed block length cannot fit in the input
  // buffer.
  static constexpr const int64_t kSnappyReaderInputBufferSizeBytes =
      1 << 30;  // 1 GiB
  // TODO(b/148804377): Set this in a smarter fashion.
  static constexpr const int64_t kSnappyReaderOutputBufferSizeBytes =
      32 << 20;  // 32 MiB
  static constexpr const size_t kHeaderSize = sizeof(uint64);

  static constexpr const char* const kClassName = "SnapshotReader";
  static constexpr const char* const kReadString = "ReadString";
  static constexpr const char* const kReadCord = "ReadCord";
  static constexpr const char* const kSeparator = "::";

  CustomReader(const std::string& filename, const string& compression_type,
               const int version, const DataTypeVector& dtypes);

  Status ReadTensors(std::vector<Tensor>* read_tensors) override;

  ~CustomReader() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTh mht_3(mht_3_v, 461, "", "./tensorflow/core/data/snapshot_utils.h", "~CustomReader");
}

 protected:
  Status Initialize(Env* env) override;

 private:
  Status ReadTensorsV0(std::vector<Tensor>* read_tensors);

  Status SnappyUncompress(
      const experimental::SnapshotTensorMetadata* metadata,
      std::vector<Tensor>* simple_tensors,
      std::vector<std::pair<std::unique_ptr<char[]>, size_t>>*
          tensor_proto_strs);

  Status ReadRecord(tstring* record);

#if defined(TF_CORD_SUPPORT)
  Status ReadRecord(absl::Cord* record);
#endif

  std::string filename_;
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::InputStreamInterface> input_stream_;
  const string compression_type_;
  const int version_;
  const DataTypeVector dtypes_;
  int num_simple_ = 0;
  int num_complex_ = 0;
  std::vector<bool> simple_tensor_mask_;  // true for simple, false for complex.
};

// Writes snapshot metadata to the given directory.
Status WriteMetadataFile(Env* env, const string& dir,
                         const experimental::SnapshotMetadataRecord* metadata);

// Reads snapshot metadata from the given directory.
Status ReadMetadataFile(Env* env, const string& dir,
                        experimental::SnapshotMetadataRecord* metadata,
                        bool* file_exists);

// Writes a dataset graph to the given directory.
Status DumpDatasetGraph(Env* env, const std::string& path, uint64 hash,
                        const GraphDef* graph);

Status DetermineOpState(const std::string& mode_string, bool file_exists,
                        const experimental::SnapshotMetadataRecord* metadata,
                        const uint64 pending_snapshot_expiry_seconds,
                        Mode* mode);

// Represents a dataset element or EOF.
struct ElementOrEOF {
  std::vector<Tensor> value;
  bool end_of_sequence = false;
};

// AsyncWriter provides API for asynchronously writing dataset elements
// (each represented as a vector of tensors) to a file.
//
// The expected use of this API is:
//
// std::unique_ptr<AsyncWriter> writer = absl_make_unique<AsyncWriter>(...);
//
// while (data_available()) {
//   std::vector<Tensor> data = read_data()
//   writer->Write(data);
// }
// writer->SignalEOF();
// writer = nullptr;  // This will block until writes are flushed.
class AsyncWriter {
 public:
  explicit AsyncWriter(Env* env, int64_t file_index,
                       const std::string& shard_directory, uint64 checkpoint_id,
                       const std::string& compression, int64_t version,
                       const DataTypeVector& output_types,
                       std::function<void(Status)> done);

  // Writes the given tensors. The method is non-blocking and returns without
  // waiting for the element to be written.
  void Write(const std::vector<Tensor>& tensors) TF_LOCKS_EXCLUDED(mu_);

  // Signals the end of input. The method is non-blocking and returns without
  // waiting for the writer to be closed.
  void SignalEOF() TF_LOCKS_EXCLUDED(mu_);

 private:
  void Consume(ElementOrEOF* be) TF_LOCKS_EXCLUDED(mu_);
  bool ElementAvailable() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status WriterThread(Env* env, const std::string& shard_directory,
                      uint64 checkpoint_id, const std::string& compression,
                      int64_t version, DataTypeVector output_types);

  mutex mu_;
  std::deque<ElementOrEOF> deque_ TF_GUARDED_BY(mu_);

  // This has to be last. During destruction, we need to make sure that the
  // Thread object is destroyed first as its destructor blocks on thread
  // completion. If there are other member variables after this, they may get
  // destroyed first before the thread finishes, potentially causing the
  // thread to access invalid memory.
  std::unique_ptr<Thread> thread_;
};

}  // namespace snapshot_util
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SNAPSHOT_UTILS_H_
