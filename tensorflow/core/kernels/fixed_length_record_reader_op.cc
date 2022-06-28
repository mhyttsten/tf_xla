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
class MHTracer_DTPStensorflowPScorePSkernelsPSfixed_length_record_reader_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfixed_length_record_reader_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfixed_length_record_reader_opDTcc() {
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

// See docs in ../ops/io_ops.cc.

#include <memory>
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

// In the constructor hop_bytes_ is set to record_bytes_ if it was 0,
// so that we will always "hop" after each read (except first).
class FixedLengthRecordReader : public ReaderBase {
 public:
  FixedLengthRecordReader(const string& node_name, int64_t header_bytes,
                          int64_t record_bytes, int64_t footer_bytes,
                          int64_t hop_bytes, const string& encoding, Env* env)
      : ReaderBase(
            strings::StrCat("FixedLengthRecordReader '", node_name, "'")),
        header_bytes_(header_bytes),
        record_bytes_(record_bytes),
        footer_bytes_(footer_bytes),
        hop_bytes_(hop_bytes == 0 ? record_bytes : hop_bytes),
        env_(env),
        record_number_(0),
        encoding_(encoding) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   mht_0_v.push_back("encoding: \"" + encoding + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSfixed_length_record_reader_opDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/kernels/fixed_length_record_reader_op.cc", "FixedLengthRecordReader");
}

  // On success:
  // * buffered_inputstream_ != nullptr,
  // * buffered_inputstream_->Tell() == header_bytes_
  Status OnWorkStartedLocked() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfixed_length_record_reader_opDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/kernels/fixed_length_record_reader_op.cc", "OnWorkStartedLocked");

    record_number_ = 0;

    lookahead_cache_.clear();

    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(current_work(), &file_));
    if (encoding_ == "ZLIB" || encoding_ == "GZIP") {
      const io::ZlibCompressionOptions zlib_options =
          encoding_ == "ZLIB" ? io::ZlibCompressionOptions::DEFAULT()
                              : io::ZlibCompressionOptions::GZIP();
      file_stream_.reset(new io::RandomAccessInputStream(file_.get()));
      buffered_inputstream_.reset(new io::ZlibInputStream(
          file_stream_.get(), static_cast<size_t>(kBufferSize),
          static_cast<size_t>(kBufferSize), zlib_options));
    } else {
      buffered_inputstream_.reset(
          new io::BufferedInputStream(file_.get(), kBufferSize));
    }
    // header_bytes_ is always skipped.
    TF_RETURN_IF_ERROR(buffered_inputstream_->SkipNBytes(header_bytes_));

    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfixed_length_record_reader_opDTcc mht_2(mht_2_v, 252, "", "./tensorflow/core/kernels/fixed_length_record_reader_op.cc", "OnWorkFinishedLocked");

    buffered_inputstream_.reset(nullptr);
    return Status::OK();
  }

  Status ReadLocked(tstring* key, tstring* value, bool* produced,
                    bool* at_end) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfixed_length_record_reader_opDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/kernels/fixed_length_record_reader_op.cc", "ReadLocked");

    // We will always "hop" the hop_bytes_ except the first record
    // where record_number_ == 0
    if (record_number_ != 0) {
      if (hop_bytes_ <= lookahead_cache_.size()) {
        // If hop_bytes_ is smaller than the cached data we skip the
        // hop_bytes_ from the cache.
        lookahead_cache_ = lookahead_cache_.substr(hop_bytes_);
      } else {
        // If hop_bytes_ is larger than the cached data, we clean up
        // the cache, then skip hop_bytes_ - cache_size from the file
        // as the cache_size has been skipped through cache.
        int64_t cache_size = lookahead_cache_.size();
        lookahead_cache_.clear();
        Status s = buffered_inputstream_->SkipNBytes(hop_bytes_ - cache_size);
        if (!s.ok()) {
          if (!errors::IsOutOfRange(s)) {
            return s;
          }
          *at_end = true;
          return Status::OK();
        }
      }
    }

    // Fill up lookahead_cache_ to record_bytes_ + footer_bytes_
    int bytes_to_read = record_bytes_ + footer_bytes_ - lookahead_cache_.size();
    Status s = buffered_inputstream_->ReadNBytes(bytes_to_read, value);
    if (!s.ok()) {
      value->clear();
      if (!errors::IsOutOfRange(s)) {
        return s;
      }
      *at_end = true;
      return Status::OK();
    }
    lookahead_cache_.append(*value, 0, bytes_to_read);
    value->clear();

    // Copy first record_bytes_ from cache to value
    *value = lookahead_cache_.substr(0, record_bytes_);

    *key = strings::StrCat(current_work(), ":", record_number_);
    *produced = true;
    ++record_number_;

    return Status::OK();
  }

  Status ResetLocked() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfixed_length_record_reader_opDTcc mht_4(mht_4_v, 313, "", "./tensorflow/core/kernels/fixed_length_record_reader_op.cc", "ResetLocked");

    record_number_ = 0;
    buffered_inputstream_.reset(nullptr);
    lookahead_cache_.clear();
    return ReaderBase::ResetLocked();
  }

  // TODO(josh11b): Implement serializing and restoring the state.

 private:
  enum { kBufferSize = 256 << 10 /* 256 kB */ };
  const int64_t header_bytes_;
  const int64_t record_bytes_;
  const int64_t footer_bytes_;
  const int64_t hop_bytes_;
  // The purpose of lookahead_cache_ is to allows "one-pass" processing
  // without revisit previous processed data of the stream. This is needed
  // because certain compression like zlib does not allow random access
  // or even obtain the uncompressed stream size before hand.
  // The max size of the lookahead_cache_ could be
  // record_bytes_ + footer_bytes_
  string lookahead_cache_;
  Env* const env_;
  int64_t record_number_;
  string encoding_;
  // must outlive buffered_inputstream_
  std::unique_ptr<RandomAccessFile> file_;
  // must outlive buffered_inputstream_
  std::unique_ptr<io::RandomAccessInputStream> file_stream_;
  std::unique_ptr<io::InputStreamInterface> buffered_inputstream_;
};

class FixedLengthRecordReaderOp : public ReaderOpKernel {
 public:
  explicit FixedLengthRecordReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfixed_length_record_reader_opDTcc mht_5(mht_5_v, 351, "", "./tensorflow/core/kernels/fixed_length_record_reader_op.cc", "FixedLengthRecordReaderOp");

    int64_t header_bytes = -1, record_bytes = -1, footer_bytes = -1,
            hop_bytes = -1;
    OP_REQUIRES_OK(context, context->GetAttr("header_bytes", &header_bytes));
    OP_REQUIRES_OK(context, context->GetAttr("record_bytes", &record_bytes));
    OP_REQUIRES_OK(context, context->GetAttr("footer_bytes", &footer_bytes));
    OP_REQUIRES_OK(context, context->GetAttr("hop_bytes", &hop_bytes));
    OP_REQUIRES(context, header_bytes >= 0,
                errors::InvalidArgument("header_bytes must be >= 0 not ",
                                        header_bytes));
    OP_REQUIRES(context, record_bytes >= 0,
                errors::InvalidArgument("record_bytes must be >= 0 not ",
                                        record_bytes));
    OP_REQUIRES(context, footer_bytes >= 0,
                errors::InvalidArgument("footer_bytes must be >= 0 not ",
                                        footer_bytes));
    OP_REQUIRES(
        context, hop_bytes >= 0,
        errors::InvalidArgument("hop_bytes must be >= 0 not ", hop_bytes));
    Env* env = context->env();
    string encoding;
    OP_REQUIRES_OK(context, context->GetAttr("encoding", &encoding));
    SetReaderFactory([this, header_bytes, record_bytes, footer_bytes, hop_bytes,
                      encoding, env]() {
      return new FixedLengthRecordReader(name(), header_bytes, record_bytes,
                                         footer_bytes, hop_bytes, encoding,
                                         env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("FixedLengthRecordReader").Device(DEVICE_CPU),
                        FixedLengthRecordReaderOp);
REGISTER_KERNEL_BUILDER(Name("FixedLengthRecordReaderV2").Device(DEVICE_CPU),
                        FixedLengthRecordReaderOp);

}  // namespace tensorflow
