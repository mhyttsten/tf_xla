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
class MHTracer_DTPStensorflowPScorePSkernelsPSdecode_compressed_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_compressed_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdecode_compressed_opDTcc() {
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

// See docs in ../ops/parse_ops.cc.

#include <algorithm>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"

namespace tensorflow {
namespace {
// Wrap memory buffer into InputStreamInterface
class MemoryInputStream : public io::InputStreamInterface {
 public:
  explicit MemoryInputStream(const char* buffer, size_t length)
      : buf_(buffer), len_(length), pos_(0) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_compressed_opDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/decode_compressed_op.cc", "MemoryInputStream");
}

  ~MemoryInputStream() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_compressed_opDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/kernels/decode_compressed_op.cc", "~MemoryInputStream");
}

  Status ReadNBytes(int64_t bytes_to_read, tstring* result) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_compressed_opDTcc mht_2(mht_2_v, 213, "", "./tensorflow/core/kernels/decode_compressed_op.cc", "ReadNBytes");

    result->clear();
    if (bytes_to_read < 0) {
      return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                     bytes_to_read);
    }
    int64_t bytes = bytes_to_read;
    Status s = Status::OK();
    if (pos_ + bytes_to_read > len_) {
      bytes = len_ - pos_;
      s = errors::OutOfRange("reached end of file");
    }
    if (bytes > 0) {
      result->resize(bytes);
      memcpy(&(*result)[0], &buf_[pos_], bytes);
      pos_ += bytes;
    }
    return s;
  }

  int64_t Tell() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_compressed_opDTcc mht_3(mht_3_v, 236, "", "./tensorflow/core/kernels/decode_compressed_op.cc", "Tell");
 return pos_; }

  Status Reset() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_compressed_opDTcc mht_4(mht_4_v, 241, "", "./tensorflow/core/kernels/decode_compressed_op.cc", "Reset");

    pos_ = 0;
    return Status::OK();
  }

 private:
  const char* buf_;  // Not owned.
  int64_t len_;
  int64_t pos_ = 0;  // Tracks where we are in the file.
};
}  // namespace

class DecodeCompressedOp : public OpKernel {
 public:
  explicit DecodeCompressedOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_compressed_opDTcc mht_5(mht_5_v, 259, "", "./tensorflow/core/kernels/decode_compressed_op.cc", "DecodeCompressedOp");

    OP_REQUIRES_OK(context,
                   context->GetAttr("compression_type", &compression_type_));
    OP_REQUIRES(context,
                (compression_type_.empty() || compression_type_ == "ZLIB" ||
                 compression_type_ == "GZIP"),
                errors::InvalidArgument(
                    "Only ZLIB, GZIP or NONE are supported compressions"));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_compressed_opDTcc mht_6(mht_6_v, 272, "", "./tensorflow/core/kernels/decode_compressed_op.cc", "Compute");

    const Tensor* bytes_tensor;
    OP_REQUIRES_OK(context, context->input("bytes", &bytes_tensor));
    const auto& bytes_flat = bytes_tensor->flat<tstring>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", bytes_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();
    if (compression_type_.empty()) {
      for (int64_t i = 0; i < bytes_flat.size(); i++) {
        output_flat(i) = bytes_flat(i);
      }
    } else {
      const io::ZlibCompressionOptions zlib_options =
          compression_type_ == "ZLIB" ? io::ZlibCompressionOptions::DEFAULT()
                                      : io::ZlibCompressionOptions::GZIP();
      for (int64_t i = 0; i < bytes_flat.size(); i++) {
        std::unique_ptr<MemoryInputStream> input_stream(
            new MemoryInputStream(bytes_flat(i).data(), bytes_flat(i).size()));
        std::unique_ptr<io::ZlibInputStream> zlib_stream(
            new io::ZlibInputStream(
                input_stream.get(), static_cast<size_t>(kBufferSize),
                static_cast<size_t>(kBufferSize), zlib_options));
        tstring output_string;
        Status s = zlib_stream->ReadNBytes(INT_MAX, &output_string);
        OP_REQUIRES(context, (s.ok() || errors::IsOutOfRange(s)), s);
        output_flat(i) = std::move(output_string);
      }
    }
  }

 private:
  enum { kBufferSize = 256 << 10 /* 256 kB */ };
  string compression_type_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeCompressed").Device(DEVICE_CPU),
                        DecodeCompressedOp)

}  // namespace tensorflow
