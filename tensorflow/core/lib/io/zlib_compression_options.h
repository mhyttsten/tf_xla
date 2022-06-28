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

#ifndef TENSORFLOW_CORE_LIB_IO_ZLIB_COMPRESSION_OPTIONS_H_
#define TENSORFLOW_CORE_LIB_IO_ZLIB_COMPRESSION_OPTIONS_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_compression_optionsDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_compression_optionsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_compression_optionsDTh() {
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


#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace io {

class ZlibCompressionOptions {
 public:
  ZlibCompressionOptions();

  static ZlibCompressionOptions DEFAULT();
  static ZlibCompressionOptions RAW();
  static ZlibCompressionOptions GZIP();

  // Defaults to Z_NO_FLUSH
  int8 flush_mode;

  // Size of the buffer used for caching the data read from source file.
  int64_t input_buffer_size = 256 << 10;

  // Size of the sink buffer where the compressed/decompressed data produced by
  // zlib is cached.
  int64_t output_buffer_size = 256 << 10;

  // The window_bits parameter is the base two logarithm of the window size
  // (the size of the history buffer). Larger values of buffer size result in
  // better compression at the expense of memory usage.
  //
  // Accepted values:
  //
  // 8..15:
  // Normal deflate with zlib header and checksum.
  //
  // -8..-15:
  // Negative values can be used for raw deflate/inflate. In this case,
  // -window_bits determines the window size. deflate() will then generate raw
  // deflate data  with no zlib header or trailer, and will not compute an
  // adler32 check value. inflate() will then process raw deflate data, not
  // looking for a zlib or gzip header, not generating a check value, and not
  // looking for any check values for comparison at the end of the stream.
  //
  // 16 + [8..15]:
  // window_bits can also be greater than 15 for optional gzip encoding. Add 16
  // to window_bits to write a simple gzip header and trailer around the
  // compressed data instead of a zlib wrapper. The gzip header will have no
  // file name, no extra data, no comment, no modification time (set to zero),
  // no header crc, and the operating system will be set to 255 (unknown). If a
  // gzip stream is being written, strm->adler is a crc32 instead of an adler32.
  //
  // 0:
  // window_bits can also be zero to request that inflate use the window size
  // in the zlib header of the compressed stream.
  //
  // While inflating, window_bits must be greater than or equal to the
  // window_bits value provided used while compressing. If a compressed stream
  // with a larger window size is given as input, inflate() will return with the
  // error code Z_DATA_ERROR instead of trying to allocate a larger window.
  //
  // Defaults to MAX_WBITS
  int8 window_bits;

  // From the zlib manual (http://www.zlib.net/manual.html):
  // The compression level must be Z_DEFAULT_COMPRESSION, or between 0 and 9:
  // 1 gives best speed, 9 gives best compression, 0 gives no compression at all
  // (the input data is simply copied a block at a time). Z_DEFAULT_COMPRESSION
  // requests a default compromise between speed and compression (currently
  // equivalent to level 6).
  int8 compression_level;

  // Only Z_DEFLATED is supported at this time.
  int8 compression_method;

  // From the zlib manual (http://www.zlib.net/manual.html):
  // The mem_level parameter specifies how much memory should be allocated for
  // the internal compression state. mem_level=1 uses minimum memory but is slow
  // and reduces compression ratio; mem_level=9 uses maximum memory for optimal
  // speed. The default value is 8.
  int8 mem_level = 9;

  // From the zlib manual (http://www.zlib.net/manual.html):
  // The strategy parameter is used to tune the compression algorithm. Use the
  // value Z_DEFAULT_STRATEGY for normal data, Z_FILTERED for data produced by
  // a filter (or predictor), Z_HUFFMAN_ONLY to force Huffman encoding only
  // (no string match), or Z_RLE to limit match distances to one
  // (run-length encoding). Filtered data consists mostly of small values with
  // a somewhat random distribution. In this case, the compression algorithm is
  // tuned to compress them better. The effect of Z_FILTERED is to force more
  // Huffman coding and less string matching; it is somewhat intermediate
  // between Z_DEFAULT_STRATEGY and Z_HUFFMAN_ONLY. Z_RLE is designed to be
  // almost as fast as Z_HUFFMAN_ONLY, but give better compression for
  // PNG image data. The strategy parameter only affects the compression ratio
  // but not the correctness of the compressed output even if it is not set
  // appropriately. Z_FIXED prevents the use of dynamic Huffman codes, allowing
  // for a simpler decoder for special applications.
  int8 compression_strategy;

  // When this is set to true and we are unable to find the header to correctly
  // decompress a file, we return an error when `ReadNBytes` is called instead
  // of CHECK-failing. Defaults to false (i.e. CHECK-failing).
  //
  // This option is ignored for `ZlibOutputBuffer`.
  bool soft_fail_on_error = false;  // NOLINT
};

inline ZlibCompressionOptions ZlibCompressionOptions::DEFAULT() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_compression_optionsDTh mht_0(mht_0_v, 291, "", "./tensorflow/core/lib/io/zlib_compression_options.h", "ZlibCompressionOptions::DEFAULT");

  return ZlibCompressionOptions();
}

inline ZlibCompressionOptions ZlibCompressionOptions::RAW() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_compression_optionsDTh mht_1(mht_1_v, 298, "", "./tensorflow/core/lib/io/zlib_compression_options.h", "ZlibCompressionOptions::RAW");

  ZlibCompressionOptions options = ZlibCompressionOptions();
  options.window_bits = -options.window_bits;
  return options;
}

inline ZlibCompressionOptions ZlibCompressionOptions::GZIP() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_compression_optionsDTh mht_2(mht_2_v, 307, "", "./tensorflow/core/lib/io/zlib_compression_options.h", "ZlibCompressionOptions::GZIP");

  ZlibCompressionOptions options = ZlibCompressionOptions();
  options.window_bits = options.window_bits + 16;
  return options;
}

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_ZLIB_COMPRESSION_OPTIONS_H_
