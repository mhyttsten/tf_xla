/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBJPEG_DECODER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBJPEG_DECODER_H_
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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh() {
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


#include <memory.h>

#include <csetjmp>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_common.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_decompress_buffered_struct.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libc_handle.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_handle.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

// Extracts the expected size of `jpeg_decompress_struct` from the "struct
// mismatch" error message and stores it in `expected_size`. Returns status code
// kTfLiteOk if the extraction was successful, error otherwise.
Status ExtractSizeFromErrorMessage(const std::string& error_message,
                                   size_t& expected_size);

class LibjpegDecoder {
 public:
  // The maximum height allowed for the decoded image. Any attempt to call
  // DecodeImage for an image with height or width over the allowed limits will
  // fail.
  // The size is define to 10,000 lines.
  static const size_t kMaxImageHeight;
  // The maximum width allowed for the decoded image. Any attempt to call
  // DecodeImage for an image with height or width over the allowed limits will
  // fail.
  // The size is define to 10,000 pixels per line.
  static const size_t kMaxImageWidth;

  // Creates and initialises the decoder.
  // Dynamically loads libjpeg (into handle_) and sets the expected size for
  // `jpeg_decompress_struct` (in expected_size_for_decompress_struct_). Returns
  // an initalised instance of decoder if successful, else returns nullptr.
  // Stores initialisation status in status.
  static std::unique_ptr<LibjpegDecoder> Create(Status& status);
  Status DecodeImage(const tflite::StringRef& encoded,
                     const JpegHeader& expected_image_dimensions,
                     unsigned char* decoded, const size_t& decoded_size) const;

 private:
  explicit LibjpegDecoder(LibCHandle libc_handle)
      : libc_handle_(std::move(libc_handle)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh mht_0(mht_0_v, 241, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h", "LibjpegDecoder");
}
  // Wraps all objects required for using the libjpeg library.
  // This is to avoid stack-allocating these variables in the function that
  // calls setjmp().
  class Impl {
   public:
    explicit Impl(size_t decompress_struct_size, const LibjpegHandle* handle);
    ~Impl() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh mht_1(mht_1_v, 251, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h", "~Impl");
 jpeg_destroy_decompress(); }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&& other) = delete;
    Impl& operator=(Impl&& other) = delete;

    // Wrapping calls to LibjpegHandle functions in Run and RunAndSetStatus.
    TfLiteStatus jpeg_CreateDecompress(int version, size_t struct_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh mht_2(mht_2_v, 261, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h", "jpeg_CreateDecompress");

      // Note: It is safe to call jpeg_destroy_decompress even if the
      // corresponding call to create_jpeg_decompress fails. See the end of
      // section "Compression details" in
      // https://www.freedesktop.org/wiki/Software/libjpeg/.
      safe_to_invoke_destroy_decompress_ = true;
      return Run(&LibjpegHandle::jpeg_create_decompress_, version, struct_size);
    }
    TfLiteStatus jpeg_stdio_src(FILE* infile) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh mht_3(mht_3_v, 272, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h", "jpeg_stdio_src");

      return Run(&LibjpegHandle::jpeg_stdio_src_, infile);
    }

    TfLiteStatus jpeg_read_header(int& read_header_result,
                                  boolean require_image) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh mht_4(mht_4_v, 280, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h", "jpeg_read_header");

      return RunAndSetResult(&LibjpegHandle::jpeg_read_header_,
                             &read_header_result, require_image);
    }

    TfLiteStatus jpeg_start_decompress(boolean& start_decompress_result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh mht_5(mht_5_v, 288, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h", "jpeg_start_decompress");

      return RunAndSetResult(&LibjpegHandle::jpeg_start_decompress_,
                             &start_decompress_result);
    }
    TfLiteStatus jpeg_read_scanlines(unsigned int& read_scanlines_result,
                                     JSAMPARRAY scanlines,
                                     JDIMENSION max_lines) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh mht_6(mht_6_v, 297, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h", "jpeg_read_scanlines");

      return RunAndSetResult(&LibjpegHandle::jpeg_read_scanlines_,
                             &read_scanlines_result, scanlines, max_lines);
    }
    TfLiteStatus jpeg_finish_decompress(boolean& finish_decompress_result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh mht_7(mht_7_v, 304, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h", "jpeg_finish_decompress");

      return RunAndSetResult(&LibjpegHandle::jpeg_finish_decompress_,
                             &finish_decompress_result);
    }
    TfLiteStatus jpeg_destroy_decompress() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh mht_8(mht_8_v, 311, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h", "jpeg_destroy_decompress");

      if (safe_to_invoke_destroy_decompress_) {
        safe_to_invoke_destroy_decompress_ = false;
        return Run(&LibjpegHandle::jpeg_destroy_decompress_);
      }
      return kTfLiteOk;
    }

    // Status from the libjpeg layer that is to be returned to the caller.
    Status status() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh mht_9(mht_9_v, 323, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h", "status");
 return status_; }

   private:
    // Delegates to one of the LibjpegHandle::jpeg_* methods.
    // This is to restrict the call to setjmp() to a stack frame free from
    // stack allocated C++ variables. The type of f is T
    // (LibjpegHandle::*f)(Args...), for some T. All args must be
    // pointers/references/primitive types. Since we use a
    // non-suspending JPEG encoded data source, return value from a
    // LibjpegHandle::jpeg_* methods is not required by client and hence
    // discarded. Returns an Ok status if the execution was successful, error
    // otherwise.
    template <typename Fn, typename... Args>
    TfLiteStatus Run(Fn f, Args... args) {
      // Note(1): C++ variables local to this function should not be stack
      // allocated
      // and should be passed as pointers or references. Using setjmp/longjmp
      // with stack allocated C++ objects that have non-trivial destructors can
      // lead to undefined behaviour.
      // https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=88046492
      // All such variables whose scope contains calls to libjpeg (that may do a
      // longjmp) should be passed in as arguments.
      //
      // Note(2): All other variables local to this function that need to be
      // accessed after longjmp() returns control to this function, should be
      // volatile-qualified.
      // After invoking longjump(), non-volatile local variables should not be
      // accessed for two reasons:
      // - their values may be indeterminate. According to the C standard, if
      // the variable's value has changed between setjmp() and longjmp(), their
      // value is considered indeterminate, and accessing them is undefined
      // behaviour.
      // https://wiki.sei.cmu.edu/confluence/display/c/MSC22-C.+Use+the+setjmp%28%29%2C+longjmp%28%29+facility+securely
      // - the register storing such variables might be clobbered. Even if the
      // variable remains unchanged between setjmp() and longjmp(), the stack
      // slot for the variable may get incorrectly clobbered. This is a known
      // LLVM bug: https://bugs.llvm.org/show_bug.cgi?id=21183
      if (setjmp(env_)) return kTfLiteError;
      (handle_->*f)(cinfo_.get(), args...);
      return kTfLiteOk;
    }
    // Extension of the Run method for non-void JPEG calls when we need to
    // collect the returned value.
    // See Run comments above for details.
    template <
        typename Fn, typename... Args,
        typename ResultType = typename std::result_of_t<Fn>,
        typename = typename std::enable_if<!std::is_void<ResultType>::value> >
    TfLiteStatus RunAndSetResult(Fn f, ResultType* result, Args... args) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTh mht_10(mht_10_v, 374, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h", "RunAndSetResult");

      if (setjmp(env_)) return kTfLiteError;
      *result = (handle_->*f)(cinfo_.get(), args...);
      return kTfLiteOk;
    }
    // Size of `jpeg_decompress_struct` as expected by libjpeg library.
    size_t decompress_struct_size_;
    const LibjpegHandle* handle_;
    // Using a buffered struct for `jpeg_decompress_struct` as the size expected
    // by libjpeg can be different from the size of the compiled struct. See
    // go/libjpeg-android. Note: Since we resize the struct, accessing some of
    // the fields of this struct may lead to undefined behaviour. For
    // decompression, only the fields within `jpeg_common_fields` are required
    // viz. error manager(`err`) and client data(`client_data`). This code
    // limits its usage to these two fields and we recommend future contributors
    // to not access fields beyond `jpeg_common_fields`.
    JpegDecompressBufferedStruct cinfo_;
    struct jpeg_error_mgr jerr_;
    // Stores the information of the calling environment which can be restored
    // later. Libjpeg aborts the program in case of any errors by using longjmp
    // and then calling exit(). The only way to avoid this, is to transfer the
    // control flow to the caller by using setjmp/longjmp.
    jmp_buf env_;
    static void ErrorExit(j_common_ptr cinfo);
    // Calls to jpeg_create_decompress and jpeg_destroy_decompress need to be
    // paired. This flag indicates if it's safe to invoke
    // jpeg_destroy_decompress.
    bool safe_to_invoke_destroy_decompress_ = false;
    // Status of the most recent execution of a LibjpegHandle::jpeg_* method
    // invoked using Run or RunAndSetResult.
    Status status_;
  };
  // Size of `jpeg_decompress_struct` as expected by the libjpeg dynamic
  // library. The expected size is different from the size of the compiled
  // struct on some Android Devices. See go/libjpeg-android.
  size_t expected_size_for_decompress_struct_;
  // Handle to the Libjpeg dynamic library.
  std::unique_ptr<LibjpegHandle> libjpeg_handle_;
  // Handle to the LibC dynamic library.
  LibCHandle libc_handle_;
};

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBJPEG_DECODER_H_
