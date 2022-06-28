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
class MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc() {
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

// This file implements a memory destination for libjpeg
// The design is very similar to jdatadst.c in libjpeg
// These functions are not meant to be used directly, see jpeg_mem.h instead.
// We are filling out stubs required by jpeglib, those stubs are private to
// the implementation, we are just making available JPGMemSrc, JPGMemDest

#include "tensorflow/core/lib/jpeg/jpeg_handle.h"

#include <setjmp.h>
#include <stddef.h>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace jpeg {

void CatchError(j_common_ptr cinfo) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/lib/jpeg/jpeg_handle.cc", "CatchError");

  (*cinfo->err->output_message)(cinfo);
  jmp_buf *jpeg_jmpbuf = reinterpret_cast<jmp_buf *>(cinfo->client_data);
  jpeg_destroy(cinfo);
  longjmp(*jpeg_jmpbuf, 1);
}

// *****************************************************************************
// *****************************************************************************
// *****************************************************************************
// Destination functions

// -----------------------------------------------------------------------------
void MemInitDestination(j_compress_ptr cinfo) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/lib/jpeg/jpeg_handle.cc", "MemInitDestination");

  MemDestMgr *dest = reinterpret_cast<MemDestMgr *>(cinfo->dest);
  VLOG(1) << "Initializing buffer=" << dest->bufsize << " bytes";
  dest->pub.next_output_byte = dest->buffer;
  dest->pub.free_in_buffer = dest->bufsize;
  dest->datacount = 0;
  if (dest->dest) {
    dest->dest->clear();
  }
}

// -----------------------------------------------------------------------------
boolean MemEmptyOutputBuffer(j_compress_ptr cinfo) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/lib/jpeg/jpeg_handle.cc", "MemEmptyOutputBuffer");

  MemDestMgr *dest = reinterpret_cast<MemDestMgr *>(cinfo->dest);
  VLOG(1) << "Writing " << dest->bufsize << " bytes";
  if (dest->dest) {
    dest->dest->append(reinterpret_cast<char *>(dest->buffer), dest->bufsize);
  }
  dest->pub.next_output_byte = dest->buffer;
  dest->pub.free_in_buffer = dest->bufsize;
  return TRUE;
}

// -----------------------------------------------------------------------------
void MemTermDestination(j_compress_ptr cinfo) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc mht_3(mht_3_v, 247, "", "./tensorflow/core/lib/jpeg/jpeg_handle.cc", "MemTermDestination");

  MemDestMgr *dest = reinterpret_cast<MemDestMgr *>(cinfo->dest);
  VLOG(1) << "Writing " << dest->bufsize - dest->pub.free_in_buffer << " bytes";
  if (dest->dest) {
    dest->dest->append(reinterpret_cast<char *>(dest->buffer),
                       dest->bufsize - dest->pub.free_in_buffer);
    VLOG(1) << "Total size= " << dest->dest->size();
  }
  dest->datacount = dest->bufsize - dest->pub.free_in_buffer;
}

// -----------------------------------------------------------------------------
void SetDest(j_compress_ptr cinfo, void *buffer, int bufsize) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/lib/jpeg/jpeg_handle.cc", "SetDest");

  SetDest(cinfo, buffer, bufsize, nullptr);
}

// -----------------------------------------------------------------------------
void SetDest(j_compress_ptr cinfo, void *buffer, int bufsize,
             tstring *destination) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc mht_5(mht_5_v, 271, "", "./tensorflow/core/lib/jpeg/jpeg_handle.cc", "SetDest");

  MemDestMgr *dest;
  if (cinfo->dest == nullptr) {
    cinfo->dest = reinterpret_cast<struct jpeg_destination_mgr *>(
        (*cinfo->mem->alloc_small)(reinterpret_cast<j_common_ptr>(cinfo),
                                   JPOOL_PERMANENT, sizeof(MemDestMgr)));
  }

  dest = reinterpret_cast<MemDestMgr *>(cinfo->dest);
  dest->bufsize = bufsize;
  dest->buffer = static_cast<JOCTET *>(buffer);
  dest->dest = destination;
  dest->pub.init_destination = MemInitDestination;
  dest->pub.empty_output_buffer = MemEmptyOutputBuffer;
  dest->pub.term_destination = MemTermDestination;
}

// *****************************************************************************
// *****************************************************************************
// *****************************************************************************
// Source functions

// -----------------------------------------------------------------------------
void MemInitSource(j_decompress_ptr cinfo) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc mht_6(mht_6_v, 297, "", "./tensorflow/core/lib/jpeg/jpeg_handle.cc", "MemInitSource");

  MemSourceMgr *src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  src->pub.next_input_byte = src->data;
  src->pub.bytes_in_buffer = src->datasize;
}

// -----------------------------------------------------------------------------
// We emulate the same error-handling as fill_input_buffer() from jdatasrc.c,
// for coherency's sake.
boolean MemFillInputBuffer(j_decompress_ptr cinfo) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc mht_7(mht_7_v, 309, "", "./tensorflow/core/lib/jpeg/jpeg_handle.cc", "MemFillInputBuffer");

  static const JOCTET kEOIBuffer[2] = {0xff, JPEG_EOI};
  MemSourceMgr *src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  if (src->pub.bytes_in_buffer == 0 && src->pub.next_input_byte == src->data) {
    // empty file -> treated as an error.
    ERREXIT(cinfo, JERR_INPUT_EMPTY);
    return FALSE;
  } else if (src->pub.bytes_in_buffer) {
    // if there's still some data left, it's probably corrupted
    return src->try_recover_truncated_jpeg ? TRUE : FALSE;
  } else if (src->pub.next_input_byte != kEOIBuffer &&
             src->try_recover_truncated_jpeg) {
    // In an attempt to recover truncated files, we insert a fake EOI
    WARNMS(cinfo, JWRN_JPEG_EOF);
    src->pub.next_input_byte = kEOIBuffer;
    src->pub.bytes_in_buffer = 2;
    return TRUE;
  } else {
    // We already inserted a fake EOI and it wasn't enough, so this time
    // it's really an error.
    ERREXIT(cinfo, JERR_FILE_READ);
    return FALSE;
  }
}

// -----------------------------------------------------------------------------
void MemTermSource(j_decompress_ptr cinfo) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc mht_8(mht_8_v, 338, "", "./tensorflow/core/lib/jpeg/jpeg_handle.cc", "MemTermSource");
}

// -----------------------------------------------------------------------------
void MemSkipInputData(j_decompress_ptr cinfo, long jump) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc mht_9(mht_9_v, 344, "", "./tensorflow/core/lib/jpeg/jpeg_handle.cc", "MemSkipInputData");

  MemSourceMgr *src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  if (jump < 0) {
    return;
  }
  if (jump > src->pub.bytes_in_buffer) {
    src->pub.bytes_in_buffer = 0;
    (void)MemFillInputBuffer(cinfo);  // warn with a fake EOI or error
  } else {
    src->pub.bytes_in_buffer -= jump;
    src->pub.next_input_byte += jump;
  }
}

// -----------------------------------------------------------------------------
void SetSrc(j_decompress_ptr cinfo, const void *data,
            unsigned long int datasize, bool try_recover_truncated_jpeg) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_handleDTcc mht_10(mht_10_v, 363, "", "./tensorflow/core/lib/jpeg/jpeg_handle.cc", "SetSrc");

  MemSourceMgr *src;

  cinfo->src = reinterpret_cast<struct jpeg_source_mgr *>(
      (*cinfo->mem->alloc_small)(reinterpret_cast<j_common_ptr>(cinfo),
                                 JPOOL_PERMANENT, sizeof(MemSourceMgr)));

  src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  src->pub.init_source = MemInitSource;
  src->pub.fill_input_buffer = MemFillInputBuffer;
  src->pub.skip_input_data = MemSkipInputData;
  src->pub.resync_to_restart = jpeg_resync_to_restart;
  src->pub.term_source = MemTermSource;
  src->data = reinterpret_cast<const unsigned char *>(data);
  src->datasize = datasize;
  src->pub.bytes_in_buffer = 0;
  src->pub.next_input_byte = nullptr;
  src->try_recover_truncated_jpeg = try_recover_truncated_jpeg;
}

}  // namespace jpeg
}  // namespace tensorflow
