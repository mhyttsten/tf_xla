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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSyuv2rgbDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSyuv2rgbDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSyuv2rgbDTcc() {
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

// This is a collection of routines which converts various YUV image formats
// to ARGB.

#include "tensorflow/tools/android/test/jni/yuv2rgb.h"

#ifndef MAX
#define MAX(a, b) ({__typeof__(a) _a = (a); __typeof__(b) _b = (b); _a > _b ? _a : _b; })
#define MIN(a, b) ({__typeof__(a) _a = (a); __typeof__(b) _b = (b); _a < _b ? _a : _b; })
#endif

// This value is 2 ^ 18 - 1, and is used to clamp the RGB values before their ranges
// are normalized to eight bits.
static const int kMaxChannelValue = 262143;

static inline uint32_t YUV2RGB(int nY, int nU, int nV) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSyuv2rgbDTcc mht_0(mht_0_v, 199, "", "./tensorflow/tools/android/test/jni/yuv2rgb.cc", "YUV2RGB");

  nY -= 16;
  nU -= 128;
  nV -= 128;
  if (nY < 0) nY = 0;

  // This is the floating point equivalent. We do the conversion in integer
  // because some Android devices do not have floating point in hardware.
  // nR = (int)(1.164 * nY + 2.018 * nU);
  // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
  // nB = (int)(1.164 * nY + 1.596 * nV);

  int nR = 1192 * nY + 1634 * nV;
  int nG = 1192 * nY - 833 * nV - 400 * nU;
  int nB = 1192 * nY + 2066 * nU;

  nR = MIN(kMaxChannelValue, MAX(0, nR));
  nG = MIN(kMaxChannelValue, MAX(0, nG));
  nB = MIN(kMaxChannelValue, MAX(0, nB));

  nR = (nR >> 10) & 0xff;
  nG = (nG >> 10) & 0xff;
  nB = (nB >> 10) & 0xff;

  return 0xff000000 | (nR << 16) | (nG << 8) | nB;
}

//  Accepts a YUV 4:2:0 image with a plane of 8 bit Y samples followed by
//  separate u and v planes with arbitrary row and column strides,
//  containing 8 bit 2x2 subsampled chroma samples.
//  Converts to a packed ARGB 32 bit output of the same pixel dimensions.
void ConvertYUV420ToARGB8888(const uint8_t* const yData,
                             const uint8_t* const uData,
                             const uint8_t* const vData, uint32_t* const output,
                             const int width, const int height,
                             const int y_row_stride, const int uv_row_stride,
                             const int uv_pixel_stride) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSyuv2rgbDTcc mht_1(mht_1_v, 238, "", "./tensorflow/tools/android/test/jni/yuv2rgb.cc", "ConvertYUV420ToARGB8888");

  uint32_t* out = output;

  for (int y = 0; y < height; y++) {
    const uint8_t* pY = yData + y_row_stride * y;

    const int uv_row_start = uv_row_stride * (y >> 1);
    const uint8_t* pU = uData + uv_row_start;
    const uint8_t* pV = vData + uv_row_start;

    for (int x = 0; x < width; x++) {
      const int uv_offset = (x >> 1) * uv_pixel_stride;
      *out++ = YUV2RGB(pY[x], pU[uv_offset], pV[uv_offset]);
    }
  }
}

//  Accepts a YUV 4:2:0 image with a plane of 8 bit Y samples followed by an
//  interleaved U/V plane containing 8 bit 2x2 subsampled chroma samples,
//  except the interleave order of U and V is reversed. Converts to a packed
//  ARGB 32 bit output of the same pixel dimensions.
void ConvertYUV420SPToARGB8888(const uint8_t* const yData,
                               const uint8_t* const uvData,
                               uint32_t* const output, const int width,
                               const int height) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSyuv2rgbDTcc mht_2(mht_2_v, 265, "", "./tensorflow/tools/android/test/jni/yuv2rgb.cc", "ConvertYUV420SPToARGB8888");

  const uint8_t* pY = yData;
  const uint8_t* pUV = uvData;
  uint32_t* out = output;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int nY = *pY++;
      int offset = (y >> 1) * width + 2 * (x >> 1);
#ifdef __APPLE__
      int nU = pUV[offset];
      int nV = pUV[offset + 1];
#else
      int nV = pUV[offset];
      int nU = pUV[offset + 1];
#endif

      *out++ = YUV2RGB(nY, nU, nV);
    }
  }
}

// The same as above, but downsamples each dimension to half size.
void ConvertYUV420SPToARGB8888HalfSize(const uint8_t* const input,
                                       uint32_t* const output, int width,
                                       int height) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSyuv2rgbDTcc mht_3(mht_3_v, 293, "", "./tensorflow/tools/android/test/jni/yuv2rgb.cc", "ConvertYUV420SPToARGB8888HalfSize");

  const uint8_t* pY = input;
  const uint8_t* pUV = input + (width * height);
  uint32_t* out = output;
  int stride = width;
  width >>= 1;
  height >>= 1;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int nY = (pY[0] + pY[1] + pY[stride] + pY[stride + 1]) >> 2;
      pY += 2;
#ifdef __APPLE__
      int nU = *pUV++;
      int nV = *pUV++;
#else
      int nV = *pUV++;
      int nU = *pUV++;
#endif

      *out++ = YUV2RGB(nY, nU, nV);
    }
    pY += stride;
  }
}

//  Accepts a YUV 4:2:0 image with a plane of 8 bit Y samples followed by an
//  interleaved U/V plane containing 8 bit 2x2 subsampled chroma samples,
//  except the interleave order of U and V is reversed. Converts to a packed
//  RGB 565 bit output of the same pixel dimensions.
void ConvertYUV420SPToRGB565(const uint8_t* const input, uint16_t* const output,
                             const int width, const int height) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSyuv2rgbDTcc mht_4(mht_4_v, 327, "", "./tensorflow/tools/android/test/jni/yuv2rgb.cc", "ConvertYUV420SPToRGB565");

  const uint8_t* pY = input;
  const uint8_t* pUV = input + (width * height);
  uint16_t* out = output;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int nY = *pY++;
      int offset = (y >> 1) * width + 2 * (x >> 1);
#ifdef __APPLE__
      int nU = pUV[offset];
      int nV = pUV[offset + 1];
#else
      int nV = pUV[offset];
      int nU = pUV[offset + 1];
#endif

      nY -= 16;
      nU -= 128;
      nV -= 128;
      if (nY < 0) nY = 0;

      // This is the floating point equivalent. We do the conversion in integer
      // because some Android devices do not have floating point in hardware.
      // nR = (int)(1.164 * nY + 2.018 * nU);
      // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
      // nB = (int)(1.164 * nY + 1.596 * nV);

      int nR = 1192 * nY + 1634 * nV;
      int nG = 1192 * nY - 833 * nV - 400 * nU;
      int nB = 1192 * nY + 2066 * nU;

      nR = MIN(kMaxChannelValue, MAX(0, nR));
      nG = MIN(kMaxChannelValue, MAX(0, nG));
      nB = MIN(kMaxChannelValue, MAX(0, nB));

      // Shift more than for ARGB8888 and apply appropriate bitmask.
      nR = (nR >> 13) & 0x1f;
      nG = (nG >> 12) & 0x3f;
      nB = (nB >> 13) & 0x1f;

      // R is high 5 bits, G is middle 6 bits, and B is low 5 bits.
      *out++ = (nR << 11) | (nG << 5) | nB;
    }
  }
}
