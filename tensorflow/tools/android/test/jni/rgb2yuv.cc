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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSrgb2yuvDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSrgb2yuvDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSrgb2yuvDTcc() {
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

// These utility functions allow for the conversion of RGB data to YUV data.

#include "tensorflow/tools/android/test/jni/rgb2yuv.h"

static inline void WriteYUV(const int x, const int y, const int width,
                            const int r8, const int g8, const int b8,
                            uint8_t* const pY, uint8_t* const pUV) {
  // Using formulas from http://msdn.microsoft.com/en-us/library/ms893078
  *pY = ((66 * r8 + 129 * g8 + 25 * b8 + 128) >> 8) + 16;

  // Odd widths get rounded up so that UV blocks on the side don't get cut off.
  const int blocks_per_row = (width + 1) / 2;

  // 2 bytes per UV block
  const int offset = 2 * (((y / 2) * blocks_per_row + (x / 2)));

  // U and V are the average values of all 4 pixels in the block.
  if (!(x & 1) && !(y & 1)) {
    // Explicitly clear the block if this is the first pixel in it.
    pUV[offset] = 0;
    pUV[offset + 1] = 0;
  }

  // V (with divide by 4 factored in)
#ifdef __APPLE__
  const int u_offset = 0;
  const int v_offset = 1;
#else
  const int u_offset = 1;
  const int v_offset = 0;
#endif
  pUV[offset + v_offset] += ((112 * r8 - 94 * g8 - 18 * b8 + 128) >> 10) + 32;

  // U (with divide by 4 factored in)
  pUV[offset + u_offset] += ((-38 * r8 - 74 * g8 + 112 * b8 + 128) >> 10) + 32;
}

void ConvertARGB8888ToYUV420SP(const uint32_t* const input,
                               uint8_t* const output, int width, int height) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSrgb2yuvDTcc mht_0(mht_0_v, 223, "", "./tensorflow/tools/android/test/jni/rgb2yuv.cc", "ConvertARGB8888ToYUV420SP");

  uint8_t* pY = output;
  uint8_t* pUV = output + (width * height);
  const uint32_t* in = input;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const uint32_t rgb = *in++;
#ifdef __APPLE__
      const int nB = (rgb >> 8) & 0xFF;
      const int nG = (rgb >> 16) & 0xFF;
      const int nR = (rgb >> 24) & 0xFF;
#else
      const int nR = (rgb >> 16) & 0xFF;
      const int nG = (rgb >> 8) & 0xFF;
      const int nB = rgb & 0xFF;
#endif
      WriteYUV(x, y, width, nR, nG, nB, pY++, pUV);
    }
  }
}

void ConvertRGB565ToYUV420SP(const uint16_t* const input, uint8_t* const output,
                             const int width, const int height) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSrgb2yuvDTcc mht_1(mht_1_v, 249, "", "./tensorflow/tools/android/test/jni/rgb2yuv.cc", "ConvertRGB565ToYUV420SP");

  uint8_t* pY = output;
  uint8_t* pUV = output + (width * height);
  const uint16_t* in = input;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const uint32_t rgb = *in++;

      const int r5 = ((rgb >> 11) & 0x1F);
      const int g6 = ((rgb >> 5) & 0x3F);
      const int b5 = (rgb & 0x1F);

      // Shift left, then fill in the empty low bits with a copy of the high
      // bits so we can stretch across the entire 0 - 255 range.
      const int r8 = r5 << 3 | r5 >> 2;
      const int g8 = g6 << 2 | g6 >> 4;
      const int b8 = b5 << 3 | b5 >> 2;

      WriteYUV(x, y, width, r8, g8, b8, pY++, pUV);
    }
  }
}
