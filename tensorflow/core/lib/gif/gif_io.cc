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
class MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_ioDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_ioDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_ioDTcc() {
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

// Functions to read images in GIF format.

#include "tensorflow/core/lib/gif/gif_io.h"

#include <algorithm>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/gif.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gif {

struct InputBufferInfo {
  const uint8_t* buf;
  int bytes_left;
};

int input_callback(GifFileType* gif_file, GifByteType* buf, int size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_ioDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/lib/gif/gif_io.cc", "input_callback");

  InputBufferInfo* const info =
      reinterpret_cast<InputBufferInfo*>(gif_file->UserData);
  if (info != nullptr) {
    if (size > info->bytes_left) size = info->bytes_left;
    memcpy(buf, info->buf, size);
    info->buf += size;
    info->bytes_left -= size;
    return size;
  }
  return 0;
}

static const char* GifErrorStringNonNull(int error_code) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_ioDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/lib/gif/gif_io.cc", "GifErrorStringNonNull");

  const char* error_string = GifErrorString(error_code);
  if (error_string == nullptr) {
    return "Unknown error";
  }
  return error_string;
}

uint8* Decode(const void* srcdata, int datasize,
              const std::function<uint8*(int, int, int, int)>& allocate_output,
              string* error_string, bool expand_animations) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_ioDTcc mht_2(mht_2_v, 235, "", "./tensorflow/core/lib/gif/gif_io.cc", "Decode");

  int error_code = D_GIF_SUCCEEDED;
  InputBufferInfo info = {reinterpret_cast<const uint8*>(srcdata), datasize};
  GifFileType* gif_file =
      DGifOpen(static_cast<void*>(&info), &input_callback, &error_code);
  const auto cleanup = gtl::MakeCleanup([gif_file]() {
    int error_code = D_GIF_SUCCEEDED;
    if (gif_file && DGifCloseFile(gif_file, &error_code) != GIF_OK) {
      LOG(WARNING) << "Fail to close gif file, reason: "
                   << GifErrorStringNonNull(error_code);
    }
  });
  if (error_code != D_GIF_SUCCEEDED) {
    *error_string = absl::StrCat("failed to open gif file: ",
                                 GifErrorStringNonNull(error_code));
    return nullptr;
  }
  if (DGifSlurp(gif_file) != GIF_OK) {
    *error_string = absl::StrCat("failed to slurp gif file: ",
                                 GifErrorStringNonNull(gif_file->Error));
    return nullptr;
  }
  if (gif_file->ImageCount <= 0) {
    *error_string = "gif file does not contain any image";
    return nullptr;
  }

  int target_num_frames = gif_file->ImageCount;

  // Don't request more memory than needed for each frame, preventing OOM
  int max_frame_width = 0;
  int max_frame_height = 0;
  for (int k = 0; k < target_num_frames; k++) {
    SavedImage* si = &gif_file->SavedImages[k];
    if (max_frame_height < si->ImageDesc.Height)
      max_frame_height = si->ImageDesc.Height;
    if (max_frame_width < si->ImageDesc.Width)
      max_frame_width = si->ImageDesc.Width;
  }

  const int width = max_frame_width;
  const int height = max_frame_height;
  const int channel = 3;
  if (!expand_animations) target_num_frames = 1;

  uint8* const dstdata =
      allocate_output(target_num_frames, width, height, channel);
  if (!dstdata) return nullptr;
  for (int k = 0; k < target_num_frames; k++) {
    uint8* this_dst = dstdata + k * width * channel * height;

    SavedImage* this_image = &gif_file->SavedImages[k];
    GifImageDesc* img_desc = &this_image->ImageDesc;

    // The Graphics Control Block tells us which index in the color map
    // correspond to "transparent color", i.e. no need to update the pixel
    // on the canvas. The "transparent color index" is specific to each
    // sub-frame.
    GraphicsControlBlock gcb;
    DGifSavedExtensionToGCB(gif_file, k, &gcb);

    int imgLeft = img_desc->Left;
    int imgTop = img_desc->Top;
    int imgRight = img_desc->Left + img_desc->Width;
    int imgBottom = img_desc->Top + img_desc->Height;

    if (k > 0) {
      uint8* last_dst = dstdata + (k - 1) * width * channel * height;
      for (int i = 0; i < height; ++i) {
        uint8* p_dst = this_dst + i * width * channel;
        uint8* l_dst = last_dst + i * width * channel;
        for (int j = 0; j < width; ++j) {
          p_dst[j * channel + 0] = l_dst[j * channel + 0];
          p_dst[j * channel + 1] = l_dst[j * channel + 1];
          p_dst[j * channel + 2] = l_dst[j * channel + 2];
        }
      }
    }

    if (img_desc->Left != 0 || img_desc->Top != 0 || img_desc->Width != width ||
        img_desc->Height != height) {
      // If the first frame does not fill the entire canvas then fill the
      // unoccupied canvas with zeros (black).
      if (k == 0) {
        for (int i = 0; i < height; ++i) {
          uint8* p_dst = this_dst + i * width * channel;
          for (int j = 0; j < width; ++j) {
            p_dst[j * channel + 0] = 0;
            p_dst[j * channel + 1] = 0;
            p_dst[j * channel + 2] = 0;
          }
        }
      }

      imgLeft = std::max(imgLeft, 0);
      imgTop = std::max(imgTop, 0);
      imgRight = std::min(imgRight, width);
      imgBottom = std::min(imgBottom, height);
    }

    ColorMapObject* color_map = this_image->ImageDesc.ColorMap
                                    ? this_image->ImageDesc.ColorMap
                                    : gif_file->SColorMap;
    if (color_map == nullptr) {
      *error_string = absl::StrCat("missing color map for frame ", k);
      return nullptr;
    }

    for (int i = imgTop; i < imgBottom; ++i) {
      uint8* p_dst = this_dst + i * width * channel;
      for (int j = imgLeft; j < imgRight; ++j) {
        GifByteType color_index =
            this_image->RasterBits[(i - img_desc->Top) * (img_desc->Width) +
                                   (j - img_desc->Left)];

        if (color_index >= color_map->ColorCount) {
          *error_string = absl::StrCat("found color index ", color_index,
                                       " outside of color map range ",
                                       color_map->ColorCount);
          return nullptr;
        }

        if (color_index == gcb.TransparentColor) {
          // Use the pixel from the previous frame. In other words, no need to
          // update our canvas for this pixel.
          continue;
        }

        const GifColorType& gif_color = color_map->Colors[color_index];
        p_dst[j * channel + 0] = gif_color.Red;
        p_dst[j * channel + 1] = gif_color.Green;
        p_dst[j * channel + 2] = gif_color.Blue;
      }
    }
  }

  return dstdata;
}

}  // namespace gif
}  // namespace tensorflow
