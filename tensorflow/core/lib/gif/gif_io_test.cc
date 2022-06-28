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
class MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_io_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_io_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_io_testDTcc() {
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

#include "tensorflow/core/lib/gif/gif_io.h"

#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gif {
namespace {

const char kTestData[] = "tensorflow/core/lib/gif/testdata/";

struct DecodeGifTestCase {
  const string filepath;
  const int num_frames;
  const int width;
  const int height;
  const int channels;
};

void ReadFileToStringOrDie(Env* env, const string& filename, string* output) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_io_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/lib/gif/gif_io_test.cc", "ReadFileToStringOrDie");

  TF_CHECK_OK(ReadFileToString(env, filename, output));
}

void TestDecodeGif(Env* env, DecodeGifTestCase testcase) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_io_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/lib/gif/gif_io_test.cc", "TestDecodeGif");

  string gif;
  ReadFileToStringOrDie(env, testcase.filepath, &gif);

  // Decode gif image data.
  std::unique_ptr<uint8[]> imgdata;
  int nframes, w, h, c;
  string error_string;
  imgdata.reset(gif::Decode(
      gif.data(), gif.size(),
      [&](int frame_cnt, int width, int height, int channels) -> uint8* {
        nframes = frame_cnt;
        w = width;
        h = height;
        c = channels;
        return new uint8[frame_cnt * height * width * channels];
      },
      &error_string));
  ASSERT_NE(imgdata, nullptr);
  // Make sure the decoded information matches the ground-truth image info.
  ASSERT_EQ(nframes, testcase.num_frames);
  ASSERT_EQ(w, testcase.width);
  ASSERT_EQ(h, testcase.height);
  ASSERT_EQ(c, testcase.channels);
}

TEST(GifTest, Gif) {
  Env* env = Env::Default();
  const string testdata_path = kTestData;
  std::vector<DecodeGifTestCase> testcases(
      {// file_path, num_of_channels, width, height, channels
       {testdata_path + "lena.gif", 1, 51, 26, 3},
       {testdata_path + "optimized.gif", 12, 20, 40, 3},
       {testdata_path + "red_black.gif", 1, 16, 16, 3},
       {testdata_path + "scan.gif", 12, 20, 40, 3},
       {testdata_path + "squares.gif", 2, 16, 16, 3}});

  for (const auto& tc : testcases) {
    TestDecodeGif(env, tc);
  }
}

void TestDecodeAnimatedGif(Env* env, const uint8* gif_data,
                           const string& png_filepath, int frame_idx) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("png_filepath: \"" + png_filepath + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_io_testDTcc mht_2(mht_2_v, 260, "", "./tensorflow/core/lib/gif/gif_io_test.cc", "TestDecodeAnimatedGif");

  string png;  // ground-truth
  ReadFileToStringOrDie(env, png_filepath, &png);

  // Compare decoded gif to ground-truth image frames in png format.
  png::DecodeContext decode;
  png::CommonInitDecode(png, 3, 8, &decode);
  const int width = static_cast<int>(decode.width);
  const int height = static_cast<int>(decode.height);
  std::unique_ptr<uint8[]> png_imgdata(
      new uint8[height * width * decode.channels]);
  png::CommonFinishDecode(reinterpret_cast<png_bytep>(png_imgdata.get()),
                          decode.channels * width * sizeof(uint8), &decode);

  int frame_len = width * height * decode.channels;
  int gif_idx = frame_len * frame_idx;
  for (int i = 0; i < frame_len; i++) {
    ASSERT_EQ(gif_data[gif_idx + i], png_imgdata[i]);
  }
}

TEST(GifTest, AnimatedGif) {
  Env* env = Env::Default();
  const string testdata_path = kTestData;

  // Read animated gif file once.
  string gif;
  ReadFileToStringOrDie(env, testdata_path + "pendulum_sm.gif", &gif);

  std::unique_ptr<uint8[]> gif_imgdata;
  int nframes, w, h, c;
  string error_string;
  gif_imgdata.reset(gif::Decode(
      gif.data(), gif.size(),
      [&](int num_frames, int width, int height, int channels) -> uint8* {
        nframes = num_frames;
        w = width;
        h = height;
        c = channels;
        return new uint8[num_frames * height * width * channels];
      },
      &error_string));

  TestDecodeAnimatedGif(env, gif_imgdata.get(),
                        testdata_path + "pendulum_sm_frame0.png", 0);
  TestDecodeAnimatedGif(env, gif_imgdata.get(),
                        testdata_path + "pendulum_sm_frame1.png", 1);
  TestDecodeAnimatedGif(env, gif_imgdata.get(),
                        testdata_path + "pendulum_sm_frame2.png", 2);
}

void TestExpandAnimations(Env* env, const string& filepath) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("filepath: \"" + filepath + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_io_testDTcc mht_3(mht_3_v, 315, "", "./tensorflow/core/lib/gif/gif_io_test.cc", "TestExpandAnimations");

  string gif;
  ReadFileToStringOrDie(env, filepath, &gif);

  std::unique_ptr<uint8[]> imgdata;
  string error_string;
  int nframes;
  // `expand_animations` is set to true by default. Set to false.
  bool expand_animations = false;
  imgdata.reset(gif::Decode(
      gif.data(), gif.size(),
      [&](int frame_cnt, int width, int height, int channels) -> uint8* {
        nframes = frame_cnt;
        return new uint8[frame_cnt * height * width * channels];
      },
      &error_string, expand_animations));

  // Check that only 1 frame is being decoded.
  ASSERT_EQ(nframes, 1);
}

TEST(GifTest, ExpandAnimations) {
  Env* env = Env::Default();
  const string testdata_path = kTestData;

  // Test all animated gif test images.
  TestExpandAnimations(env, testdata_path + "scan.gif");
  TestExpandAnimations(env, testdata_path + "pendulum_sm.gif");
  TestExpandAnimations(env, testdata_path + "squares.gif");
}

void TestInvalidGifFormat(const string& header_bytes) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("header_bytes: \"" + header_bytes + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSgifPSgif_io_testDTcc mht_4(mht_4_v, 350, "", "./tensorflow/core/lib/gif/gif_io_test.cc", "TestInvalidGifFormat");

  std::unique_ptr<uint8[]> imgdata;
  string error_string;
  int nframes;
  imgdata.reset(gif::Decode(
      header_bytes.data(), header_bytes.size(),
      [&](int frame_cnt, int width, int height, int channels) -> uint8* {
        nframes = frame_cnt;
        return new uint8[frame_cnt * height * width * channels];
      },
      &error_string));

  // Check that decoding image formats other than gif throws an error.
  string err_msg = "failed to open gif file";
  ASSERT_EQ(error_string.substr(0, 23), err_msg);
}

TEST(GifTest, BadGif) {
  // Input header bytes of other image formats to gif decoder.
  TestInvalidGifFormat("\x89\x50\x4E\x47\x0D\x0A\x1A\x0A");  // png
  TestInvalidGifFormat("\x42\x4d");                          // bmp
  TestInvalidGifFormat("\xff\xd8\xff");                      // jpeg
  TestInvalidGifFormat("\x49\x49\x2A\x00");                  // tiff
}

}  // namespace
}  // namespace gif
}  // namespace tensorflow
