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
class MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_mem_unittestDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_mem_unittestDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_mem_unittestDTcc() {
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

#include "tensorflow/core/lib/jpeg/jpeg_mem.h"

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>

#include "absl/base/casts.h"
#include "tensorflow/core/lib/jpeg/jpeg_handle.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"


namespace tensorflow {
namespace jpeg {
namespace {

const char kTestData[] = "tensorflow/core/lib/jpeg/testdata/";

int ComputeSumAbsoluteDifference(const uint8* a, const uint8* b, int width,
                                 int height, int a_stride, int b_stride) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_mem_unittestDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/lib/jpeg/jpeg_mem_unittest.cc", "ComputeSumAbsoluteDifference");

  int totalerr = 0;
  for (int i = 0; i < height; i++) {
    const uint8* const pa = a + i * a_stride;
    const uint8* const pb = b + i * b_stride;
    for (int j = 0; j < 3 * width; j++) {
      totalerr += abs(static_cast<int>(pa[j]) - static_cast<int>(pb[j]));
    }
  }
  return totalerr;
}

// Reads the contents of the file into output
void ReadFileToStringOrDie(Env* env, const string& filename, string* output) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_mem_unittestDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/lib/jpeg/jpeg_mem_unittest.cc", "ReadFileToStringOrDie");

  TF_CHECK_OK(ReadFileToString(env, filename, output));
}

void TestJPEG(Env* env, const string& jpegfile) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("jpegfile: \"" + jpegfile + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_mem_unittestDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/lib/jpeg/jpeg_mem_unittest.cc", "TestJPEG");

  // Read the data from the jpeg file into memory
  string jpeg;
  ReadFileToStringOrDie(env, jpegfile, &jpeg);
  const int fsize = jpeg.size();
  const uint8* const temp = absl::bit_cast<const uint8*>(jpeg.data());

  // Try partial decoding (half of the data)
  int w, h, c;
  std::unique_ptr<uint8[]> imgdata;

  UncompressFlags flags;
  flags.components = 3;

  // Set min_acceptable_fraction to something insufficient
  flags.min_acceptable_fraction = 0.8;
  imgdata.reset(Uncompress(temp, fsize / 2, flags, &w, &h, &c, nullptr));
  CHECK(imgdata == nullptr);

  // Now, use a value that makes fsize/2 be enough for a black-filling
  flags.min_acceptable_fraction = 0.01;
  imgdata.reset(Uncompress(temp, fsize / 2, flags, &w, &h, &c, nullptr));
  CHECK(imgdata != nullptr);

  // Finally, uncompress the whole data
  flags.min_acceptable_fraction = 1.0;
  imgdata.reset(Uncompress(temp, fsize, flags, &w, &h, &c, nullptr));
  CHECK(imgdata != nullptr);
}

TEST(JpegMemTest, Jpeg) {
  Env* env = Env::Default();
  const string data_path = kTestData;

  // Name of a valid jpeg file on the disk
  TestJPEG(env, data_path + "jpeg_merge_test1.jpg");

  // Exercise CMYK machinery as well
  TestJPEG(env, data_path + "jpeg_merge_test1_cmyk.jpg");
}

void TestCropAndDecodeJpeg(Env* env, const string& jpegfile,
                           const UncompressFlags& default_flags) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("jpegfile: \"" + jpegfile + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_mem_unittestDTcc mht_3(mht_3_v, 280, "", "./tensorflow/core/lib/jpeg/jpeg_mem_unittest.cc", "TestCropAndDecodeJpeg");

  // Read the data from the jpeg file into memory
  string jpeg;
  ReadFileToStringOrDie(env, jpegfile, &jpeg);
  const int fsize = jpeg.size();
  const auto* temp = absl::bit_cast<const uint8*>(jpeg.data());

  // Decode the whole image.
  std::unique_ptr<uint8[]> imgdata1;
  int w1, h1, c1;
  {
    UncompressFlags flags = default_flags;
    if (flags.stride == 0) {
      imgdata1.reset(Uncompress(temp, fsize, flags, &w1, &h1, &c1, nullptr));
    } else {
      // If stride is not zero, the default allocator would fail because it
      // allocate w*h*c bytes, but the actual required bytes should be stride*h.
      // Therefore, we provide a specialized allocator here.
      uint8* buffer = nullptr;
      imgdata1.reset(Uncompress(temp, fsize, flags, nullptr,
                                [&](int width, int height, int components) {
                                  w1 = width;
                                  h1 = height;
                                  c1 = components;
                                  buffer = new uint8[flags.stride * height];
                                  return buffer;
                                }));
    }
    ASSERT_NE(imgdata1, nullptr);
  }

  auto check_crop_and_decode_func = [&](int crop_x, int crop_y, int crop_width,
                                        int crop_height) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_mem_unittestDTcc mht_4(mht_4_v, 315, "", "./tensorflow/core/lib/jpeg/jpeg_mem_unittest.cc", "lambda");

    std::unique_ptr<uint8[]> imgdata2;
    int w, h, c;
    UncompressFlags flags = default_flags;
    flags.crop = true;
    flags.crop_x = crop_x;
    flags.crop_y = crop_y;
    flags.crop_width = crop_width;
    flags.crop_height = crop_height;
    if (flags.stride == 0) {
      imgdata2.reset(Uncompress(temp, fsize, flags, &w, &h, &c, nullptr));
    } else {
      uint8* buffer = nullptr;
      imgdata2.reset(Uncompress(temp, fsize, flags, nullptr,
                                [&](int width, int height, int components) {
                                  w = width;
                                  h = height;
                                  c = components;
                                  buffer = new uint8[flags.stride * height];
                                  return buffer;
                                }));
    }
    ASSERT_NE(imgdata2, nullptr);

    ASSERT_EQ(w, crop_width);
    ASSERT_EQ(h, crop_height);
    ASSERT_EQ(c, c1);

    const int stride1 = (flags.stride != 0) ? flags.stride : w1 * c;
    const int stride2 = (flags.stride != 0) ? flags.stride : w * c;
    for (int i = 0; i < crop_height; i++) {
      const uint8* p1 = &imgdata1[(i + crop_y) * stride1 + crop_x * c];
      const uint8* p2 = &imgdata2[i * stride2];

      for (int j = 0; j < c * w; j++) {
        ASSERT_EQ(p1[j], p2[j])
            << "p1 != p2 in [" << i << "][" << j / 3 << "][" << j % 3 << "]";
      }
    }
  };

  // Check different crop windows.
  check_crop_and_decode_func(0, 0, 5, 5);
  check_crop_and_decode_func(0, 0, w1, 5);
  check_crop_and_decode_func(0, 0, 5, h1);
  check_crop_and_decode_func(0, 0, w1, h1);
  check_crop_and_decode_func(w1 - 5, h1 - 6, 5, 6);
  check_crop_and_decode_func(5, 6, 10, 15);
}

TEST(JpegMemTest, CropAndDecodeJpeg) {
  Env* env = Env::Default();
  const string data_path = kTestData;
  UncompressFlags flags;

  // Test basic flags for jpeg and cmyk jpeg.
  TestCropAndDecodeJpeg(env, data_path + "jpeg_merge_test1.jpg", flags);
  TestCropAndDecodeJpeg(env, data_path + "jpeg_merge_test1_cmyk.jpg", flags);
}

TEST(JpegMemTest, CropAndDecodeJpegWithRatio) {
  Env* env = Env::Default();
  const string data_path = kTestData;
  UncompressFlags flags;
  for (int ratio : {1, 2, 4, 8}) {
    flags.ratio = ratio;
    TestCropAndDecodeJpeg(env, data_path + "jpeg_merge_test1.jpg", flags);
  }
}

TEST(JpegMemTest, CropAndDecodeJpegWithComponents) {
  Env* env = Env::Default();
  const string data_path = kTestData;
  UncompressFlags flags;
  for (const int components : {0, 1, 3}) {
    flags.components = components;
    TestCropAndDecodeJpeg(env, data_path + "jpeg_merge_test1.jpg", flags);
  }
}

TEST(JpegMemTest, CropAndDecodeJpegWithUpScaling) {
  Env* env = Env::Default();
  const string data_path = kTestData;
  UncompressFlags flags;
  flags.fancy_upscaling = true;
  TestCropAndDecodeJpeg(env, data_path + "jpeg_merge_test1.jpg", flags);
}

TEST(JpegMemTest, CropAndDecodeJpegWithStride) {
  Env* env = Env::Default();
  const string data_path = kTestData;

  // Read the data from the jpeg file into memory
  string jpeg;
  ReadFileToStringOrDie(env, data_path + "jpeg_merge_test1.jpg", &jpeg);
  const int fsize = jpeg.size();
  const auto* temp = absl::bit_cast<const uint8*>(jpeg.data());

  int w, h, c;
  ASSERT_TRUE(GetImageInfo(temp, fsize, &w, &h, &c));

  // stride must be either 0 or > w*c; otherwise, uncompress would fail.
  UncompressFlags flags;
  flags.stride = w * c;
  TestCropAndDecodeJpeg(env, data_path + "jpeg_merge_test1.jpg", flags);
  flags.stride = w * c * 3;
  TestCropAndDecodeJpeg(env, data_path + "jpeg_merge_test1.jpg", flags);
  flags.stride = w * c + 100;
  TestCropAndDecodeJpeg(env, data_path + "jpeg_merge_test1.jpg", flags);
}

void CheckInvalidCropWindowFailed(const uint8* const temp, int fsize, int x,
                                  int y, int w, int h) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_mem_unittestDTcc mht_5(mht_5_v, 430, "", "./tensorflow/core/lib/jpeg/jpeg_mem_unittest.cc", "CheckInvalidCropWindowFailed");

  std::unique_ptr<uint8[]> imgdata;
  int ww, hh, cc;
  UncompressFlags flags;
  flags.components = 3;
  flags.crop = true;
  flags.crop_x = x;
  flags.crop_y = y;
  flags.crop_width = w;
  flags.crop_height = h;
  imgdata.reset(Uncompress(temp, fsize, flags, &ww, &hh, &cc, nullptr));
  CHECK(imgdata == nullptr);
}

TEST(JpegMemTest, CropAndDecodeJpegWithInvalidCropWindow) {
  Env* env = Env::Default();
  const string data_path = kTestData;

  // Read the data from the jpeg file into memory
  string jpeg;
  ReadFileToStringOrDie(env, data_path + "jpeg_merge_test1.jpg", &jpeg);
  const int fsize = jpeg.size();
  const auto* temp = absl::bit_cast<const uint8*>(jpeg.data());

  int w, h, c;
  ASSERT_TRUE(GetImageInfo(temp, fsize, &w, &h, &c));

  // Width and height for the crop window must be non zero.
  CheckInvalidCropWindowFailed(temp, fsize, 11, 11, /*w=*/0, 11);
  CheckInvalidCropWindowFailed(temp, fsize, 11, 11, 11, /*h=*/0);

  // Crop window must be non negative.
  CheckInvalidCropWindowFailed(temp, fsize, /*x=*/-1, 11, 11, 11);
  CheckInvalidCropWindowFailed(temp, fsize, 11, /*y=*/-1, 11, 11);
  CheckInvalidCropWindowFailed(temp, fsize, 11, 11, /*w=*/-1, 11);
  CheckInvalidCropWindowFailed(temp, fsize, 11, 11, 11, /*h=*/-1);

  // Invalid crop window width: x + crop_width = w + 1 > w
  CheckInvalidCropWindowFailed(temp, fsize, /*x=*/w - 10, 11, 11, 11);
  // Invalid crop window height: y + crop_height= h + 1 > h
  CheckInvalidCropWindowFailed(temp, fsize, 11, /*y=*/h - 10, 11, 11);
}

TEST(JpegMemTest, Jpeg2) {
  // create known data, for size in_w x in_h
  const int in_w = 256;
  const int in_h = 256;
  const int stride1 = 3 * in_w;
  const std::unique_ptr<uint8[]> refdata1(new uint8[stride1 * in_h]);
  for (int i = 0; i < in_h; i++) {
    for (int j = 0; j < in_w; j++) {
      const int offset = i * stride1 + 3 * j;
      refdata1[offset + 0] = i;
      refdata1[offset + 1] = j;
      refdata1[offset + 2] = static_cast<uint8>((i + j) >> 1);
    }
  }

  // duplicate with weird input stride
  const int stride2 = 3 * 357;
  const std::unique_ptr<uint8[]> refdata2(new uint8[stride2 * in_h]);
  for (int i = 0; i < in_h; i++) {
    memcpy(&refdata2[i * stride2], &refdata1[i * stride1], 3 * in_w);
  }

  // Test compression
  string cpdata1, cpdata2;
  {
    const string kXMP = "XMP_TEST_123";

    // Compress it to JPEG
    CompressFlags flags;
    flags.format = FORMAT_RGB;
    flags.quality = 97;
    flags.xmp_metadata = kXMP;
    cpdata1 = Compress(refdata1.get(), in_w, in_h, flags);
    flags.stride = stride2;
    cpdata2 = Compress(refdata2.get(), in_w, in_h, flags);
    // Different input stride shouldn't change the output
    CHECK_EQ(cpdata1, cpdata2);

    // Verify valid XMP.
    CHECK_NE(string::npos, cpdata1.find(kXMP));

    // Test the other API, where a storage string is supplied
    tstring cptest;
    flags.stride = 0;
    Compress(refdata1.get(), in_w, in_h, flags, &cptest);
    CHECK_EQ(cptest, cpdata1);
    flags.stride = stride2;
    Compress(refdata2.get(), in_w, in_h, flags, &cptest);
    CHECK_EQ(cptest, cpdata2);
  }

  // Uncompress twice: once with 3 components and once with autodetect.
  std::unique_ptr<uint8[]> imgdata1;
  for (const int components : {0, 3}) {
    // Uncompress it
    UncompressFlags flags;
    flags.components = components;
    int w, h, c;
    imgdata1.reset(Uncompress(cpdata1.c_str(), cpdata1.length(), flags, &w, &h,
                              &c, nullptr));

    // Check obvious formatting stuff
    CHECK_EQ(w, in_w);
    CHECK_EQ(h, in_h);
    CHECK_EQ(c, 3);
    CHECK(imgdata1.get());

    // Compare the two images
    const int totalerr = ComputeSumAbsoluteDifference(
        imgdata1.get(), refdata1.get(), in_w, in_h, stride1, stride1);
    CHECK_LE(totalerr, 85000);
  }

  // check the second image too. Should be bitwise identical to the first.
  // uncompress using a weird stride
  {
    UncompressFlags flags;
    flags.stride = 3 * 411;
    const std::unique_ptr<uint8[]> imgdata2(new uint8[flags.stride * in_h]);
    CHECK(imgdata2.get() == Uncompress(cpdata2.c_str(), cpdata2.length(), flags,
                                       nullptr /* nwarn */,
                                       [=, &imgdata2](int w, int h, int c) {
                                         CHECK_EQ(w, in_w);
                                         CHECK_EQ(h, in_h);
                                         CHECK_EQ(c, 3);
                                         return imgdata2.get();
                                       }));
    const int totalerr = ComputeSumAbsoluteDifference(
        imgdata1.get(), imgdata2.get(), in_w, in_h, stride1, flags.stride);
    CHECK_EQ(totalerr, 0);
  }

  {
    // Uncompress it with a faster, lossier algorithm.
    UncompressFlags flags;
    flags.components = 3;
    flags.dct_method = JDCT_IFAST;
    int w, h, c;
    imgdata1.reset(Uncompress(cpdata1.c_str(), cpdata1.length(), flags, &w, &h,
                              &c, nullptr));

    // Check obvious formatting stuff
    CHECK_EQ(w, in_w);
    CHECK_EQ(h, in_h);
    CHECK_EQ(c, 3);
    CHECK(imgdata1.get());

    // Compare the two images
    const int totalerr = ComputeSumAbsoluteDifference(
        imgdata1.get(), refdata1.get(), in_w, in_h, stride1, stride1);
    ASSERT_LE(totalerr, 200000);
  }
}

// Takes JPEG data and reads its headers to determine whether or not the JPEG
// was chroma downsampled.
bool IsChromaDownsampled(const string& jpegdata) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("jpegdata: \"" + jpegdata + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_mem_unittestDTcc mht_6(mht_6_v, 593, "", "./tensorflow/core/lib/jpeg/jpeg_mem_unittest.cc", "IsChromaDownsampled");

  // Initialize libjpeg structures to have a memory source
  // Modify the usual jpeg error manager to catch fatal errors.
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  jmp_buf jpeg_jmpbuf;
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = CatchError;
  if (setjmp(jpeg_jmpbuf)) return false;

  // set up, read header, set image parameters, save size
  jpeg_create_decompress(&cinfo);
  SetSrc(&cinfo, jpegdata.c_str(), jpegdata.size(), false);

  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);  // required to transfer image size to cinfo
  const int components = cinfo.output_components;
  if (components == 1) return false;

  // Check validity
  CHECK_EQ(3, components);
  CHECK_EQ(cinfo.comp_info[1].h_samp_factor, cinfo.comp_info[2].h_samp_factor)
      << "The h sampling factors should be the same.";
  CHECK_EQ(cinfo.comp_info[1].v_samp_factor, cinfo.comp_info[2].v_samp_factor)
      << "The v sampling factors should be the same.";
  for (int i = 0; i < components; ++i) {
    CHECK_GT(cinfo.comp_info[i].h_samp_factor, 0) << "Invalid sampling factor.";
    CHECK_EQ(cinfo.comp_info[i].h_samp_factor, cinfo.comp_info[i].v_samp_factor)
        << "The sampling factor should be the same in both directions.";
  }

  // We're downsampled if we use fewer samples for color than for brightness.
  // Do this before deallocating cinfo.
  const bool downsampled =
      cinfo.comp_info[1].h_samp_factor < cinfo.comp_info[0].h_samp_factor;

  jpeg_destroy_decompress(&cinfo);
  return downsampled;
}

TEST(JpegMemTest, ChromaDownsampling) {
  // Read the data from a test jpeg file into memory
  const string jpegfile = string(kTestData) + "jpeg_merge_test1.jpg";
  string jpeg;
  ReadFileToStringOrDie(Env::Default(), jpegfile, &jpeg);

  // Verify that compressing the JPEG with chroma downsampling works.
  //
  // First, uncompress the JPEG.
  UncompressFlags unflags;
  unflags.components = 3;
  int w, h, c;
  int64_t num_warnings;
  std::unique_ptr<uint8[]> uncompressed(Uncompress(
      jpeg.c_str(), jpeg.size(), unflags, &w, &h, &c, &num_warnings));
  CHECK(uncompressed != nullptr);
  CHECK_EQ(num_warnings, 0);

  // Recompress the JPEG with and without chroma downsampling
  for (const bool downsample : {false, true}) {
    CompressFlags flags;
    flags.format = FORMAT_RGB;
    flags.quality = 85;
    flags.chroma_downsampling = downsample;
    tstring recompressed;
    Compress(uncompressed.get(), w, h, flags, &recompressed);
    CHECK(!recompressed.empty());
    CHECK_EQ(IsChromaDownsampled(recompressed), downsample);
  }
}

void TestBadJPEG(Env* env, const string& bad_jpeg_file, int expected_width,
                 int expected_height, const string& reference_RGB_file,
                 const bool try_recover_truncated_jpeg) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("bad_jpeg_file: \"" + bad_jpeg_file + "\"");
   mht_7_v.push_back("reference_RGB_file: \"" + reference_RGB_file + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSjpegPSjpeg_mem_unittestDTcc mht_7(mht_7_v, 672, "", "./tensorflow/core/lib/jpeg/jpeg_mem_unittest.cc", "TestBadJPEG");

  string jpeg;
  ReadFileToStringOrDie(env, bad_jpeg_file, &jpeg);

  UncompressFlags flags;
  flags.components = 3;
  flags.try_recover_truncated_jpeg = try_recover_truncated_jpeg;

  int width, height, components;
  std::unique_ptr<uint8[]> imgdata;
  imgdata.reset(Uncompress(jpeg.c_str(), jpeg.size(), flags, &width, &height,
                           &components, nullptr));
  if (expected_width > 0) {  // we expect the file to decode into 'something'
    CHECK_EQ(width, expected_width);
    CHECK_EQ(height, expected_height);
    CHECK_EQ(components, 3);
    CHECK(imgdata.get());
    if (!reference_RGB_file.empty()) {
      string ref;
      ReadFileToStringOrDie(env, reference_RGB_file, &ref);
      CHECK(!memcmp(ref.data(), imgdata.get(), ref.size()));
    }
  } else {  // no decodable
    CHECK(!imgdata.get()) << "file:" << bad_jpeg_file;
  }
}

TEST(JpegMemTest, BadJpeg) {
  Env* env = Env::Default();
  const string data_path = kTestData;

  // Test corrupt file
  TestBadJPEG(env, data_path + "bad_huffman.jpg", 1024, 768, "", false);
  TestBadJPEG(env, data_path + "corrupt.jpg", 0 /*120*/, 90, "", false);

  // Truncated files, undecodable because of missing lines:
  TestBadJPEG(env, data_path + "corrupt34_2.jpg", 0, 3300, "", false);
  TestBadJPEG(env, data_path + "corrupt34_3.jpg", 0, 3300, "", false);
  TestBadJPEG(env, data_path + "corrupt34_4.jpg", 0, 3300, "", false);

  // Try in 'recover' mode now:
  TestBadJPEG(env, data_path + "corrupt34_2.jpg", 2544, 3300, "", true);
  TestBadJPEG(env, data_path + "corrupt34_3.jpg", 2544, 3300, "", true);
  TestBadJPEG(env, data_path + "corrupt34_4.jpg", 2544, 3300, "", true);
}

}  // namespace
}  // namespace jpeg
}  // namespace tensorflow
