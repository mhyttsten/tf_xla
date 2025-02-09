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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/ram_file_block_cache.h"

#include <cstring>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/cloud/now_seconds_env.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

Status ReadCache(RamFileBlockCache* cache, const string& filename,
                 size_t offset, size_t n, std::vector<char>* out) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "ReadCache");

  out->clear();
  out->resize(n, 0);
  size_t bytes_transferred = 0;
  Status status =
      cache->Read(filename, offset, n, out->data(), &bytes_transferred);
  EXPECT_LE(bytes_transferred, n);
  out->resize(bytes_transferred, n);
  return status;
}

TEST(RamFileBlockCacheTest, IsCacheEnabled) {
  auto fetcher = [](const string& filename, size_t offset, size_t n,
                    char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + filename + "\"");
   mht_1_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    // Do nothing.
    return Status::OK();
  };
  RamFileBlockCache cache1(0, 0, 0, fetcher);
  RamFileBlockCache cache2(16, 0, 0, fetcher);
  RamFileBlockCache cache3(0, 32, 0, fetcher);
  RamFileBlockCache cache4(16, 32, 0, fetcher);

  EXPECT_FALSE(cache1.IsCacheEnabled());
  EXPECT_FALSE(cache2.IsCacheEnabled());
  EXPECT_FALSE(cache3.IsCacheEnabled());
  EXPECT_TRUE(cache4.IsCacheEnabled());
}

TEST(RamFileBlockCacheTest, ValidateAndUpdateFileSignature) {
  int calls = 0;
  auto fetcher = [&calls](const string& filename, size_t offset, size_t n,
                          char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("filename: \"" + filename + "\"");
   mht_2_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_2(mht_2_v, 242, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    calls++;
    memset(buffer, 'x', n);
    *bytes_transferred = n;
    return Status::OK();
  };
  string filename = "file";
  RamFileBlockCache cache(16, 32, 0, fetcher);
  std::vector<char> out;

  // First read.
  EXPECT_TRUE(cache.ValidateAndUpdateFileSignature(filename, 123));
  TF_EXPECT_OK(ReadCache(&cache, filename, 0, 16, &out));
  EXPECT_EQ(calls, 1);

  // Second read. Hit cache.
  EXPECT_TRUE(cache.ValidateAndUpdateFileSignature(filename, 123));
  TF_EXPECT_OK(ReadCache(&cache, filename, 0, 16, &out));
  EXPECT_EQ(calls, 1);

  // Third read. File signatures are different.
  EXPECT_FALSE(cache.ValidateAndUpdateFileSignature(filename, 321));
  TF_EXPECT_OK(ReadCache(&cache, filename, 0, 16, &out));
  EXPECT_EQ(calls, 2);
}

TEST(RamFileBlockCacheTest, PassThrough) {
  const string want_filename = "foo/bar";
  const size_t want_offset = 42;
  const size_t want_n = 1024;
  int calls = 0;
  auto fetcher = [&calls, want_filename, want_offset, want_n](
                     const string& got_filename, size_t got_offset,
                     size_t got_n, char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("got_filename: \"" + got_filename + "\"");
   mht_3_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_3(mht_3_v, 280, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    EXPECT_EQ(got_filename, want_filename);
    EXPECT_EQ(got_offset, want_offset);
    EXPECT_EQ(got_n, want_n);
    calls++;
    memset(buffer, 'x', got_n);
    *bytes_transferred = got_n;
    return Status::OK();
  };
  // If block_size, max_bytes, or both are zero, or want_n is larger than
  // max_bytes the cache is a pass-through.
  RamFileBlockCache cache1(1, 0, 0, fetcher);
  RamFileBlockCache cache2(0, 1, 0, fetcher);
  RamFileBlockCache cache3(0, 0, 0, fetcher);
  RamFileBlockCache cache4(1000, 1000, 0, fetcher);
  std::vector<char> out;
  TF_EXPECT_OK(ReadCache(&cache1, want_filename, want_offset, want_n, &out));
  EXPECT_EQ(calls, 1);
  TF_EXPECT_OK(ReadCache(&cache2, want_filename, want_offset, want_n, &out));
  EXPECT_EQ(calls, 2);
  TF_EXPECT_OK(ReadCache(&cache3, want_filename, want_offset, want_n, &out));
  EXPECT_EQ(calls, 3);
  TF_EXPECT_OK(ReadCache(&cache4, want_filename, want_offset, want_n, &out));
  EXPECT_EQ(calls, 4);
}

TEST(RamFileBlockCacheTest, BlockAlignment) {
  // Initialize a 256-byte buffer.  This is the file underlying the reads we'll
  // do in this test.
  const size_t size = 256;
  std::vector<char> buf;
  for (int i = 0; i < size; i++) {
    buf.push_back(i);
  }
  // The fetcher just fetches slices of the buffer.
  auto fetcher = [&buf](const string& filename, size_t offset, size_t n,
                        char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("filename: \"" + filename + "\"");
   mht_4_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_4(mht_4_v, 321, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    if (offset < buf.size()) {
      size_t bytes_to_copy = std::min<size_t>(buf.size() - offset, n);
      memcpy(buffer, buf.data() + offset, bytes_to_copy);
      *bytes_transferred = bytes_to_copy;
    } else {
      *bytes_transferred = 0;
    }
    return Status::OK();
  };
  for (size_t block_size = 2; block_size <= 4; block_size++) {
    // Make a cache of N-byte block size (1 block) and verify that reads of
    // varying offsets and lengths return correct data.
    RamFileBlockCache cache(block_size, block_size, 0, fetcher);
    for (size_t offset = 0; offset < 10; offset++) {
      for (size_t n = block_size - 2; n <= block_size + 2; n++) {
        std::vector<char> got;
        TF_EXPECT_OK(ReadCache(&cache, "", offset, n, &got));
        // Verify the size of the read.
        if (offset + n <= size) {
          // Expect a full read.
          EXPECT_EQ(got.size(), n) << "block size = " << block_size
                                   << ", offset = " << offset << ", n = " << n;
        } else {
          // Expect a partial read.
          EXPECT_EQ(got.size(), size - offset)
              << "block size = " << block_size << ", offset = " << offset
              << ", n = " << n;
        }
        // Verify the contents of the read.
        std::vector<char>::const_iterator begin = buf.begin() + offset;
        std::vector<char>::const_iterator end =
            offset + n > buf.size() ? buf.end() : begin + n;
        std::vector<char> want(begin, end);
        EXPECT_EQ(got, want) << "block size = " << block_size
                             << ", offset = " << offset << ", n = " << n;
      }
    }
  }
}

TEST(RamFileBlockCacheTest, CacheHits) {
  const size_t block_size = 16;
  std::set<size_t> calls;
  auto fetcher = [&calls, block_size](const string& filename, size_t offset,
                                      size_t n, char* buffer,
                                      size_t* bytes_transferred) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("filename: \"" + filename + "\"");
   mht_5_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_5(mht_5_v, 372, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    EXPECT_EQ(n, block_size);
    EXPECT_EQ(offset % block_size, 0);
    EXPECT_EQ(calls.find(offset), calls.end()) << "at offset " << offset;
    calls.insert(offset);
    memset(buffer, 'x', n);
    *bytes_transferred = n;
    return Status::OK();
  };
  const uint32 block_count = 256;
  RamFileBlockCache cache(block_size, block_count * block_size, 0, fetcher);
  std::vector<char> out;
  out.resize(block_count, 0);
  // The cache has space for `block_count` blocks. The loop with i = 0 should
  // fill the cache, and the loop with i = 1 should be all cache hits. The
  // fetcher checks that it is called once and only once for each offset (to
  // fetch the corresponding block).
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < block_count; j++) {
      TF_EXPECT_OK(ReadCache(&cache, "", block_size * j, block_size, &out));
    }
  }
}

TEST(RamFileBlockCacheTest, OutOfRange) {
  // Tests reads of a 24-byte file with block size 16.
  const size_t block_size = 16;
  const size_t file_size = 24;
  bool first_block = false;
  bool second_block = false;
  auto fetcher = [block_size, file_size, &first_block, &second_block](
                     const string& filename, size_t offset, size_t n,
                     char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("filename: \"" + filename + "\"");
   mht_6_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_6(mht_6_v, 409, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    EXPECT_EQ(n, block_size);
    EXPECT_EQ(offset % block_size, 0);
    size_t bytes_to_copy = 0;
    if (offset == 0) {
      // The first block (16 bytes) of the file.
      memset(buffer, 'x', n);
      bytes_to_copy = n;
      first_block = true;
    } else if (offset == block_size) {
      // The second block (8 bytes) of the file.
      bytes_to_copy = file_size - block_size;
      memset(buffer, 'x', bytes_to_copy);
      second_block = true;
    }
    *bytes_transferred = bytes_to_copy;
    return Status::OK();
  };
  RamFileBlockCache cache(block_size, block_size, 0, fetcher);
  std::vector<char> out;
  // Reading the first 16 bytes should be fine.
  TF_EXPECT_OK(ReadCache(&cache, "", 0, block_size, &out));
  EXPECT_TRUE(first_block);
  EXPECT_EQ(out.size(), block_size);
  // Reading at offset file_size + 4 will read the second block (since the read
  // at file_size + 4 = 28 will be aligned to an offset of 16) but will return
  // OutOfRange because the offset is past the end of the 24-byte file.
  Status status = ReadCache(&cache, "", file_size + 4, 4, &out);
  EXPECT_EQ(status.code(), error::OUT_OF_RANGE);
  EXPECT_TRUE(second_block);
  // Reading the second full block will return 8 bytes, from a cache hit.
  second_block = false;
  TF_EXPECT_OK(ReadCache(&cache, "", block_size, block_size, &out));
  EXPECT_FALSE(second_block);
  EXPECT_EQ(out.size(), file_size - block_size);
}

TEST(RamFileBlockCacheTest, Inconsistent) {
  // Tests the detection of interrupted reads leading to partially filled blocks
  // where we expected complete blocks.
  const size_t block_size = 16;
  // This fetcher returns OK but only fills in one byte for any offset.
  auto fetcher = [block_size](const string& filename, size_t offset, size_t n,
                              char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("filename: \"" + filename + "\"");
   mht_7_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_7(mht_7_v, 457, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    EXPECT_EQ(n, block_size);
    EXPECT_EQ(offset % block_size, 0);
    EXPECT_GE(n, 1);
    memset(buffer, 'x', 1);
    *bytes_transferred = 1;
    return Status::OK();
  };
  RamFileBlockCache cache(block_size, 2 * block_size, 0, fetcher);
  std::vector<char> out;
  // Read the second block; this should yield an OK status and a single byte.
  TF_EXPECT_OK(ReadCache(&cache, "", block_size, block_size, &out));
  EXPECT_EQ(out.size(), 1);
  // Now read the first block; this should yield an INTERNAL error because we
  // had already cached a partial block at a later position.
  Status status = ReadCache(&cache, "", 0, block_size, &out);
  EXPECT_EQ(status.code(), error::INTERNAL);
}

TEST(RamFileBlockCacheTest, LRU) {
  const size_t block_size = 16;
  std::list<size_t> calls;
  auto fetcher = [&calls, block_size](const string& filename, size_t offset,
                                      size_t n, char* buffer,
                                      size_t* bytes_transferred) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("filename: \"" + filename + "\"");
   mht_8_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_8(mht_8_v, 486, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    EXPECT_EQ(n, block_size);
    EXPECT_FALSE(calls.empty()) << "at offset = " << offset;
    if (!calls.empty()) {
      EXPECT_EQ(offset, calls.front());
      calls.pop_front();
    }
    memset(buffer, 'x', n);
    *bytes_transferred = n;
    return Status::OK();
  };
  const uint32 block_count = 2;
  RamFileBlockCache cache(block_size, block_count * block_size, 0, fetcher);
  std::vector<char> out;
  // Read blocks from the cache, and verify the LRU behavior based on the
  // fetcher calls that the cache makes.
  calls.push_back(0);
  // Cache miss - drains an element from `calls`.
  TF_EXPECT_OK(ReadCache(&cache, "", 0, 1, &out));
  // Cache hit - does not drain an element from `calls`.
  TF_EXPECT_OK(ReadCache(&cache, "", 0, 1, &out));
  calls.push_back(block_size);
  // Cache miss followed by cache hit.
  TF_EXPECT_OK(ReadCache(&cache, "", block_size, 1, &out));
  TF_EXPECT_OK(ReadCache(&cache, "", block_size, 1, &out));
  calls.push_back(2 * block_size);
  // Cache miss followed by cache hit.  Causes eviction of LRU element.
  TF_EXPECT_OK(ReadCache(&cache, "", 2 * block_size, 1, &out));
  TF_EXPECT_OK(ReadCache(&cache, "", 2 * block_size, 1, &out));
  // LRU element was at offset 0.  Cache miss.
  calls.push_back(0);
  TF_EXPECT_OK(ReadCache(&cache, "", 0, 1, &out));
  // Element at 2 * block_size is still in cache, and this read should update
  // its position in the LRU list so it doesn't get evicted by the next read.
  TF_EXPECT_OK(ReadCache(&cache, "", 2 * block_size, 1, &out));
  // Element at block_size was evicted.  Reading this element will also cause
  // the LRU element (at 0) to be evicted.
  calls.push_back(block_size);
  TF_EXPECT_OK(ReadCache(&cache, "", block_size, 1, &out));
  // Element at 0 was evicted again.
  calls.push_back(0);
  TF_EXPECT_OK(ReadCache(&cache, "", 0, 1, &out));
}

TEST(RamFileBlockCacheTest, MaxStaleness) {
  int calls = 0;
  auto fetcher = [&calls](const string& filename, size_t offset, size_t n,
                          char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("filename: \"" + filename + "\"");
   mht_9_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_9(mht_9_v, 538, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    calls++;
    memset(buffer, 'x', n);
    *bytes_transferred = n;
    return Status::OK();
  };
  std::vector<char> out;
  std::unique_ptr<NowSecondsEnv> env(new NowSecondsEnv);
  // Create a cache with max staleness of 2 seconds, and verify that it works as
  // expected.
  RamFileBlockCache cache1(8, 16, 2 /* max staleness */, fetcher, env.get());
  // Execute the first read to load the block.
  TF_EXPECT_OK(ReadCache(&cache1, "", 0, 1, &out));
  EXPECT_EQ(calls, 1);
  // Now advance the clock one second at a time and redo the read. The call
  // count should advance every 3 seconds (i.e. every time the staleness is
  // greater than 2).
  for (int i = 1; i <= 10; i++) {
    env->SetNowSeconds(i + 1);
    TF_EXPECT_OK(ReadCache(&cache1, "", 0, 1, &out));
    EXPECT_EQ(calls, 1 + i / 3);
  }
  // Now create a cache with max staleness of 0, and verify that it also works
  // as expected.
  calls = 0;
  env->SetNowSeconds(0);
  RamFileBlockCache cache2(8, 16, 0 /* max staleness */, fetcher, env.get());
  // Execute the first read to load the block.
  TF_EXPECT_OK(ReadCache(&cache2, "", 0, 1, &out));
  EXPECT_EQ(calls, 1);
  // Advance the clock by a huge amount and verify that the cached block is
  // used to satisfy the read.
  env->SetNowSeconds(365 * 24 * 60 * 60);  // ~1 year, just for fun.
  TF_EXPECT_OK(ReadCache(&cache2, "", 0, 1, &out));
  EXPECT_EQ(calls, 1);
}

TEST(RamFileBlockCacheTest, RemoveFile) {
  int calls = 0;
  auto fetcher = [&calls](const string& filename, size_t offset, size_t n,
                          char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("filename: \"" + filename + "\"");
   mht_10_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_10(mht_10_v, 583, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    calls++;
    char c = (filename == "a") ? 'a' : (filename == "b") ? 'b' : 'x';
    if (offset > 0) {
      // The first block is lower case and all subsequent blocks are upper case.
      c = toupper(c);
    }
    memset(buffer, c, n);
    *bytes_transferred = n;
    return Status::OK();
  };
  // This cache has space for 4 blocks; we'll read from two files.
  const size_t n = 3;
  RamFileBlockCache cache(8, 32, 0, fetcher);
  std::vector<char> out;
  std::vector<char> a(n, 'a');
  std::vector<char> b(n, 'b');
  std::vector<char> A(n, 'A');
  std::vector<char> B(n, 'B');
  // Fill the cache.
  TF_EXPECT_OK(ReadCache(&cache, "a", 0, n, &out));
  EXPECT_EQ(out, a);
  EXPECT_EQ(calls, 1);
  TF_EXPECT_OK(ReadCache(&cache, "a", 8, n, &out));
  EXPECT_EQ(out, A);
  EXPECT_EQ(calls, 2);
  TF_EXPECT_OK(ReadCache(&cache, "b", 0, n, &out));
  EXPECT_EQ(out, b);
  EXPECT_EQ(calls, 3);
  TF_EXPECT_OK(ReadCache(&cache, "b", 8, n, &out));
  EXPECT_EQ(out, B);
  EXPECT_EQ(calls, 4);
  // All four blocks should be in the cache now.
  TF_EXPECT_OK(ReadCache(&cache, "a", 0, n, &out));
  EXPECT_EQ(out, a);
  TF_EXPECT_OK(ReadCache(&cache, "a", 8, n, &out));
  EXPECT_EQ(out, A);
  TF_EXPECT_OK(ReadCache(&cache, "b", 0, n, &out));
  EXPECT_EQ(out, b);
  TF_EXPECT_OK(ReadCache(&cache, "b", 8, n, &out));
  EXPECT_EQ(out, B);
  EXPECT_EQ(calls, 4);
  // Remove the blocks from "a".
  cache.RemoveFile("a");
  // Both blocks from "b" should still be there.
  TF_EXPECT_OK(ReadCache(&cache, "b", 0, n, &out));
  EXPECT_EQ(out, b);
  TF_EXPECT_OK(ReadCache(&cache, "b", 8, n, &out));
  EXPECT_EQ(out, B);
  EXPECT_EQ(calls, 4);
  // The blocks from "a" should not be there.
  TF_EXPECT_OK(ReadCache(&cache, "a", 0, n, &out));
  EXPECT_EQ(out, a);
  EXPECT_EQ(calls, 5);
  TF_EXPECT_OK(ReadCache(&cache, "a", 8, n, &out));
  EXPECT_EQ(out, A);
  EXPECT_EQ(calls, 6);
}

TEST(RamFileBlockCacheTest, Prune) {
  int calls = 0;
  auto fetcher = [&calls](const string& filename, size_t offset, size_t n,
                          char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("filename: \"" + filename + "\"");
   mht_11_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_11(mht_11_v, 650, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    calls++;
    memset(buffer, 'x', n);
    *bytes_transferred = n;
    return Status::OK();
  };
  std::vector<char> out;
  // Our fake environment is initialized with the current timestamp.
  std::unique_ptr<NowSecondsEnv> env(new NowSecondsEnv);
  uint64 now = Env::Default()->NowSeconds();
  env->SetNowSeconds(now);
  RamFileBlockCache cache(8, 32, 1 /* max staleness */, fetcher, env.get());
  // Read three blocks into the cache, and advance the timestamp by one second
  // with each read. Start with a block of "a" at the current timestamp `now`.
  TF_EXPECT_OK(ReadCache(&cache, "a", 0, 1, &out));
  // Now load a block of a different file "b" at timestamp `now` + 1
  env->SetNowSeconds(now + 1);
  TF_EXPECT_OK(ReadCache(&cache, "b", 0, 1, &out));
  // Now load a different block of file "a" at timestamp `now` + 1. When the
  // first block of "a" expires, this block should also be removed because it
  // also belongs to file "a".
  TF_EXPECT_OK(ReadCache(&cache, "a", 8, 1, &out));
  // Ensure that all blocks are in the cache (i.e. reads are cache hits).
  EXPECT_EQ(cache.CacheSize(), 24);
  EXPECT_EQ(calls, 3);
  TF_EXPECT_OK(ReadCache(&cache, "a", 0, 1, &out));
  TF_EXPECT_OK(ReadCache(&cache, "b", 0, 1, &out));
  TF_EXPECT_OK(ReadCache(&cache, "a", 8, 1, &out));
  EXPECT_EQ(calls, 3);
  // Advance the fake timestamp so that "a" becomes stale via its first block.
  env->SetNowSeconds(now + 2);
  // The pruning thread periodically compares env->NowSeconds() with the oldest
  // block's timestamp to see if it should evict any files. At the current fake
  // timestamp of `now` + 2, file "a" is stale because its first block is stale,
  // but file "b" is not stale yet. Thus, once the pruning thread wakes up (in
  // one second of wall time), it should remove "a" and leave "b" alone.
  uint64 start = Env::Default()->NowSeconds();
  do {
    Env::Default()->SleepForMicroseconds(100000);
  } while (cache.CacheSize() == 24 && Env::Default()->NowSeconds() - start < 3);
  // There should be one block left in the cache, and it should be the first
  // block of "b".
  EXPECT_EQ(cache.CacheSize(), 8);
  TF_EXPECT_OK(ReadCache(&cache, "b", 0, 1, &out));
  EXPECT_EQ(calls, 3);
  // Advance the fake time to `now` + 3, at which point "b" becomes stale.
  env->SetNowSeconds(now + 3);
  // Wait for the pruner to remove "b".
  start = Env::Default()->NowSeconds();
  do {
    Env::Default()->SleepForMicroseconds(100000);
  } while (cache.CacheSize() == 8 && Env::Default()->NowSeconds() - start < 3);
  // The cache should now be empty.
  EXPECT_EQ(cache.CacheSize(), 0);
}

TEST(RamFileBlockCacheTest, ParallelReads) {
  // This fetcher won't respond until either `callers` threads are calling it
  // concurrently (at which point it will respond with success to all callers),
  // or 10 seconds have elapsed (at which point it will respond with an error).
  const int callers = 4;
  BlockingCounter counter(callers);
  auto fetcher = [&counter](const string& filename, size_t offset, size_t n,
                            char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("filename: \"" + filename + "\"");
   mht_12_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_12(mht_12_v, 718, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    counter.DecrementCount();
    if (!counter.WaitFor(std::chrono::seconds(10))) {
      // This avoids having the test time out, which is harder to debug.
      return errors::FailedPrecondition("desired concurrency not reached");
    }
    memset(buffer, 'x', n);
    *bytes_transferred = n;
    return Status::OK();
  };
  const int block_size = 8;
  RamFileBlockCache cache(block_size, 2 * callers * block_size, 0, fetcher);
  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < callers; i++) {
    threads.emplace_back(
        Env::Default()->StartThread({}, "caller", [&cache, i, block_size]() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_13(mht_13_v, 736, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

          std::vector<char> out;
          TF_EXPECT_OK(
              ReadCache(&cache, "a", i * block_size, block_size, &out));
          std::vector<char> x(block_size, 'x');
          EXPECT_EQ(out, x);
        }));
  }
  // The `threads` destructor blocks until the threads can be joined, once their
  // respective reads finish (which happens once they are all concurrently being
  // executed, or 10 seconds have passed).
}

TEST(RamFileBlockCacheTest, CoalesceConcurrentReads) {
  // Concurrent reads to the same file blocks should be de-duplicated.
  const size_t block_size = 16;
  int num_requests = 0;
  Notification notification;
  auto fetcher = [&num_requests, &notification, block_size](
                     const string& filename, size_t offset, size_t n,
                     char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("filename: \"" + filename + "\"");
   mht_14_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_14(mht_14_v, 761, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    EXPECT_EQ(n, block_size);
    EXPECT_EQ(offset, 0);
    num_requests++;
    memset(buffer, 'x', n);
    *bytes_transferred = n;
    notification.Notify();
    // Wait for other thread to issue read.
    Env::Default()->SleepForMicroseconds(100000);  // 0.1 secs
    return Status::OK();
  };
  RamFileBlockCache cache(block_size, block_size, 0, fetcher);
  // Fork off thread for parallel read.
  std::unique_ptr<Thread> concurrent(
      Env::Default()->StartThread({}, "concurrent", [&cache, block_size] {
        std::vector<char> out;
        TF_EXPECT_OK(ReadCache(&cache, "", 0, block_size / 2, &out));
        EXPECT_EQ(out.size(), block_size / 2);
      }));
  notification.WaitForNotification();
  std::vector<char> out;
  TF_EXPECT_OK(ReadCache(&cache, "", block_size / 2, block_size / 2, &out));
  EXPECT_EQ(out.size(), block_size / 2);

  EXPECT_EQ(1, num_requests);
}

TEST(RamFileBlockCacheTest, Flush) {
  int calls = 0;
  auto fetcher = [&calls](const string& filename, size_t offset, size_t n,
                          char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("filename: \"" + filename + "\"");
   mht_15_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cache_testDTcc mht_15(mht_15_v, 796, "", "./tensorflow/core/platform/cloud/ram_file_block_cache_test.cc", "lambda");

    calls++;
    memset(buffer, 'x', n);
    *bytes_transferred = n;
    return Status::OK();
  };
  RamFileBlockCache cache(16, 32, 0, fetcher);
  std::vector<char> out;
  TF_EXPECT_OK(ReadCache(&cache, "", 0, 16, &out));
  TF_EXPECT_OK(ReadCache(&cache, "", 0, 16, &out));
  EXPECT_EQ(calls, 1);
  cache.Flush();
  TF_EXPECT_OK(ReadCache(&cache, "", 0, 16, &out));
  EXPECT_EQ(calls, 2);
}

}  // namespace
}  // namespace tensorflow
