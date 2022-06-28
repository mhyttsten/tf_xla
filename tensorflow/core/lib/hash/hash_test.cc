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
class MHTracer_DTPStensorflowPScorePSlibPShashPShash_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPShashPShash_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPShashPShash_testDTcc() {
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

#include <map>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

TEST(Hash, SignedUnsignedIssue) {
  const unsigned char d1[1] = {0x62};
  const unsigned char d2[2] = {0xc3, 0x97};
  const unsigned char d3[3] = {0xe2, 0x99, 0xa5};
  const unsigned char d4[4] = {0xe1, 0x80, 0xb9, 0x32};
  const unsigned char d5[48] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };

  struct Case {
    uint32 hash32;
    uint64 hash64;
    const unsigned char* data;
    size_t size;
    uint32 seed;
  };

  for (Case c : std::vector<Case>{
           {0x471a8188u, 0x4c61ea3eeda4cb87ull, nullptr, 0, 0xbc9f1d34},
           {0xd615eba5u, 0x091309f7ef916c8aull, d1, sizeof(d1), 0xbc9f1d34},
           {0x0c3cccdau, 0xa815bcdf1d1af01cull, d2, sizeof(d2), 0xbc9f1d34},
           {0x3ba37e0eu, 0x02167564e4d06430ull, d3, sizeof(d3), 0xbc9f1d34},
           {0x16174eb3u, 0x8f7ed82ffc21071full, d4, sizeof(d4), 0xbc9f1d34},
           {0x98b1926cu, 0xce196580c97aff1eull, d5, sizeof(d5), 0x12345678},
       }) {
    EXPECT_EQ(c.hash32,
              Hash32(reinterpret_cast<const char*>(c.data), c.size, c.seed));
    EXPECT_EQ(c.hash64,
              Hash64(reinterpret_cast<const char*>(c.data), c.size, c.seed));

    // Check hashes with inputs aligned differently.
    for (int align = 1; align <= 7; align++) {
      std::string input(align, 'x');
      input.append(reinterpret_cast<const char*>(c.data), c.size);
      EXPECT_EQ(c.hash32, Hash32(&input[align], c.size, c.seed));
      EXPECT_EQ(c.hash64, Hash64(&input[align], c.size, c.seed));
    }
  }
}

TEST(Hash, HashPtrIsNotIdentityFunction) {
  int* ptr = reinterpret_cast<int*>(0xcafe0000);
  EXPECT_NE(hash<int*>()(ptr), size_t{0xcafe0000});
}

static void BM_Hash32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPShashPShash_testDTcc mht_0(mht_0_v, 244, "", "./tensorflow/core/lib/hash/hash_test.cc", "BM_Hash32");

  int len = state.range(0);
  std::string input(len, 'x');
  uint32 h = 0;
  for (auto s : state) {
    h = Hash32(input.data(), len, 1);
  }
  state.SetBytesProcessed(state.iterations() * len);
  VLOG(1) << h;
}
BENCHMARK(BM_Hash32)->Range(1, 1024);

TEST(StringPieceHasher, Equality) {
  StringPieceHasher hasher;

  StringPiece s1("foo");
  StringPiece s2("bar");
  StringPiece s3("baz");
  StringPiece s4("zot");

  EXPECT_TRUE(hasher(s1) != hasher(s2));
  EXPECT_TRUE(hasher(s1) != hasher(s3));
  EXPECT_TRUE(hasher(s1) != hasher(s4));
  EXPECT_TRUE(hasher(s2) != hasher(s3));
  EXPECT_TRUE(hasher(s2) != hasher(s4));
  EXPECT_TRUE(hasher(s3) != hasher(s4));

  EXPECT_TRUE(hasher(s1) == hasher(s1));
  EXPECT_TRUE(hasher(s2) == hasher(s2));
  EXPECT_TRUE(hasher(s3) == hasher(s3));
  EXPECT_TRUE(hasher(s4) == hasher(s4));
}

TEST(StringPieceHasher, HashMap) {
  string s1("foo");
  string s2("bar");
  string s3("baz");

  StringPiece p1(s1);
  StringPiece p2(s2);
  StringPiece p3(s3);

  std::unordered_map<StringPiece, int, StringPieceHasher> map;

  map.insert(std::make_pair(p1, 0));
  map.insert(std::make_pair(p2, 1));
  map.insert(std::make_pair(p3, 2));
  EXPECT_EQ(map.size(), 3);

  bool found[3] = {false, false, false};
  for (auto const& val : map) {
    int x = val.second;
    EXPECT_TRUE(x >= 0 && x < 3);
    EXPECT_TRUE(!found[x]);
    found[x] = true;
  }
  EXPECT_EQ(found[0], true);
  EXPECT_EQ(found[1], true);
  EXPECT_EQ(found[2], true);

  auto new_iter = map.find("zot");
  EXPECT_TRUE(new_iter == map.end());

  new_iter = map.find("bar");
  EXPECT_TRUE(new_iter != map.end());

  map.erase(new_iter);
  EXPECT_EQ(map.size(), 2);

  found[0] = false;
  found[1] = false;
  found[2] = false;
  for (const auto& iter : map) {
    int x = iter.second;
    EXPECT_TRUE(x >= 0 && x < 3);
    EXPECT_TRUE(!found[x]);
    found[x] = true;
  }
  EXPECT_EQ(found[0], true);
  EXPECT_EQ(found[1], false);
  EXPECT_EQ(found[2], true);
}

}  // namespace tensorflow
