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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmap_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmap_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmap_testDTcc() {
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

#include "tensorflow/core/lib/gtl/flatmap.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gtl {
namespace {

typedef FlatMap<int64_t, int32> NumMap;

// If map has an entry for k, return the corresponding value, else return def.
int32 Get(const NumMap& map, int64_t k, int32_t def = -1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmap_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/lib/gtl/flatmap_test.cc", "Get");

  auto iter = map.find(k);
  if (iter == map.end()) {
    EXPECT_EQ(map.count(k), 0);
    return def;
  } else {
    EXPECT_EQ(map.count(k), 1);
    EXPECT_EQ(&map.at(k), &iter->second);
    EXPECT_EQ(iter->first, k);
    return iter->second;
  }
}

// Return contents of map as a sorted list of pairs.
typedef std::vector<std::pair<int64_t, int32>> NumMapContents;
NumMapContents Contents(const NumMap& map) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmap_testDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/lib/gtl/flatmap_test.cc", "Contents");

  NumMapContents result;
  for (const auto& p : map) {
    result.push_back({p.first, p.second});
  }
  std::sort(result.begin(), result.end());
  return result;
}

// Fill entries with keys [start,limit).
void Fill(NumMap* map, int64_t start, int64_t limit) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmap_testDTcc mht_2(mht_2_v, 233, "", "./tensorflow/core/lib/gtl/flatmap_test.cc", "Fill");

  for (int64_t i = start; i < limit; i++) {
    map->insert({i, i * 100});
  }
}

TEST(FlatMapTest, Find) {
  NumMap map;
  EXPECT_EQ(Get(map, 1), -1);
  map.insert({1, 100});
  map.insert({2, 200});
  EXPECT_EQ(Get(map, 1), 100);
  EXPECT_EQ(Get(map, 2), 200);
  EXPECT_EQ(Get(map, 3), -1);
}

TEST(FlatMapTest, Insert) {
  NumMap map;
  EXPECT_EQ(Get(map, 1), -1);

  // New entry.
  auto result = map.insert({1, 100});
  EXPECT_TRUE(result.second);
  EXPECT_EQ(result.first->first, 1);
  EXPECT_EQ(result.first->second, 100);
  EXPECT_EQ(Get(map, 1), 100);

  // Attempt to insert over existing entry.
  result = map.insert({1, 200});
  EXPECT_FALSE(result.second);
  EXPECT_EQ(result.first->first, 1);
  EXPECT_EQ(result.first->second, 100);
  EXPECT_EQ(Get(map, 1), 100);

  // Overwrite through iterator.
  result.first->second = 300;
  EXPECT_EQ(result.first->second, 300);
  EXPECT_EQ(Get(map, 1), 300);

  // Should get updated value.
  result = map.insert({1, 400});
  EXPECT_FALSE(result.second);
  EXPECT_EQ(result.first->first, 1);
  EXPECT_EQ(result.first->second, 300);
  EXPECT_EQ(Get(map, 1), 300);
}

TEST(FlatMapTest, InsertGrowth) {
  NumMap map;
  const int n = 100;
  Fill(&map, 0, 100);
  EXPECT_EQ(map.size(), n);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(Get(map, i), i * 100) << i;
  }
}

TEST(FlatMapTest, Emplace) {
  NumMap map;

  // New entry.
  auto result = map.emplace(1, 100);
  EXPECT_TRUE(result.second);
  EXPECT_EQ(result.first->first, 1);
  EXPECT_EQ(result.first->second, 100);
  EXPECT_EQ(Get(map, 1), 100);

  // Attempt to insert over existing entry.
  result = map.emplace(1, 200);
  EXPECT_FALSE(result.second);
  EXPECT_EQ(result.first->first, 1);
  EXPECT_EQ(result.first->second, 100);
  EXPECT_EQ(Get(map, 1), 100);

  // Overwrite through iterator.
  result.first->second = 300;
  EXPECT_EQ(result.first->second, 300);
  EXPECT_EQ(Get(map, 1), 300);

  // Update a second value
  result = map.emplace(2, 400);
  EXPECT_TRUE(result.second);
  EXPECT_EQ(result.first->first, 2);
  EXPECT_EQ(result.first->second, 400);
  EXPECT_EQ(Get(map, 2), 400);
}

TEST(FlatMapTest, EmplaceUniquePtr) {
  FlatMap<int64_t, std::unique_ptr<string>> smap;
  smap.emplace(1, std::unique_ptr<string>(new string("hello")));
}

TEST(FlatMapTest, Size) {
  NumMap map;
  EXPECT_EQ(map.size(), 0);

  map.insert({1, 100});
  map.insert({2, 200});
  EXPECT_EQ(map.size(), 2);
}

TEST(FlatMapTest, Empty) {
  NumMap map;
  EXPECT_TRUE(map.empty());

  map.insert({1, 100});
  map.insert({2, 200});
  EXPECT_FALSE(map.empty());
}

TEST(FlatMapTest, ArrayOperator) {
  NumMap map;

  // Create new element if not found.
  auto v1 = &map[1];
  EXPECT_EQ(*v1, 0);
  EXPECT_EQ(Get(map, 1), 0);

  // Write through returned reference.
  *v1 = 100;
  EXPECT_EQ(map[1], 100);
  EXPECT_EQ(Get(map, 1), 100);

  // Reuse existing element if found.
  auto v1a = &map[1];
  EXPECT_EQ(v1, v1a);
  EXPECT_EQ(*v1, 100);

  // Create another element.
  map[2] = 200;
  EXPECT_EQ(Get(map, 1), 100);
  EXPECT_EQ(Get(map, 2), 200);
}

TEST(FlatMapTest, Count) {
  NumMap map;
  EXPECT_EQ(map.count(1), 0);
  EXPECT_EQ(map.count(2), 0);

  map.insert({1, 100});
  EXPECT_EQ(map.count(1), 1);
  EXPECT_EQ(map.count(2), 0);

  map.insert({2, 200});
  EXPECT_EQ(map.count(1), 1);
  EXPECT_EQ(map.count(2), 1);
}

TEST(FlatMapTest, Iter) {
  NumMap map;
  EXPECT_EQ(Contents(map), NumMapContents());

  map.insert({1, 100});
  map.insert({2, 200});
  EXPECT_EQ(Contents(map), NumMapContents({{1, 100}, {2, 200}}));
}

TEST(FlatMapTest, Erase) {
  NumMap map;
  EXPECT_EQ(map.erase(1), 0);
  map[1] = 100;
  map[2] = 200;
  EXPECT_EQ(map.erase(3), 0);
  EXPECT_EQ(map.erase(1), 1);
  EXPECT_EQ(map.size(), 1);
  EXPECT_EQ(Get(map, 2), 200);
  EXPECT_EQ(Contents(map), NumMapContents({{2, 200}}));
  EXPECT_EQ(map.erase(2), 1);
  EXPECT_EQ(Contents(map), NumMapContents());
}

TEST(FlatMapTest, EraseIter) {
  NumMap map;
  Fill(&map, 1, 11);
  size_t size = 10;
  for (auto iter = map.begin(); iter != map.end();) {
    iter = map.erase(iter);
    size--;
    EXPECT_EQ(map.size(), size);
  }
  EXPECT_EQ(Contents(map), NumMapContents());
}

TEST(FlatMapTest, EraseIterPair) {
  NumMap map;
  Fill(&map, 1, 11);
  NumMap expected;
  auto p1 = map.begin();
  expected.insert(*p1);
  ++p1;
  expected.insert(*p1);
  ++p1;
  auto p2 = map.end();
  EXPECT_EQ(map.erase(p1, p2), map.end());
  EXPECT_EQ(map.size(), 2);
  EXPECT_EQ(Contents(map), Contents(expected));
}

TEST(FlatMapTest, EraseLongChains) {
  // Make a map with lots of elements and erase a bunch of them to ensure
  // that we are likely to hit them on future lookups.
  NumMap map;
  const int num = 128;
  Fill(&map, 0, num);
  for (int i = 0; i < num; i += 3) {
    EXPECT_EQ(map.erase(i), 1);
  }
  for (int i = 0; i < num; i++) {
    if ((i % 3) != 0) {
      EXPECT_EQ(Get(map, i), i * 100);
    } else {
      EXPECT_EQ(map.count(i), 0);
    }
  }

  // Erase remainder to trigger table shrinking.
  const size_t orig_buckets = map.bucket_count();
  for (int i = 0; i < num; i++) {
    map.erase(i);
  }
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(map.bucket_count(), orig_buckets);
  map[1] = 100;  // Actual shrinking is triggered by an insert.
  EXPECT_LT(map.bucket_count(), orig_buckets);
}

TEST(FlatMap, AlternatingInsertRemove) {
  NumMap map;
  map.insert({1000, 1000});
  map.insert({2000, 1000});
  map.insert({3000, 1000});
  for (int i = 0; i < 10000; i++) {
    map.insert({i, i});
    map.erase(i);
  }
}

TEST(FlatMap, ClearNoResize) {
  NumMap map;
  Fill(&map, 0, 100);
  const size_t orig = map.bucket_count();
  map.clear_no_resize();
  EXPECT_EQ(map.size(), 0);
  EXPECT_EQ(Contents(map), NumMapContents());
  EXPECT_EQ(map.bucket_count(), orig);
}

TEST(FlatMap, Clear) {
  NumMap map;
  Fill(&map, 0, 100);
  const size_t orig = map.bucket_count();
  map.clear();
  EXPECT_EQ(map.size(), 0);
  EXPECT_EQ(Contents(map), NumMapContents());
  EXPECT_LT(map.bucket_count(), orig);
}

TEST(FlatMap, Copy) {
  for (int n = 0; n < 10; n++) {
    NumMap src;
    Fill(&src, 0, n);
    NumMap copy = src;
    EXPECT_EQ(Contents(src), Contents(copy));
    NumMap copy2;
    copy2 = src;
    EXPECT_EQ(Contents(src), Contents(copy2));
    copy2 = *&copy2;  // Self-assignment, avoiding -Wself-assign.
    EXPECT_EQ(Contents(src), Contents(copy2));
  }
}

TEST(FlatMap, InitFromIter) {
  for (int n = 0; n < 10; n++) {
    NumMap src;
    Fill(&src, 0, n);
    auto vec = Contents(src);
    NumMap dst(vec.begin(), vec.end());
    EXPECT_EQ(Contents(dst), vec);
  }
}

TEST(FlatMap, InitializerList) {
  NumMap a{{1, 10}, {2, 20}, {3, 30}};
  NumMap b({{1, 10}, {2, 20}, {3, 30}});
  NumMap c = {{1, 10}, {2, 20}, {3, 30}};

  typedef std::unordered_map<int64_t, int32> StdNumMap;
  StdNumMap std({{1, 10}, {2, 20}, {3, 30}});
  StdNumMap::value_type std_r1 = *std.find(1);
  StdNumMap::value_type std_r2 = *std.find(2);
  StdNumMap::value_type std_r3 = *std.find(3);
  NumMap d{std_r1, std_r2, std_r3};
  NumMap e({std_r1, std_r2, std_r3});
  NumMap f = {std_r1, std_r2, std_r3};

  for (NumMap* map : std::vector<NumMap*>({&a, &b, &c, &d, &e, &f})) {
    EXPECT_EQ(Get(*map, 1), 10);
    EXPECT_EQ(Get(*map, 2), 20);
    EXPECT_EQ(Get(*map, 3), 30);
    EXPECT_EQ(Contents(*map), NumMapContents({{1, 10}, {2, 20}, {3, 30}}));
  }
}

TEST(FlatMap, InsertIter) {
  NumMap a, b;
  Fill(&a, 1, 10);
  Fill(&b, 8, 20);
  b[9] = 10000;  // Should not get inserted into a since a already has 9
  a.insert(b.begin(), b.end());
  NumMap expected;
  Fill(&expected, 1, 20);
  EXPECT_EQ(Contents(a), Contents(expected));
}

TEST(FlatMap, Eq) {
  NumMap empty;

  NumMap elems;
  Fill(&elems, 0, 5);
  EXPECT_FALSE(empty == elems);
  EXPECT_TRUE(empty != elems);

  NumMap copy = elems;
  EXPECT_TRUE(copy == elems);
  EXPECT_FALSE(copy != elems);

  NumMap changed = elems;
  changed[3] = 1;
  EXPECT_FALSE(changed == elems);
  EXPECT_TRUE(changed != elems);

  NumMap changed2 = elems;
  changed2.erase(3);
  EXPECT_FALSE(changed2 == elems);
  EXPECT_TRUE(changed2 != elems);
}

TEST(FlatMap, Swap) {
  NumMap a, b;
  Fill(&a, 1, 5);
  Fill(&b, 100, 200);
  NumMap c = a;
  NumMap d = b;
  EXPECT_EQ(c, a);
  EXPECT_EQ(d, b);
  c.swap(d);
  EXPECT_EQ(c, b);
  EXPECT_EQ(d, a);
}

TEST(FlatMap, Reserve) {
  NumMap src;
  Fill(&src, 1, 100);
  NumMap a = src;
  a.reserve(10);
  EXPECT_EQ(a, src);
  NumMap b = src;
  b.rehash(1000);
  EXPECT_EQ(b, src);
}

TEST(FlatMap, EqualRangeMutable) {
  NumMap map;
  Fill(&map, 1, 10);

  // Existing element
  auto p1 = map.equal_range(3);
  EXPECT_TRUE(p1.first != p1.second);
  EXPECT_EQ(p1.first->first, 3);
  EXPECT_EQ(p1.first->second, 300);
  ++p1.first;
  EXPECT_TRUE(p1.first == p1.second);

  // Missing element
  auto p2 = map.equal_range(100);
  EXPECT_TRUE(p2.first == p2.second);
}

TEST(FlatMap, EqualRangeConst) {
  NumMap tmp;
  Fill(&tmp, 1, 10);

  const NumMap map = tmp;

  // Existing element
  auto p1 = map.equal_range(3);
  EXPECT_TRUE(p1.first != p1.second);
  EXPECT_EQ(p1.first->first, 3);
  EXPECT_EQ(p1.first->second, 300);
  ++p1.first;
  EXPECT_TRUE(p1.first == p1.second);

  // Missing element
  auto p2 = map.equal_range(100);
  EXPECT_TRUE(p2.first == p2.second);
}

TEST(FlatMap, Prefetch) {
  NumMap map;
  Fill(&map, 0, 1000);
  // Prefetch present and missing keys.
  for (int i = 0; i < 2000; i++) {
    map.prefetch_value(i);
  }
}

// Non-assignable values should work.
struct NA {
  int64_t value;
  NA() : value(-1) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmap_testDTcc mht_3(mht_3_v, 645, "", "./tensorflow/core/lib/gtl/flatmap_test.cc", "NA");
}
  explicit NA(int64_t v) : value(v) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmap_testDTcc mht_4(mht_4_v, 649, "", "./tensorflow/core/lib/gtl/flatmap_test.cc", "NA");
}
  NA(const NA& x) : value(x.value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmap_testDTcc mht_5(mht_5_v, 653, "", "./tensorflow/core/lib/gtl/flatmap_test.cc", "NA");
}
  bool operator==(const NA& x) const { return value == x.value; }
};
struct HashNA {
  size_t operator()(NA x) const { return x.value; }
};

TEST(FlatMap, NonAssignable) {
  FlatMap<NA, NA, HashNA> map;
  for (int i = 0; i < 100; i++) {
    map[NA(i)] = NA(i * 100);
  }
  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(map.count(NA(i)), 1);
    auto iter = map.find(NA(i));
    EXPECT_NE(iter, map.end());
    EXPECT_EQ(iter->first, NA(i));
    EXPECT_EQ(iter->second, NA(i * 100));
    EXPECT_EQ(map[NA(i)], NA(i * 100));
  }
  map.erase(NA(10));
  EXPECT_EQ(map.count(NA(10)), 0);
}

TEST(FlatMap, ForwardIterator) {
  // Test the requirements of forward iterators
  typedef FlatMap<NA, NA, HashNA> NAMap;
  NAMap map({{NA(1), NA(10)}, {NA(2), NA(20)}});
  NAMap::iterator it1 = map.find(NA(1));
  NAMap::iterator it2 = map.find(NA(2));

  // Test operator != and ==
  EXPECT_TRUE(it1 != map.end());
  EXPECT_TRUE(it2 != map.end());
  EXPECT_FALSE(it1 == map.end());
  EXPECT_FALSE(it2 == map.end());
  EXPECT_TRUE(it1 != it2);
  EXPECT_FALSE(it1 == it2);

  // Test operator * and ->
  EXPECT_EQ((*it1).first, NA(1));
  EXPECT_EQ((*it1).second, NA(10));
  EXPECT_EQ((*it2).first, NA(2));
  EXPECT_EQ((*it2).second, NA(20));
  EXPECT_EQ(it1->first, NA(1));
  EXPECT_EQ(it1->second, NA(10));
  EXPECT_EQ(it2->first, NA(2));
  EXPECT_EQ(it2->second, NA(20));

  // Test prefix ++
  NAMap::iterator copy_it1 = it1;
  NAMap::iterator copy_it2 = it2;
  EXPECT_EQ(copy_it1->first, NA(1));
  EXPECT_EQ(copy_it1->second, NA(10));
  EXPECT_EQ(copy_it2->first, NA(2));
  EXPECT_EQ(copy_it2->second, NA(20));
  NAMap::iterator& pp_copy_it1 = ++copy_it1;
  NAMap::iterator& pp_copy_it2 = ++copy_it2;
  EXPECT_TRUE(pp_copy_it1 == copy_it1);
  EXPECT_TRUE(pp_copy_it2 == copy_it2);
  // Check either possible ordering of the two items
  EXPECT_TRUE(copy_it1 != it1);
  EXPECT_TRUE(copy_it2 != it2);
  if (copy_it1 == map.end()) {
    EXPECT_TRUE(copy_it2 != map.end());
    EXPECT_EQ(copy_it2->first, NA(1));
    EXPECT_EQ(copy_it2->second, NA(10));
    EXPECT_EQ(pp_copy_it2->first, NA(1));
    EXPECT_EQ(pp_copy_it2->second, NA(10));
  } else {
    EXPECT_TRUE(copy_it2 == map.end());
    EXPECT_EQ(copy_it1->first, NA(2));
    EXPECT_EQ(copy_it1->second, NA(20));
    EXPECT_EQ(pp_copy_it1->first, NA(2));
    EXPECT_EQ(pp_copy_it1->second, NA(20));
  }
  // Ensure it{1,2} haven't moved
  EXPECT_EQ(it1->first, NA(1));
  EXPECT_EQ(it1->second, NA(10));
  EXPECT_EQ(it2->first, NA(2));
  EXPECT_EQ(it2->second, NA(20));

  // Test postfix ++
  copy_it1 = it1;
  copy_it2 = it2;
  EXPECT_EQ(copy_it1->first, NA(1));
  EXPECT_EQ(copy_it1->second, NA(10));
  EXPECT_EQ(copy_it2->first, NA(2));
  EXPECT_EQ(copy_it2->second, NA(20));
  NAMap::iterator copy_it1_pp = copy_it1++;
  NAMap::iterator copy_it2_pp = copy_it2++;
  EXPECT_TRUE(copy_it1_pp != copy_it1);
  EXPECT_TRUE(copy_it2_pp != copy_it2);
  EXPECT_TRUE(copy_it1_pp == it1);
  EXPECT_TRUE(copy_it2_pp == it2);
  EXPECT_EQ(copy_it1_pp->first, NA(1));
  EXPECT_EQ(copy_it1_pp->second, NA(10));
  EXPECT_EQ(copy_it2_pp->first, NA(2));
  EXPECT_EQ(copy_it2_pp->second, NA(20));
  // Check either possible ordering of the two items
  EXPECT_TRUE(copy_it1 != it1);
  EXPECT_TRUE(copy_it2 != it2);
  if (copy_it1 == map.end()) {
    EXPECT_TRUE(copy_it2 != map.end());
    EXPECT_EQ(copy_it2->first, NA(1));
    EXPECT_EQ(copy_it2->second, NA(10));
  } else {
    EXPECT_TRUE(copy_it2 == map.end());
    EXPECT_EQ(copy_it1->first, NA(2));
    EXPECT_EQ(copy_it1->second, NA(20));
  }
  // Ensure it{1,2} haven't moved
  EXPECT_EQ(it1->first, NA(1));
  EXPECT_EQ(it1->second, NA(10));
  EXPECT_EQ(it2->first, NA(2));
  EXPECT_EQ(it2->second, NA(20));
}

// Test with heap-allocated objects so that mismanaged constructions
// or destructions will show up as errors under a sanitizer or
// heap checker.
TEST(FlatMap, ConstructDestruct) {
  FlatMap<string, string> map;
  string k1 = "the quick brown fox jumped over the lazy dog";
  string k2 = k1 + k1;
  string k3 = k1 + k2;
  map[k1] = k2;
  map[k3] = k1;
  EXPECT_EQ(k1, map.find(k1)->first);
  EXPECT_EQ(k2, map.find(k1)->second);
  EXPECT_EQ(k1, map[k3]);
  map.erase(k3);
  EXPECT_EQ(string(), map[k3]);

  map.clear();
  map[k1] = k2;
  EXPECT_EQ(k2, map[k1]);

  map.reserve(100);
  EXPECT_EQ(k2, map[k1]);
}

// Type to use to ensure that custom equality operator is used
// that ignores extra value.
struct CustomCmpKey {
  int64_t a;
  int64_t b;
  CustomCmpKey(int64_t v1, int64_t v2) : a(v1), b(v2) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmap_testDTcc mht_6(mht_6_v, 803, "", "./tensorflow/core/lib/gtl/flatmap_test.cc", "CustomCmpKey");
}
  bool operator==(const CustomCmpKey& x) const { return a == x.a && b == x.b; }
};
struct HashA {
  size_t operator()(CustomCmpKey x) const { return x.a; }
};
struct EqA {
  // Ignore b fields.
  bool operator()(CustomCmpKey x, CustomCmpKey y) const { return x.a == y.a; }
};
TEST(FlatMap, CustomCmp) {
  FlatMap<CustomCmpKey, int, HashA, EqA> map;
  map[CustomCmpKey(100, 200)] = 300;
  EXPECT_EQ(300, map[CustomCmpKey(100, 200)]);
  EXPECT_EQ(300, map[CustomCmpKey(100, 500)]);  // Differences in key.b ignored
}

// Test unique_ptr handling.
typedef std::unique_ptr<int> UniqInt;
static UniqInt MakeUniq(int i) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatmap_testDTcc mht_7(mht_7_v, 825, "", "./tensorflow/core/lib/gtl/flatmap_test.cc", "MakeUniq");
 return UniqInt(new int(i)); }

struct HashUniq {
  size_t operator()(const UniqInt& p) const { return *p; }
};
struct EqUniq {
  bool operator()(const UniqInt& a, const UniqInt& b) const { return *a == *b; }
};
typedef FlatMap<UniqInt, UniqInt, HashUniq, EqUniq> UniqMap;

TEST(FlatMap, UniqueMap) {
  UniqMap map;

  // Fill map
  const int N = 10;
  for (int i = 0; i < N; i++) {
    if ((i % 2) == 0) {
      map[MakeUniq(i)] = MakeUniq(i + 100);
    } else {
      map.emplace(MakeUniq(i), MakeUniq(i + 100));
    }
  }
  EXPECT_EQ(map.size(), N);

  // move constructor
  UniqMap map2(std::move(map));

  // Lookups
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(*map2.at(MakeUniq(i)), i + 100);
  }

  // move assignment
  UniqMap map3;
  map3 = std::move(map2);

  // find+erase
  EXPECT_EQ(map3.count(MakeUniq(2)), 1);
  map3.erase(MakeUniq(2));
  EXPECT_EQ(map3.count(MakeUniq(2)), 0);

  // clear
  map3.clear();
  EXPECT_EQ(map3.size(), 0);

  // Check that moved-from maps are in a valid (though unspecified) state.
  EXPECT_GE(map.size(), 0);
  EXPECT_GE(map2.size(), 0);
  // This insert should succeed no matter what state `map` is in, because
  // MakeUniq(-1) is never called above: This key can't possibly exist.
  EXPECT_TRUE(map.emplace(MakeUniq(-1), MakeUniq(-1)).second);
}

TEST(FlatMap, UniqueMapIter) {
  UniqMap map;
  const int kCount = 10;
  const int kValueDelta = 100;
  for (int i = 1; i <= kCount; i++) {
    map[MakeUniq(i)] = MakeUniq(i + kValueDelta);
  }
  int key_sum = 0;
  int val_sum = 0;
  for (const auto& p : map) {
    key_sum += *p.first;
    val_sum += *p.second;
  }
  EXPECT_EQ(key_sum, (kCount * (kCount + 1)) / 2);
  EXPECT_EQ(val_sum, key_sum + (kCount * kValueDelta));
}

}  // namespace
}  // namespace gtl
}  // namespace tensorflow
