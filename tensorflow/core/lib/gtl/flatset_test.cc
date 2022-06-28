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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatset_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatset_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatset_testDTcc() {
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

#include "tensorflow/core/lib/gtl/flatset.h"

#include <algorithm>
#include <string>
#include <vector>
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gtl {
namespace {

typedef FlatSet<int64_t> NumSet;

// Returns true iff set has an entry for k.
// Also verifies that find and count give consistent results.
bool Has(const NumSet& set, int64_t k) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatset_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/lib/gtl/flatset_test.cc", "Has");

  auto iter = set.find(k);
  if (iter == set.end()) {
    EXPECT_EQ(set.count(k), 0);
    return false;
  } else {
    EXPECT_EQ(set.count(k), 1);
    EXPECT_EQ(*iter, k);
    return true;
  }
}

// Return contents of set as a sorted list of numbers.
typedef std::vector<int64_t> NumSetContents;
NumSetContents Contents(const NumSet& set) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatset_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/lib/gtl/flatset_test.cc", "Contents");

  NumSetContents result(set.begin(), set.end());
  std::sort(result.begin(), result.end());
  return result;
}

// Fill entries with keys [start,limit).
void Fill(NumSet* set, int64_t start, int64_t limit) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatset_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/lib/gtl/flatset_test.cc", "Fill");

  for (int64_t i = start; i < limit; i++) {
    set->insert(i);
  }
}

TEST(FlatSetTest, Find) {
  NumSet set;
  EXPECT_FALSE(Has(set, 1));
  set.insert(1);
  set.insert(2);
  EXPECT_TRUE(Has(set, 1));
  EXPECT_TRUE(Has(set, 2));
  EXPECT_FALSE(Has(set, 3));
}

TEST(FlatSetTest, Insert) {
  NumSet set;
  EXPECT_FALSE(Has(set, 1));

  // New entry.
  auto result = set.insert(1);
  EXPECT_TRUE(result.second);
  EXPECT_EQ(*result.first, 1);
  EXPECT_TRUE(Has(set, 1));

  // Attempt to insert over existing entry.
  result = set.insert(1);
  EXPECT_FALSE(result.second);
  EXPECT_EQ(*result.first, 1);
  EXPECT_TRUE(Has(set, 1));
}

TEST(FlatSetTest, InsertGrowth) {
  NumSet set;
  const int n = 100;
  Fill(&set, 0, 100);
  EXPECT_EQ(set.size(), n);
  for (int i = 0; i < n; i++) {
    EXPECT_TRUE(Has(set, i)) << i;
  }
}

TEST(FlatSetTest, Emplace) {
  NumSet set;

  // New entry.
  auto result = set.emplace(73);
  EXPECT_TRUE(result.second);
  EXPECT_EQ(*result.first, 73);
  EXPECT_TRUE(Has(set, 73));

  // Attempt to insert an existing entry.
  result = set.emplace(73);
  EXPECT_FALSE(result.second);
  EXPECT_EQ(*result.first, 73);
  EXPECT_TRUE(Has(set, 73));

  // Add a second value
  result = set.emplace(103);
  EXPECT_TRUE(result.second);
  EXPECT_EQ(*result.first, 103);
  EXPECT_TRUE(Has(set, 103));
}

TEST(FlatSetTest, Size) {
  NumSet set;
  EXPECT_EQ(set.size(), 0);

  set.insert(1);
  set.insert(2);
  EXPECT_EQ(set.size(), 2);
}

TEST(FlatSetTest, Empty) {
  NumSet set;
  EXPECT_TRUE(set.empty());

  set.insert(1);
  set.insert(2);
  EXPECT_FALSE(set.empty());
}

TEST(FlatSetTest, Count) {
  NumSet set;
  EXPECT_EQ(set.count(1), 0);
  EXPECT_EQ(set.count(2), 0);

  set.insert(1);
  EXPECT_EQ(set.count(1), 1);
  EXPECT_EQ(set.count(2), 0);

  set.insert(2);
  EXPECT_EQ(set.count(1), 1);
  EXPECT_EQ(set.count(2), 1);
}

TEST(FlatSetTest, Iter) {
  NumSet set;
  EXPECT_EQ(Contents(set), NumSetContents());

  set.insert(1);
  set.insert(2);
  EXPECT_EQ(Contents(set), NumSetContents({1, 2}));
}

TEST(FlatSetTest, Erase) {
  NumSet set;
  EXPECT_EQ(set.erase(1), 0);
  set.insert(1);
  set.insert(2);
  EXPECT_EQ(set.erase(3), 0);
  EXPECT_EQ(set.erase(1), 1);
  EXPECT_EQ(set.size(), 1);
  EXPECT_TRUE(Has(set, 2));
  EXPECT_EQ(Contents(set), NumSetContents({2}));
  EXPECT_EQ(set.erase(2), 1);
  EXPECT_EQ(Contents(set), NumSetContents());
}

TEST(FlatSetTest, EraseIter) {
  NumSet set;
  Fill(&set, 1, 11);
  size_t size = 10;
  for (auto iter = set.begin(); iter != set.end();) {
    iter = set.erase(iter);
    size--;
    EXPECT_EQ(set.size(), size);
  }
  EXPECT_EQ(Contents(set), NumSetContents());
}

TEST(FlatSetTest, EraseIterPair) {
  NumSet set;
  Fill(&set, 1, 11);
  NumSet expected;
  auto p1 = set.begin();
  expected.insert(*p1);
  ++p1;
  expected.insert(*p1);
  ++p1;
  auto p2 = set.end();
  EXPECT_EQ(set.erase(p1, p2), set.end());
  EXPECT_EQ(set.size(), 2);
  EXPECT_EQ(Contents(set), Contents(expected));
}

TEST(FlatSetTest, EraseLongChains) {
  // Make a set with lots of elements and erase a bunch of them to ensure
  // that we are likely to hit them on future lookups.
  NumSet set;
  const int num = 128;
  Fill(&set, 0, num);
  for (int i = 0; i < num; i += 3) {
    EXPECT_EQ(set.erase(i), 1);
  }
  for (int i = 0; i < num; i++) {
    // Multiples of 3 should be not present.
    EXPECT_EQ(Has(set, i), ((i % 3) != 0)) << i;
  }

  // Erase remainder to trigger table shrinking.
  const size_t orig_buckets = set.bucket_count();
  for (int i = 0; i < num; i++) {
    set.erase(i);
  }
  EXPECT_TRUE(set.empty());
  EXPECT_EQ(set.bucket_count(), orig_buckets);
  set.insert(1);  // Actual shrinking is triggered by an insert.
  EXPECT_LT(set.bucket_count(), orig_buckets);
}

TEST(FlatSet, ClearNoResize) {
  NumSet set;
  Fill(&set, 0, 100);
  const size_t orig = set.bucket_count();
  set.clear_no_resize();
  EXPECT_EQ(set.size(), 0);
  EXPECT_EQ(Contents(set), NumSetContents());
  EXPECT_EQ(set.bucket_count(), orig);
}

TEST(FlatSet, Clear) {
  NumSet set;
  Fill(&set, 0, 100);
  const size_t orig = set.bucket_count();
  set.clear();
  EXPECT_EQ(set.size(), 0);
  EXPECT_EQ(Contents(set), NumSetContents());
  EXPECT_LT(set.bucket_count(), orig);
}

TEST(FlatSet, Copy) {
  for (int n = 0; n < 10; n++) {
    NumSet src;
    Fill(&src, 0, n);
    NumSet copy = src;
    EXPECT_EQ(Contents(src), Contents(copy));
    NumSet copy2;
    copy2 = src;
    EXPECT_EQ(Contents(src), Contents(copy2));
    copy2 = *&copy2;  // Self-assignment, avoiding -Wself-assign.
    EXPECT_EQ(Contents(src), Contents(copy2));
  }
}

TEST(FlatSet, InitFromIter) {
  for (int n = 0; n < 10; n++) {
    NumSet src;
    Fill(&src, 0, n);
    auto vec = Contents(src);
    NumSet dst(vec.begin(), vec.end());
    EXPECT_EQ(Contents(dst), vec);
  }
}

TEST(FlatSet, InitializerList) {
  NumSet a{1, 2, 3};
  NumSet b({1, 2, 3});
  NumSet c = {1, 2, 3};
  for (NumSet* set : std::vector<NumSet*>({&a, &b, &c})) {
    EXPECT_TRUE(Has(*set, 1));
    EXPECT_TRUE(Has(*set, 2));
    EXPECT_TRUE(Has(*set, 3));
    EXPECT_EQ(Contents(*set), NumSetContents({1, 2, 3}));
  }
}

TEST(FlatSet, InsertIter) {
  NumSet a, b;
  Fill(&a, 1, 10);
  Fill(&b, 8, 20);
  b.insert(9);  // Should not get inserted into a since a already has 9
  a.insert(b.begin(), b.end());
  NumSet expected;
  Fill(&expected, 1, 20);
  EXPECT_EQ(Contents(a), Contents(expected));
}

TEST(FlatSet, Eq) {
  NumSet empty;

  NumSet elems;
  Fill(&elems, 0, 5);
  EXPECT_FALSE(empty == elems);
  EXPECT_TRUE(empty != elems);

  NumSet copy = elems;
  EXPECT_TRUE(copy == elems);
  EXPECT_FALSE(copy != elems);

  NumSet changed = elems;
  changed.insert(7);
  EXPECT_FALSE(changed == elems);
  EXPECT_TRUE(changed != elems);

  NumSet changed2 = elems;
  changed2.erase(3);
  EXPECT_FALSE(changed2 == elems);
  EXPECT_TRUE(changed2 != elems);
}

TEST(FlatSet, Swap) {
  NumSet a, b;
  Fill(&a, 1, 5);
  Fill(&b, 100, 200);
  NumSet c = a;
  NumSet d = b;
  EXPECT_EQ(c, a);
  EXPECT_EQ(d, b);
  c.swap(d);
  EXPECT_EQ(c, b);
  EXPECT_EQ(d, a);
}

TEST(FlatSet, Reserve) {
  NumSet src;
  Fill(&src, 1, 100);
  NumSet a = src;
  a.reserve(10);
  EXPECT_EQ(a, src);
  NumSet b = src;
  b.rehash(1000);
  EXPECT_EQ(b, src);
}

TEST(FlatSet, EqualRangeMutable) {
  NumSet set;
  Fill(&set, 1, 10);

  // Existing element
  auto p1 = set.equal_range(3);
  EXPECT_TRUE(p1.first != p1.second);
  EXPECT_EQ(*p1.first, 3);
  ++p1.first;
  EXPECT_TRUE(p1.first == p1.second);

  // Missing element
  auto p2 = set.equal_range(100);
  EXPECT_TRUE(p2.first == p2.second);
}

TEST(FlatSet, EqualRangeConst) {
  NumSet tmp;
  Fill(&tmp, 1, 10);

  const NumSet set = tmp;

  // Existing element
  auto p1 = set.equal_range(3);
  EXPECT_TRUE(p1.first != p1.second);
  EXPECT_EQ(*p1.first, 3);
  ++p1.first;
  EXPECT_TRUE(p1.first == p1.second);

  // Missing element
  auto p2 = set.equal_range(100);
  EXPECT_TRUE(p2.first == p2.second);
}

TEST(FlatSet, Prefetch) {
  NumSet set;
  Fill(&set, 0, 1000);
  // Prefetch present and missing keys.
  for (int i = 0; i < 2000; i++) {
    set.prefetch_value(i);
  }
}

// Non-assignable values should work.
struct NA {
  int64_t value;
  NA() : value(-1) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatset_testDTcc mht_3(mht_3_v, 564, "", "./tensorflow/core/lib/gtl/flatset_test.cc", "NA");
}
  explicit NA(int64_t v) : value(v) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatset_testDTcc mht_4(mht_4_v, 568, "", "./tensorflow/core/lib/gtl/flatset_test.cc", "NA");
}
  NA(const NA& x) : value(x.value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatset_testDTcc mht_5(mht_5_v, 572, "", "./tensorflow/core/lib/gtl/flatset_test.cc", "NA");
}
  bool operator==(const NA& x) const { return value == x.value; }
};
struct HashNA {
  size_t operator()(NA x) const { return x.value; }
};

TEST(FlatSet, NonAssignable) {
  FlatSet<NA, HashNA> set;
  for (int i = 0; i < 100; i++) {
    set.insert(NA(i));
  }
  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(set.count(NA(i)), 1);
    auto iter = set.find(NA(i));
    EXPECT_NE(iter, set.end());
    EXPECT_EQ(*iter, NA(i));
  }
  set.erase(NA(10));
  EXPECT_EQ(set.count(NA(10)), 0);
}

TEST(FlatSet, ForwardIterator) {
  // Test the requirements of forward iterators
  typedef FlatSet<NA, HashNA> NASet;
  NASet set({NA(1), NA(2)});
  NASet::iterator it1 = set.find(NA(1));
  NASet::iterator it2 = set.find(NA(2));

  // Test operator != and ==
  EXPECT_TRUE(it1 != set.end());
  EXPECT_TRUE(it2 != set.end());
  EXPECT_FALSE(it1 == set.end());
  EXPECT_FALSE(it2 == set.end());
  EXPECT_TRUE(it1 != it2);
  EXPECT_FALSE(it1 == it2);

  // Test operator * and ->
  EXPECT_EQ(*it1, NA(1));
  EXPECT_EQ(*it2, NA(2));
  EXPECT_EQ(it1->value, 1);
  EXPECT_EQ(it2->value, 2);

  // Test prefix ++
  NASet::iterator copy_it1 = it1;
  NASet::iterator copy_it2 = it2;
  EXPECT_EQ(*copy_it1, NA(1));
  EXPECT_EQ(*copy_it2, NA(2));
  NASet::iterator& pp_copy_it1 = ++copy_it1;
  NASet::iterator& pp_copy_it2 = ++copy_it2;
  EXPECT_TRUE(pp_copy_it1 == copy_it1);
  EXPECT_TRUE(pp_copy_it2 == copy_it2);
  // Check either possible ordering of the two items
  EXPECT_TRUE(copy_it1 != it1);
  EXPECT_TRUE(copy_it2 != it2);
  if (copy_it1 == set.end()) {
    EXPECT_TRUE(copy_it2 != set.end());
    EXPECT_EQ(*copy_it2, NA(1));
    EXPECT_EQ(*pp_copy_it2, NA(1));
  } else {
    EXPECT_TRUE(copy_it2 == set.end());
    EXPECT_EQ(*copy_it1, NA(2));
    EXPECT_EQ(*pp_copy_it1, NA(2));
  }
  // Ensure it{1,2} haven't moved
  EXPECT_EQ(*it1, NA(1));
  EXPECT_EQ(*it2, NA(2));

  // Test postfix ++
  copy_it1 = it1;
  copy_it2 = it2;
  EXPECT_EQ(*copy_it1, NA(1));
  EXPECT_EQ(*copy_it2, NA(2));
  NASet::iterator copy_it1_pp = copy_it1++;
  NASet::iterator copy_it2_pp = copy_it2++;
  EXPECT_TRUE(copy_it1_pp != copy_it1);
  EXPECT_TRUE(copy_it2_pp != copy_it2);
  EXPECT_TRUE(copy_it1_pp == it1);
  EXPECT_TRUE(copy_it2_pp == it2);
  EXPECT_EQ(*copy_it1_pp, NA(1));
  EXPECT_EQ(*copy_it2_pp, NA(2));
  // Check either possible ordering of the two items
  EXPECT_TRUE(copy_it1 != it1);
  EXPECT_TRUE(copy_it2 != it2);
  if (copy_it1 == set.end()) {
    EXPECT_TRUE(copy_it2 != set.end());
    EXPECT_EQ(*copy_it2, NA(1));
  } else {
    EXPECT_TRUE(copy_it2 == set.end());
    EXPECT_EQ(*copy_it1, NA(2));
  }
  // Ensure it{1,2} haven't moved
  EXPECT_EQ(*it1, NA(1));
  EXPECT_EQ(*it2, NA(2));
}

// Test with heap-allocated objects so that mismanaged constructions
// or destructions will show up as errors under a sanitizer or
// heap checker.
TEST(FlatSet, ConstructDestruct) {
  FlatSet<string> set;
  string k1 = "the quick brown fox jumped over the lazy dog";
  string k2 = k1 + k1;
  string k3 = k1 + k2;
  set.insert(k1);
  set.insert(k3);
  EXPECT_EQ(set.count(k1), 1);
  EXPECT_EQ(set.count(k2), 0);
  EXPECT_EQ(set.count(k3), 1);

  set.erase(k3);
  EXPECT_EQ(set.count(k3), 0);

  set.clear();
  set.insert(k1);
  EXPECT_EQ(set.count(k1), 1);
  EXPECT_EQ(set.count(k3), 0);

  set.reserve(100);
  EXPECT_EQ(set.count(k1), 1);
  EXPECT_EQ(set.count(k3), 0);
}

// Type to use to ensure that custom equality operator is used
// that ignores extra value.
struct CustomCmpKey {
  int64_t a;
  int64_t b;
  CustomCmpKey(int64_t v1, int64_t v2) : a(v1), b(v2) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatset_testDTcc mht_6(mht_6_v, 703, "", "./tensorflow/core/lib/gtl/flatset_test.cc", "CustomCmpKey");
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
TEST(FlatSet, CustomCmp) {
  FlatSet<CustomCmpKey, HashA, EqA> set;
  set.insert(CustomCmpKey(100, 200));
  EXPECT_EQ(set.count(CustomCmpKey(100, 200)), 1);
  EXPECT_EQ(set.count(CustomCmpKey(100, 500)), 1);  // key.b ignored
}

// Test unique_ptr handling.
typedef std::unique_ptr<int> UniqInt;
static UniqInt MakeUniq(int i) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSflatset_testDTcc mht_7(mht_7_v, 725, "", "./tensorflow/core/lib/gtl/flatset_test.cc", "MakeUniq");
 return UniqInt(new int(i)); }

struct HashUniq {
  size_t operator()(const UniqInt& p) const { return *p; }
};
struct EqUniq {
  bool operator()(const UniqInt& a, const UniqInt& b) const { return *a == *b; }
};
typedef FlatSet<UniqInt, HashUniq, EqUniq> UniqSet;

TEST(FlatSet, UniqueSet) {
  UniqSet set;

  // Fill set
  const int N = 10;
  for (int i = 0; i < N; i++) {
    set.emplace(MakeUniq(i));
  }
  EXPECT_EQ(set.size(), N);

  // Move constructor
  UniqSet set2(std::move(set));

  // Lookups
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(set2.count(MakeUniq(i)), 1);
  }

  // Move-assignment operator
  UniqSet set3;
  set3 = std::move(set2);

  // erase
  set3.erase(MakeUniq(2));
  EXPECT_EQ(set3.count(MakeUniq(2)), 0);

  // clear
  set.clear();
  EXPECT_EQ(set.size(), 0);

  // Check that moved-from sets are in a valid (though unspecified) state.
  EXPECT_GE(set.size(), 0);
  EXPECT_GE(set2.size(), 0);
  // This insert should succeed no matter what state `set` is in, because
  // MakeUniq(-1) is never called above: This key can't possibly exist.
  EXPECT_TRUE(set.emplace(MakeUniq(-1)).second);
}

TEST(FlatSet, UniqueSetIter) {
  UniqSet set;
  const int kCount = 10;
  for (int i = 1; i <= kCount; i++) {
    set.emplace(MakeUniq(i));
  }
  int sum = 0;
  for (const auto& p : set) {
    sum += *p;
  }
  EXPECT_EQ(sum, (kCount * (kCount + 1)) / 2);
}

TEST(FlatSet, InsertUncopyable) {
  UniqSet set;
  EXPECT_TRUE(set.insert(MakeUniq(0)).second);
  EXPECT_EQ(set.size(), 1);
}

/* This would be a good negative compilation test, if we could do that.

TEST(FlatSet, MutableIterator_ShouldNotCompile) {
  NumSet set;
  set.insert(5);
  EXPECT_TRUE(Has(set, 5));
  EXPECT_EQ(Contents(set), NumSetContents({5}));

  // Here's where things go bad.  We shouldn't be allowed to mutate the set key
  // directly, since there's no way the update the underlying hashtable after
  // the mutation, regardless of how we implemented it.
  //
  // This doesn't compile, since iterator is an alias of const_iterator.
  *set.begin() = 6;

  // If it does compile, this should expose a failure.
  EXPECT_TRUE(Has(set, 6));
  EXPECT_EQ(Contents(set), NumSetContents({6}));
}
*/

}  // namespace
}  // namespace gtl
}  // namespace tensorflow
