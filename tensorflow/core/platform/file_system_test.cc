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
class MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc() {
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

#include "tensorflow/core/platform/file_system.h"

#include <sys/stat.h>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/null_file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

static const char* const kPrefix = "ipfs://solarsystem";

// A file system that has Planets, Satellites and Sub Satellites. Sub satellites
// cannot have children further.
class InterPlanetaryFileSystem : public NullFileSystem {
 public:
  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  Status FileExists(const string& fname, TransactionToken* token) override {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/platform/file_system_test.cc", "FileExists");

    string parsed_path;
    ParsePath(fname, &parsed_path);
    if (BodyExists(parsed_path)) {
      return Status::OK();
    }
    return Status(tensorflow::error::NOT_FOUND, "File does not exist");
  }

  // Adds the dir to the parent's children list and creates an entry for itself.
  Status CreateDir(const string& dirname, TransactionToken* token) override {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/platform/file_system_test.cc", "CreateDir");

    string parsed_path;
    ParsePath(dirname, &parsed_path);
    // If the directory already exists, throw an error.
    if (celestial_bodies_.find(parsed_path) != celestial_bodies_.end()) {
      return Status(tensorflow::error::ALREADY_EXISTS,
                    "dirname already exists.");
    }
    std::vector<string> split_path = str_util::Split(parsed_path, '/');
    // If the path is too long then we don't support it.
    if (split_path.size() > 3) {
      return Status(tensorflow::error::INVALID_ARGUMENT, "Bad dirname");
    }
    if (split_path.empty()) {
      return Status::OK();
    }
    if (split_path.size() == 1) {
      celestial_bodies_[""].insert(parsed_path);
      celestial_bodies_.insert(
          std::pair<string, std::set<string>>(parsed_path, {}));
      return Status::OK();
    }
    if (split_path.size() == 2) {
      if (!BodyExists(split_path[0])) {
        return Status(tensorflow::error::FAILED_PRECONDITION,
                      "Base dir not created");
      }
      celestial_bodies_[split_path[0]].insert(split_path[1]);
      celestial_bodies_.insert(
          std::pair<string, std::set<string>>(parsed_path, {}));
      return Status::OK();
    }
    if (split_path.size() == 3) {
      const string& parent_path = this->JoinPath(split_path[0], split_path[1]);
      if (!BodyExists(parent_path)) {
        return Status(tensorflow::error::FAILED_PRECONDITION,
                      "Base dir not created");
      }
      celestial_bodies_[parent_path].insert(split_path[2]);
      celestial_bodies_.insert(
          std::pair<string, std::set<string>>(parsed_path, {}));
      return Status::OK();
    }
    return Status(tensorflow::error::FAILED_PRECONDITION, "Failed to create");
  }

  Status IsDirectory(const string& dirname, TransactionToken* token) override {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc mht_2(mht_2_v, 271, "", "./tensorflow/core/platform/file_system_test.cc", "IsDirectory");

    string parsed_path;
    ParsePath(dirname, &parsed_path);
    // Simulate evil_directory has bad permissions by throwing a LOG(FATAL)
    if (parsed_path == "evil_directory") {
      LOG(FATAL) << "evil_directory cannot be accessed";
    }
    std::vector<string> split_path = str_util::Split(parsed_path, '/');
    if (split_path.size() > 2) {
      return Status(tensorflow::error::FAILED_PRECONDITION, "Not a dir");
    }
    if (celestial_bodies_.find(parsed_path) != celestial_bodies_.end()) {
      return Status::OK();
    }
    return Status(tensorflow::error::FAILED_PRECONDITION, "Not a dir");
  }

  Status GetChildren(const string& dir, TransactionToken* token,
                     std::vector<string>* result) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc mht_3(mht_3_v, 293, "", "./tensorflow/core/platform/file_system_test.cc", "GetChildren");

    TF_RETURN_IF_ERROR(IsDirectory(dir, nullptr));
    string parsed_path;
    ParsePath(dir, &parsed_path);
    result->insert(result->begin(), celestial_bodies_[parsed_path].begin(),
                   celestial_bodies_[parsed_path].end());
    return Status::OK();
  }

 private:
  bool BodyExists(const string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc mht_4(mht_4_v, 307, "", "./tensorflow/core/platform/file_system_test.cc", "BodyExists");

    return celestial_bodies_.find(name) != celestial_bodies_.end();
  }

  void ParsePath(const string& name, string* parsed_path) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc mht_5(mht_5_v, 315, "", "./tensorflow/core/platform/file_system_test.cc", "ParsePath");

    StringPiece scheme, host, path;
    this->ParseURI(name, &scheme, &host, &path);
    ASSERT_EQ(scheme, "ipfs");
    ASSERT_EQ(host, "solarsystem");
    absl::ConsumePrefix(&path, "/");
    *parsed_path = string(path);
  }

  std::map<string, std::set<string>> celestial_bodies_ = {
      std::pair<string, std::set<string>>(
          "", {"Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
               "Uranus", "Neptune"}),
      std::pair<string, std::set<string>>("Mercury", {}),
      std::pair<string, std::set<string>>("Venus", {}),
      std::pair<string, std::set<string>>("Earth", {"Moon"}),
      std::pair<string, std::set<string>>("Mars", {}),
      std::pair<string, std::set<string>>("Jupiter",
                                          {"Europa", "Io", "Ganymede"}),
      std::pair<string, std::set<string>>("Saturn", {}),
      std::pair<string, std::set<string>>("Uranus", {}),
      std::pair<string, std::set<string>>("Neptune", {}),
      std::pair<string, std::set<string>>("Earth/Moon", {}),
      std::pair<string, std::set<string>>("Jupiter/Europa", {}),
      std::pair<string, std::set<string>>("Jupiter/Io", {}),
      std::pair<string, std::set<string>>("Jupiter/Ganymede", {})};
};

// Returns all the matched entries as a comma separated string removing the
// common prefix of BaseDir().
string Match(InterPlanetaryFileSystem* ipfs, const string& suffix_pattern) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("suffix_pattern: \"" + suffix_pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc mht_6(mht_6_v, 349, "", "./tensorflow/core/platform/file_system_test.cc", "Match");

  std::vector<string> results;
  Status s = ipfs->GetMatchingPaths(ipfs->JoinPath(kPrefix, suffix_pattern),
                                    nullptr, &results);
  if (!s.ok()) {
    return s.ToString();
  } else {
    std::vector<StringPiece> trimmed_results;
    std::sort(results.begin(), results.end());
    for (const string& result : results) {
      StringPiece trimmed_result(result);
      EXPECT_TRUE(
          absl::ConsumePrefix(&trimmed_result, strings::StrCat(kPrefix, "/")));
      trimmed_results.push_back(trimmed_result);
    }
    return absl::StrJoin(trimmed_results, ",");
  }
}

TEST(InterPlanetaryFileSystemTest, IPFSMatch) {
  InterPlanetaryFileSystem ipfs;
  EXPECT_EQ(Match(&ipfs, "thereisnosuchfile"), "");
  EXPECT_EQ(Match(&ipfs, "*"),
            "Earth,Jupiter,Mars,Mercury,Neptune,Saturn,Uranus,Venus");
  // Returns Jupiter's moons.
  EXPECT_EQ(Match(&ipfs, "Jupiter/*"),
            "Jupiter/Europa,Jupiter/Ganymede,Jupiter/Io");
  // Returns Jupiter's and Earth's moons.
  EXPECT_EQ(Match(&ipfs, "*/*"),
            "Earth/Moon,Jupiter/Europa,Jupiter/Ganymede,Jupiter/Io");
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "Planet0"), nullptr));
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "Planet1"), nullptr));
  EXPECT_EQ(Match(&ipfs, "Planet[0-1]"), "Planet0,Planet1");
  EXPECT_EQ(Match(&ipfs, "Planet?"), "Planet0,Planet1");
}

TEST(InterPlanetaryFileSystemTest, MatchSimple) {
  InterPlanetaryFileSystem ipfs;
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "match-00"), nullptr));
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "match-0a"), nullptr));
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "match-01"), nullptr));
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "match-aaa"), nullptr));

  EXPECT_EQ(Match(&ipfs, "match-*"), "match-00,match-01,match-0a,match-aaa");
  EXPECT_EQ(Match(&ipfs, "match-0[0-9]"), "match-00,match-01");
  EXPECT_EQ(Match(&ipfs, "match-?[0-9]"), "match-00,match-01");
  EXPECT_EQ(Match(&ipfs, "match-?a*"), "match-0a,match-aaa");
  EXPECT_EQ(Match(&ipfs, "match-??"), "match-00,match-01,match-0a");
}

// Create 2 directories abcd and evil_directory. Look for abcd and make sure
// that evil_directory isn't accessed.
TEST(InterPlanetaryFileSystemTest, MatchOnlyNeeded) {
  InterPlanetaryFileSystem ipfs;
  TF_EXPECT_OK(ipfs.CreateDir(ipfs.JoinPath(kPrefix, "abcd"), nullptr));
  TF_EXPECT_OK(
      ipfs.CreateDir(ipfs.JoinPath(kPrefix, "evil_directory"), nullptr));

  EXPECT_EQ(Match(&ipfs, "abcd"), "abcd");
}

TEST(InterPlanetaryFileSystemTest, MatchDirectory) {
  InterPlanetaryFileSystem ipfs;
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(
      ipfs.JoinPath(kPrefix, "match-00/abc/x"), nullptr));
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(
      ipfs.JoinPath(kPrefix, "match-0a/abc/x"), nullptr));
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(
      ipfs.JoinPath(kPrefix, "match-01/abc/x"), nullptr));
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(
      ipfs.JoinPath(kPrefix, "match-aaa/abc/x"), nullptr));

  EXPECT_EQ(Match(&ipfs, "match-*/abc/x"),
            "match-00/abc/x,match-01/abc/x,match-0a/abc/x,match-aaa/abc/x");
  EXPECT_EQ(Match(&ipfs, "match-0[0-9]/abc/x"),
            "match-00/abc/x,match-01/abc/x");
  EXPECT_EQ(Match(&ipfs, "match-?[0-9]/abc/x"),
            "match-00/abc/x,match-01/abc/x");
  EXPECT_EQ(Match(&ipfs, "match-?a*/abc/x"), "match-0a/abc/x,match-aaa/abc/x");
  EXPECT_EQ(Match(&ipfs, "match-?[^a]/abc/x"), "match-00/abc/x,match-01/abc/x");
}

TEST(InterPlanetaryFileSystemTest, MatchMultipleWildcards) {
  InterPlanetaryFileSystem ipfs;
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(
      ipfs.JoinPath(kPrefix, "match-00/abc/00"), nullptr));
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(
      ipfs.JoinPath(kPrefix, "match-00/abc/01"), nullptr));
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(
      ipfs.JoinPath(kPrefix, "match-00/abc/09"), nullptr));
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(
      ipfs.JoinPath(kPrefix, "match-01/abc/00"), nullptr));
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(
      ipfs.JoinPath(kPrefix, "match-01/abc/04"), nullptr));
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(
      ipfs.JoinPath(kPrefix, "match-01/abc/10"), nullptr));
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(
      ipfs.JoinPath(kPrefix, "match-02/abc/00"), nullptr));

  EXPECT_EQ(Match(&ipfs, "match-0[0-1]/abc/0[0-8]"),
            "match-00/abc/00,match-00/abc/01,match-01/abc/00,match-01/abc/04");
}

TEST(InterPlanetaryFileSystemTest, RecursivelyCreateAlreadyExistingDir) {
  InterPlanetaryFileSystem ipfs;
  const string dirname = ipfs.JoinPath(kPrefix, "match-00/abc/00");
  TF_EXPECT_OK(ipfs.RecursivelyCreateDir(dirname));
  // We no longer check for recursively creating the directory again because
  // `ipfs.IsDirectory` is badly implemented, fixing it will break other tests
  // in this suite and we already test creating the directory again in
  // env_test.cc as well as in the modular filesystem tests.
}

TEST(InterPlanetaryFileSystemTest, HasAtomicMove) {
  InterPlanetaryFileSystem ipfs;
  const string dirname = io::JoinPath(kPrefix, "match-00/abc/00");
  bool has_atomic_move;
  TF_EXPECT_OK(ipfs.HasAtomicMove(dirname, &has_atomic_move));
  EXPECT_EQ(has_atomic_move, true);
}

// A simple file system with a root directory and a single file underneath it.
class TestFileSystem : public NullFileSystem {
 public:
  // Only allow for a single root directory.
  Status IsDirectory(const string& dirname, TransactionToken* token) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc mht_7(mht_7_v, 478, "", "./tensorflow/core/platform/file_system_test.cc", "IsDirectory");

    if (dirname == "." || dirname.empty()) {
      return Status::OK();
    }
    return Status(tensorflow::error::FAILED_PRECONDITION, "Not a dir");
  }

  // Simulating a FS with a root dir and a single file underneath it.
  Status GetChildren(const string& dir, TransactionToken* token,
                     std::vector<string>* result) override {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_testDTcc mht_8(mht_8_v, 491, "", "./tensorflow/core/platform/file_system_test.cc", "GetChildren");

    if (dir == "." || dir.empty()) {
      result->push_back("test");
    }
    return Status::OK();
  }
};

// Making sure that ./<pattern> and <pattern> have the same result.
TEST(TestFileSystemTest, RootDirectory) {
  TestFileSystem fs;
  std::vector<string> results;
  auto ret = fs.GetMatchingPaths("./te*", nullptr, &results);
  EXPECT_EQ(1, results.size());
  EXPECT_EQ("./test", results[0]);
  ret = fs.GetMatchingPaths("te*", nullptr, &results);
  EXPECT_EQ(1, results.size());
  EXPECT_EQ("./test", results[0]);
}

}  // namespace tensorflow
