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
class MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/layout.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/layout_util.h"

namespace xla {

TileProto Tile::ToProto() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTcc mht_0(mht_0_v, 193, "", "./tensorflow/compiler/xla/layout.cc", "Tile::ToProto");

  TileProto tile_proto;
  for (int64_t i : dimensions()) {
    tile_proto.add_dimensions(i);
  }
  return tile_proto;
}

std::string Tile::ToString() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTcc mht_1(mht_1_v, 204, "", "./tensorflow/compiler/xla/layout.cc", "Tile::ToString");

  std::vector<std::string> elements;
  const auto& dims = dimensions();
  elements.reserve(dims.size());
  for (auto dim : dims) {
    if (dim >= 0) {
      elements.push_back(std::to_string(dim));
    } else {
      if (dim == kCombineDimension) {
        elements.push_back("*");
      } else {
        elements.push_back(absl::StrCat("Invalid value ", dim));
      }
    }
  }
  return absl::StrCat("(", absl::StrJoin(elements, ","), ")");
}

/* static */ Layout Layout::CreateFromProto(const LayoutProto& proto) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTcc mht_2(mht_2_v, 225, "", "./tensorflow/compiler/xla/layout.cc", "Layout::CreateFromProto");

  Layout layout;
  layout.set_format(proto.format());
  layout.minor_to_major_.reserve(proto.minor_to_major_size());
  for (const int64_t dimension : proto.minor_to_major()) {
    layout.add_minor_to_major(dimension);
  }
  for (const TileProto& tile_proto : proto.tiles()) {
    *layout.add_tiles() = Tile::CreateFromProto(tile_proto);
  }
  layout.set_element_size_in_bits(proto.element_size_in_bits());
  layout.set_memory_space(proto.memory_space());
  return layout;
}

LayoutProto Layout::ToProto() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTcc mht_3(mht_3_v, 243, "", "./tensorflow/compiler/xla/layout.cc", "Layout::ToProto");

  LayoutProto proto;
  proto.set_format(format_);
  proto.mutable_minor_to_major()->Reserve(minor_to_major_size());
  for (const int64_t dimension : minor_to_major()) {
    proto.add_minor_to_major(dimension);
  }
  for (const Tile& tile : tiles()) {
    *proto.add_tiles() = tile.ToProto();
  }
  proto.set_element_size_in_bits(element_size_in_bits());
  proto.set_memory_space(memory_space_);
  return proto;
}

std::string Layout::ToString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTcc mht_4(mht_4_v, 261, "", "./tensorflow/compiler/xla/layout.cc", "Layout::ToString");

  if (format() == DENSE) {
    std::string colon_string = tiles().empty() ? "" : "T";
    for (const Tile& tile : tiles()) {
      absl::StrAppend(&colon_string, tile.ToString());
    }
    if (element_size_in_bits() != 0) {
      absl::StrAppend(&colon_string, "E(", element_size_in_bits(), ")");
    }
    if (memory_space() != 0) {
      absl::StrAppend(&colon_string, "S(", memory_space(), ")");
    }
    return absl::StrCat("{", absl::StrJoin(minor_to_major(), ","),
                        colon_string.empty() ? "" : ":", colon_string, "}");
  } else {
    CHECK_EQ(format(), INVALID_FORMAT);
    return "invalid{}";
  }
}

bool Layout::Equal::operator()(const Layout& lhs, const Layout& rhs) {
  if (lhs.format() != rhs.format()) {
    return false;
  }
  if (lhs.format() == DENSE && lhs.minor_to_major() != rhs.minor_to_major()) {
    return false;
  }
  if (!ignore_tiles_ && lhs.tiles() != rhs.tiles()) {
    return false;
  }
  if (!ignore_element_size_ &&
      lhs.element_size_in_bits() != rhs.element_size_in_bits()) {
    return false;
  }
  if (!ignore_memory_space_ && lhs.memory_space() != rhs.memory_space()) {
    return false;
  }
  return true;
}

bool Layout::operator==(const Layout& other) const {
  return Equal()(*this, other);
}

std::ostream& operator<<(std::ostream& out, const Tile& tile) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTcc mht_5(mht_5_v, 308, "", "./tensorflow/compiler/xla/layout.cc", "operator<<");

  out << tile.ToString();
  return out;
}

std::ostream& operator<<(std::ostream& out, const Layout& layout) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSlayoutDTcc mht_6(mht_6_v, 316, "", "./tensorflow/compiler/xla/layout.cc", "operator<<");

  out << layout.ToString();
  return out;
}

}  // namespace xla
