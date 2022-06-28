/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COLOCATION_GRAPH_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COLOCATION_GRAPH_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePScolocation_graphDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePScolocation_graphDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePScolocation_graphDTh() {
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


#include <unordered_map>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/inspecting_placer.h"
#include "tensorflow/core/common_runtime/placer_inspection_required_ops_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

// Represents a node in the disjoint node forest and the
// accumulated constraints on the device used by that node.
class Member {
 public:
  Member() = default;

  Status SetParentAndSupportedDevices(
      const Node& node, const std::vector<DeviceType>& types,
      const DeviceNameUtils::ParsedName* local_address_spec);

  const DeviceNameUtils::ParsedName& requested_device_name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScolocation_graphDTh mht_0(mht_0_v, 213, "", "./tensorflow/core/common_runtime/colocation_graph.h", "requested_device_name");

    return requested_device_name_;
  }

  Status SetAssignedDeviceName(const string& device_name);
  Status SetResourceDeviceName(const Node& node);
  Status SetRequestedDeviceName(const Node& node);

  Status FillPossibleDevices(PossibleDevices* possible_device) const;

  // Returns whether `src_root` is assigned to a CompositeDevice and `this` is
  // assigned to a physical device.
  bool IsEdgeFromCompositeDeviceToPhysicalDevice(const Member& src_root) const;

  Status EnsureCompatibilityAcrossResourceEdge(
      const Node& src, const Member& src_root,
      const Node& dst, /*dst_root is this*/
      bool log_device_placement);

  const PrioritizedDeviceTypeVector& supported_device_types() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScolocation_graphDTh mht_1(mht_1_v, 235, "", "./tensorflow/core/common_runtime/colocation_graph.h", "supported_device_types");

    return supported_device_types_;
  }

  // If `dry_run` is true, just sets `new_root` and `old_root` and does not
  // actually modify anything in the `tree`.
  static void Merge(std::vector<Member>* tree, int x_root, int y_root,
                    Member** new_root, Member** old_root, bool dry_run);

  // Returns the root node of the disjoint tree to which the node with the
  // given id is connected.
  // FindRoot should be called only for debugging or after the members have
  // been updated with direct root pointers because it does not update
  // root pointers and can traverse many links. It exists to have
  // a const version of FindAndUpdateRoot
  static int FindRoot(const std::vector<Member>& tree, int node_id);
  static int FindAndUpdateRoot(std::vector<Member>* tree, int node_id);

  Status MergeDeviceNames(const Member& other, bool allow_soft_placement);

  // Updates this to contain the intersection of the device types in
  // this and "other". If the intersection is empty, returns false and does
  // not update this. Else returns true and updates this.
  bool MergeSupportedDevices(const Member& other);

  Status AssignDevice(const Node& node);

  // If user does not explicitly request XLA device and non-XLA device is
  // supported for this node, use only the non-XLA device. See b/140896502.
  void MaybeExcludeXlaDevices();

  // Limit the possible devices of this (should be a root) to the device
  // specifications in `devices`.
  Status LimitToPossibleDevices(const PossibleDevices& devices,
                                bool allow_soft_placement);

  void set_possible_devices(std::vector<Device*>&& devices) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScolocation_graphDTh mht_2(mht_2_v, 274, "", "./tensorflow/core/common_runtime/colocation_graph.h", "set_possible_devices");

    possible_devices_ = devices;
  }
  const std::vector<Device*>& possible_devices() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScolocation_graphDTh mht_3(mht_3_v, 280, "", "./tensorflow/core/common_runtime/colocation_graph.h", "possible_devices");
 return possible_devices_; }

  // Returns a (parsed) device name that is based on requested_device_name()
  // but with potentially cleared device type and ID fields. A field is cleared
  // if the assigned_device_name does not specify it. If it does, the field
  // is not cleared because soft placement cannot violate assigned device names.
  DeviceNameUtils::ParsedName GetSoftDeviceName() const;

  // Same as GetSoftDeviceName but device type and device ID fields are not
  // cleared if resource device has them set.
  DeviceNameUtils::ParsedName GetPreferredSoftDeviceName() const;

  string DebugString() const;

 private:
  // Updates this to contain the intersection of the device types in
  // this and `other_devices`.
  bool MergeSupportedDevices(const PrioritizedDeviceTypeVector& other_devices);

  // The id of the node that is the parent of this one, or its own
  // id if it is a root. parent <= 0 indicates that this member is invalid.
  int parent_ = -1;

  // A proxy for the depth of the tree that is used to prefer
  // connecting smaller trees to larger trees when merging disjoint
  // sets.
  int rank_ = 0;

  // Once colocation groups have been formed, the Placer starts actually
  // choosing devices. All nodes in a group must be assigned to the same
  // device. Once we assigned the first device to some node in this group,
  // we set assigned_device_name_index to this device name's index in the
  // graph.
  // The `*_device_name_` fields will contain the parsed name of this device
  // and `possible_devices`, if computed, will contain just this device.
  // `assigned_device_name_index` is an optimization to avoid parsing and
  // comparing device names. The value of -1 signals that a single device
  // has not been chosen yet.
  int assigned_device_name_index_ = -1;

  // The merged form of the device requested for this node, with those of all of
  // its children. requested_device_name_ is always kept a specialization (i.e.
  // DeviceNameUtils::IsSpecification) of assigned_device_name_. When no device
  // is requested, this field is set to assigned_device_name_.  As a
  // specialization of assigned_device_name_, requested_device_name_ represents
  // the most specific form of all assigned and requested devices of this node
  // and its children, if this node is a root. requested_device_name_ is used
  // to finally select devices for nodes.  We can override requested devices due
  // to resource colocation constraints but not assigned devices (unless soft
  // placement is on).
  // INVARIANT: requested_device_name_ is always kept a
  // DeviceNameUtils::IsSpecification of assigned_device_name_ and
  // resource_device_name_. This makes requested_device_name_ the "accumulation
  // of all wishes" about the device.
  DeviceNameUtils::ParsedName requested_device_name_;

  // The merged form of the device assigned for this node, with
  // those of all of its children.
  // This field is used to raise errors due to unsatisfiable constraints.
  // Can be a partial specification.
  DeviceNameUtils::ParsedName assigned_device_name_;

  // The merged form of the requested resource device assigned for this node,
  // with those of all of its children.
  // This field is used to raise errors due to unsatisfiable constraints.
  // Can be a partial specification.
  // resource_device_name_ is initialized with user-requested device on nodes
  // producing resources, e.g. VarHandleOp.
  // For historical reasons, with soft placement enabled, Placer can "move"
  // resources (place resource producing ops on a device different from what
  // the user explicitly requested) when the colocation group of a resource
  // producing op contains ops that are not supported on the user-requested
  // resource device. A classic example of this is a sparse optimizer (only
  // supported on CPU) used on a GPU variable. In this case, the whole group
  // will be assigned to some device supported by all ops in the colocation
  // group. This is a surprising and unfortunate behavior because:
  //   1. Since soft_placement is on by default, users don't know that their
  //   variables are created on a different device than what they requested.
  //   Among other things, this can lead to surprising poor performance.
  //   2. Eager runtime cannot "move" resources. The same code can "work" when
  //   wrapped in tf.function but will fail when run eagerly.
  //   3. Extra complexity here to preserve these resource moving capabilities.
  DeviceNameUtils::ParsedName resource_device_name_;

  // The intersection of all device types supported by this node,
  // and those of all of its children, in priority order
  // of the preferred device.
  // It is possible that supported_device_types_ has an empty intersection with
  // requested/assigned/resource devices. We could have detected such cases
  // as soon as they happen and raise an error. Instead, for historical reasons,
  // we leave such error detection to the final device picking stage.
  PrioritizedDeviceTypeVector supported_device_types_;

  // If this node is a root, stores a list of Devices to which this node
  // and all of its children can be assigned.
  // `possible_devices` is empty if they have not yet been computed.
  std::vector<Device*> possible_devices_;
};

// This class maintains the connected components of a colocation
// constraint graph, and uses this information to assign a satisfying
// device placement to the nodes of the graph.
//
// This implementation uses the Union-Find algorithm to efficiently maintain the
// connected components and incrementally adds edges via
// ColocationGraph::ColocateNodes() invocations.
//
// ColocationGraph does not assign any devices to graph nodes. The
// `log_device_placement` argument is used to log messages when requested
// device is ignored.
class ColocationGraph {
 public:
  // graph, flib_def, and device_set must not be null and must outlive
  // this ColocationGraph. default_local_device can be null. If not, must
  // outlive this.
  ColocationGraph(const Graph* graph, const FunctionStack& stack,
                  const FunctionLibraryDefinition* flib_def,
                  const DeviceSet* device_set,
                  const Device* default_local_device, bool allow_soft_placement,
                  bool log_device_placement);

  Status Initialize();

  const std::vector<Member>& members() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScolocation_graphDTh mht_4(mht_4_v, 406, "", "./tensorflow/core/common_runtime/colocation_graph.h", "members");
 return members_; }

  // Limit the group containing `node` to the device specifications in
  // `devices`.
  Status LimitToPossibleDevices(const Node& node,
                                const PossibleDevices& devices);

  // Limits the possible devices of `node`'s colocation group to the device
  // to which `node` is assigned. This makes sure that all nodes in this
  // colocation group will be assigned to the same device. Without this
  // explicit restriction, heuristics can choose a different possible device
  // for other nodes in the group.
  Status LimitToAssignedDevice(const Node& node);

  // Returns the root node of the disjoint tree to which the node with the
  // given id is connected.
  // Updates the internal pointers so that future calls will returns faster.
  int FindAndUpdateRoot(int node_id) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScolocation_graphDTh mht_5(mht_5_v, 426, "", "./tensorflow/core/common_runtime/colocation_graph.h", "FindAndUpdateRoot");

    return Member::FindAndUpdateRoot(&members_, node_id);
  }

  // For the given node, subject to the constraints previously given
  // to this ColocationGraph, set its assigned_device_name. Returns OK
  // if a satisfying device can be found, otherwise an error.
  //
  // Note: This method returns a pointer to a field within members_.
  // The caller must not use the returned pointer after there is any possibility
  // that the members_[i].possible_devices field has been modified.
  Status GetDevicesForNode(Node* node,
                           const std::vector<Device*>** possible_devices);

  // Returns debugging info for the node referred to by 'node_root'.
  string DebugInfo(const int node_root) const;

  string DebugString() const;

  // Returns a list of devices having type in supported_device_types.  The
  // returned list is sorted by preferred type (higher numeric type is
  // preferred).
  static std::vector<Device*> FilterSupportedDevices(
      const std::vector<Device*>& devices,
      const PrioritizedDeviceTypeVector& supported_device_types,
      const Device* default_local_device);

 private:
  // Adds each node of the Graph to this ColocationGraph as a singleton.
  //
  // NOTE: The implementation assumes that the ids of nodes passed to
  // this method are dense and zero-based; the memory used will be linear in
  // the largest node ID.
  // NOTE: If this method returns an error, *this is left in an undefined
  // state.
  Status ColocateAllNodes();

  Status ColocateResourceOrRefEdge(const Node* src, const Node* dst);

  // Adds colocation constraints to data types known not to support copying.
  Status ColocateUncopiableTypeEdges(
      std::unordered_set<Node*>* inspection_required);

  // Updates this ColocationGraph by making sure that all nodes
  // touching resource and/or ref tensors are colocated.
  // As it iterates over the edges, fills the `inspection_required` set with
  // the nodes that
  // PlacerInspectionRequiredOpChecker::IsPlacerInspectionRequired
  // deems as requiring deep inspection by placer. This is an optimization.
  // TODO(mdan): Deprecate in favor of ColocateUncopiableTypeEdges.
  Status ColocateResourceAndRefEdges(
      std::unordered_set<Node*>* inspection_required);

  // Updates this ColocationGraph by making sure that all nodes having inputs of
  // a DT_VARIANT data type with a host-only underlying types (e.g. strings) can
  // be placed only on CPU device. We do that by reverse-DFS traversal from all
  // nodes that take variant inputs to the node that produces that variant.
  // TODO(ezhulenev): This function does not yet support "deep op" inspection,
  // that we have for DT_RESOURCE edges.
  Status AddHostOnlyDataTypesConstraints();

  Status AddInspectionConstraints(
      const std::unordered_set<Node*>& inspection_required);

  // Applies colocation groups for `node`'s inputs and outputs to this
  // ColocationGraph.
  // `groups` are the colocation groups to which `nodes`'s inputs and outputs
  // belong.
  // `node` is a node requiring deep inspection (e.g. a node calling
  // a function)
  //
  // For example, consider a `node` taking two inputs and producing one output
  //    a  b
  //    |  |
  //    v  v
  //    node
  //     |
  //     v
  //     c
  //
  // `groups` can tell us that `a` and `c` must be colocated and their device
  // must be a GPU. `b` might be in a group by itself without any device
  // restrictions.
  //
  // ApplyIOColocationGroups will have an effect of calling
  // ColocateNodes(a, c) and LimitToPossibleDevices(`a`, "GPU"). The colocation
  // group of the `node` itself is not directly impacted.
  //
  Status ApplyIOColocationGroups(const IOColocationGroups& groups,
                                 const Node& node);

  Status ColocateNodeToGroup(
      std::unordered_map<StringPiece, const Node*, StringPieceHasher>*
          colocation_group_root,
      const Node* node, StringPiece colocation_group);

  // Merge the (possibly disjoint) sets containing nodes "x" and
  // "y". Returns OK if the all nodes in the union of these sets can
  // be placed on the same device type.
  //
  // If this method returns an error, *this is unchanged.
  Status ColocateNodes(const Node& x, const Node& y);

  // This overload of ColocateNodes() allows a caller to provide the root node
  // ids for the two nodes. For large graphs, this noticeably reduces the
  // graph load time.
  // If this method returns an error, *this is unchanged.
  Status ColocateNodes(const Node& x, int x_root, const Node& y, int y_root);

  void GetSoftDeviceCandidates(const Node& node, const Member& root_member,
                               int root_id,
                               std::vector<Device*>* possible_devices);

  Status InitializeMembers();

  Status InitializeMemberWithAssignedDevice(const string& assigned_device_name,
                                            const string& node_type,
                                            Member* member);

  Status InitializeMember(const Node& node, Member* member);

  // Returns the root node of the disjoint tree to which the node with the
  // given id is connected.
  // FindRoot should be called only for debugging or after the members have
  // been updated with direct root pointers because it does not update
  // root pointers and can traverse many links. It exists to have
  // a const version of FindAndUpdateRoot
  int FindRoot(int node_id) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScolocation_graphDTh mht_6(mht_6_v, 556, "", "./tensorflow/core/common_runtime/colocation_graph.h", "FindRoot");

    return Member::FindRoot(members_, node_id);
  }

  const Graph& graph_;
  const FunctionStack stack_;
  std::vector<Member> members_;
  InspectingPlacer inspecting_placer_;
  PlacerInspectionRequiredOpChecker inspection_required_checker_;
  const DeviceSet& device_set_;
  const std::vector<DeviceType> device_types_;
  const DeviceNameUtils::ParsedName local_address_spec_;
  const Device* default_local_device_;
  const bool allow_soft_placement_;
  const bool log_device_placement_;

  TF_DISALLOW_COPY_AND_ASSIGN(ColocationGraph);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COLOCATION_GRAPH_H_
