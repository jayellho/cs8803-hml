/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <list>
#include <memory>

namespace NetworkAnalytical {

/// Forward declarations of network components
class Chunk;
class Link;
class Device;

/// Route is a list of devices
using Route = std::list<std::shared_ptr<Device>>;

/// Callback function pointer: "void func(void*)"
using Callback = void (*)(void*);

/// Callback function argument: void*
using CallbackArg = void*;

/// Device ID which starts from 0
using DeviceId = int;

/// Chunk size in Bytes
using ChunkSize = uint64_t;

/// Bandwidth in GB/s
using Bandwidth = double;

/// Latency in ns
using Latency = double;

/// Event time in ns
using EventTime = uint64_t;

/// Basic multi-dimensional topology building blocks
enum class TopologyBuildingBlock { Undefined, Ring, FullyConnected, Switch, Mesh2D };

}  // namespace NetworkAnalytical
