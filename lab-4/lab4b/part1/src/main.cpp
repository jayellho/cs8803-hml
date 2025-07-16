/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include <common/EventQueue.h>
#include <network/Chunk.h>
#include <topology/Mesh2D.h>
#include <iostream>

using namespace NetworkAnalytical;

void chunk_arrived_callback(void* const event_queue_ptr) {
    // typecast event_queue_ptr
    auto* const event_queue = static_cast<EventQueue*>(event_queue_ptr);

    // print chunk arrival time
    const auto current_time = event_queue->get_current_time();
    std::cout << "A chunk arrived at destination at time: " << current_time << " ns" << std::endl;
}

int main() {
    // Instantiate shared resources
    const auto event_queue = std::make_shared<EventQueue>();
    auto* event_queue_ptr = static_cast<void*>(event_queue.get());
    Topology::set_event_queue(event_queue);

    // Construct a 2D Mesh topology
    const auto width = 4;
    const auto height = 4;
    const auto bandwidth = 50;  // 50 GB/s
    const auto latency = 500;  // 500 ns

    const auto topology = std::make_shared<Mesh2D>(width, height, bandwidth, latency);

    const auto npus_count = topology->get_npus_count();
    std::cout << "Mesh topolopgy created with width: " << width << " , height: " << height << ", total NPUs: " << npus_count
              << std::endl;

    // message settings
    const auto chunk_size = 1'048'576;  // 1 MB

    // send four chunks as test cases
    const auto test_cases = {
        std::make_pair(1, 7),
        std::make_pair(5, 14),
        std::make_pair(11, 5),
        std::make_pair(15, 4)
    };

    for (const auto& [src, dest] : test_cases) {
    
        auto route = topology->route(src, dest); // route from src to dest
        std::cout << "Route from NPU " << src << " to NPU " << dest << ": ";
        for (const auto& npu : route) {
            std::cout << npu->get_id() << " ";
        }
        std::cout << std::endl;

        // send a chunk
        auto chunk = std::make_unique<Chunk>(chunk_size, route, chunk_arrived_callback, event_queue_ptr);
        topology->send(std::move(chunk));
    }

    // Run simulation
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    // Print simulation result
    const auto finish_time = event_queue->get_current_time();
    std::cout << "Simulation finished at time: " << finish_time << " ns" << std::endl;

    return 0;
}
