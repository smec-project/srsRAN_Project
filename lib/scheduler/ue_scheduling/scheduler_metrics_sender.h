/*
 *
 * Copyright 2021-2024 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#pragma once

#include "srsran/du/du_cell_config.h"
#include "srsran/ran/rnti.h"
#include "srsran/ran/slot_point.h"
#include "srsran/support/srsran_assert.h"
// #include "srsran/support/format/format_utils.h"
#include "srsran/ran/du_types.h"
#include "srsran/srslog/srslog.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace srsran {

// Define message types
enum class metrics_type {
  PRB_ALLOC, // PRB allocation metrics
  SR_IND,    // SR indication
  BSR_IND    // BSR indication
};

// Base metrics structure
struct base_metrics {
  metrics_type  type;
  du_ue_index_t ue_index;
  rnti_t        crnti;
};

// PRB allocation metrics
struct prb_metrics : public base_metrics {
  unsigned   nof_prbs;
  slot_point slot;
};

// SR indication metrics
struct sr_metrics : public base_metrics {
  slot_point slot;
};

// BSR metrics
struct bsr_metrics : public base_metrics {
  unsigned   nof_bytes; // Total buffer size in bytes
  slot_point slot_rx;
};

/// \brief Class responsible for sending scheduler metrics over TCP connection.
///
/// Sends metrics in binary format as 4 32-bit integers:
/// - PRB: [type=0][rnti][prbs][slot]
/// - SR:  [type=1][rnti][slot][0]
/// - BSR: [type=2][rnti][bytes][slot]
class scheduler_metrics_sender
{
public:
  static constexpr int DEFAULT_METRICS_PORT = 5556;

  // Binary message structure: 4 x 32-bit integers = 16 bytes
  struct metrics_message {
    uint32_t type;   // 0=PRB, 1=SR, 2=BSR
    uint32_t rnti;   // RNTI value
    uint32_t field1; // PRBs/slot/bytes depending on type
    uint32_t field2; // slot/0/slot depending on type
  };

  static scheduler_metrics_sender& instance()
  {
    static scheduler_metrics_sender instance;
    return instance;
  }

  // Delete copy/move constructors
  scheduler_metrics_sender(const scheduler_metrics_sender&)            = delete;
  scheduler_metrics_sender& operator=(const scheduler_metrics_sender&) = delete;

private:
  // Private constructor
  scheduler_metrics_sender();

public:
  ~scheduler_metrics_sender() { stop(); }

  /// \brief Initialize the TCP server.
  /// \param[in] port TCP port to listen on.
  /// \return True on success, false otherwise.
  bool init(int port = DEFAULT_METRICS_PORT);

  /// \brief Stop the TCP server and cleanup resources.
  void stop();

  /// \brief Send binary message to connected client.
  /// \param[in] msg Binary message to send.
  /// \return True if message was sent successfully, false otherwise.
  bool send_binary_message(const metrics_message& msg);

  bool send_prb_metrics(const prb_metrics& metrics);
  bool send_sr_metrics(const sr_metrics& metrics);
  bool send_bsr_metrics(const bsr_metrics& metrics);

private:
  /// \brief Create binary metrics message.
  /// \param[in] metrics UE scheduling metrics to format.
  /// \return Binary message structure.
  metrics_message create_prb_message(const prb_metrics& metrics);
  metrics_message create_sr_message(const sr_metrics& metrics);
  metrics_message create_bsr_message(const bsr_metrics& metrics);

  /// \brief Thread function to handle incoming connections.
  void accept_connections();

  std::mutex            mutex;
  int                   server_fd;
  int                   client_fd;
  std::atomic<bool>     running;
  std::thread           accept_thread;
  srslog::basic_logger& logger;
};

} // namespace srsran