import re
import argparse
import numpy as np
from datetime import datetime
import os
import multiprocessing
from functools import partial
import time

class TrainDataLabeler:
    # Define event type mapping
    EVENT_TYPES = {
        'SR': 0,
        'BSR': 1,
        'PRB': 2,
        'REQUEST_START': 3,
        'REQUEST_END': 4
    }
    
    def __init__(self):
        # Store events for each RNTI
        self.events = {}
        # Store all RNTIs that have requests
        self.active_rntis = set()
        
    def parse_log_file(self, filename):
        """
        Parse the log file and extract all events including request start/end
        """
        request_start_seq = {}
        request_end_seq = {}
        with open(filename, 'r') as f:
            for line in f:
                try:
                    # Skip lines without timestamp
                    if 'at ' not in line:
                        continue
                        
                    # Extract timestamp
                    timestamp = float(line.split('at ')[-1].strip())
                    
                    # Parse request events
                    if 'Request' in line:
                        parts = line.split()
                        seq_num = int(parts[1])
                        rnti = parts[4]
                        self.active_rntis.add(rnti)
                        
                        if 'start at' in line:
                            if rnti not in request_start_seq:
                                request_start_seq[rnti] = 1
                            self.add_event(rnti, timestamp, 'REQUEST_START', {'seq': request_start_seq[rnti]})
                            request_start_seq[rnti] += 1
                        elif 'completed in' in line:
                            if rnti not in request_end_seq:
                                request_end_seq[rnti] = 1
                            self.add_event(rnti, timestamp, 'REQUEST_END', {'seq': request_end_seq[rnti]})
                            request_end_seq[rnti] += 1
                        continue

                    if 'SR received' in line:
                        rnti = line.split('RNTI=')[1].split(',')[0].replace('0x', '')
                        self.add_event(rnti, timestamp, 'SR')
                        
                    elif 'bsr received' in line.lower():
                        parts = line.split('RNTI=')[1].split(',')
                        rnti = parts[0].replace('0x', '')
                        bytes_str = parts[2].split('bytes=')[1].split()[0]
                        bytes_val = int(bytes_str)
                        self.add_event(rnti, timestamp, 'BSR', {'bytes': bytes_val})
                        
                    elif 'PRB received' in line:
                        rnti = line.split('RNTI=')[1].split(',')[0].replace('0x', '')
                        parts = line.split(',')
                        prbs = int(parts[2].split('=')[1].split()[0])
                        self.add_event(rnti, timestamp, 'PRB', {'prbs': prbs})
                        
                except Exception as e:
                    print(f"Error parsing line: {line.strip()}")
                    print(f"Error: {str(e)}")
                    continue

        # After parsing, remove events for RNTIs without requests
        print(request_start_seq, request_end_seq)
        self.events = {rnti: events for rnti, events in self.events.items() if rnti in self.active_rntis}

    def add_event(self, rnti, timestamp, event_type, value=None):
        """
        Add an event to the events dictionary
        """
        if rnti not in self.events:
            self.events[rnti] = []
        self.events[rnti].append({
            'timestamp': timestamp,
            'type': event_type,
            'value': value
        })

    def quantize_event(self, event, is_new_request=0):
        """
        Quantize a single event into the required format with label
        Returns: [type, bytes, prbs, rel_time, time_diff, is_new_request]
        """
        event_type = event['type']
        quantized = np.zeros(6, dtype=np.float32)
        
        # Convert absolute timestamp to relative timestamp (milliseconds)
        base_time = event.get('base_time', event['timestamp'])
        rel_time = (event['timestamp'] - base_time) * 1000  # Convert to milliseconds
        
        quantized[0] = self.EVENT_TYPES[event_type]
        quantized[1] = event['value']['bytes'] if event_type == 'BSR' else 0
        quantized[2] = event['value']['prbs'] if event_type == 'PRB' else 0
        quantized[3] = rel_time  # Store relative time in milliseconds
        quantized[4] = event.get('time_diff', 0)  # Already in milliseconds
        quantized[5] = is_new_request
        
        return quantized

    def analyze_ue_events(self, target_rnti):
        """
        Analyze and quantize all events for a specific RNTI
        Returns:
            quantized_events: numpy array of quantized events
            bsr_request_map: dictionary mapping BSR index to request sequence number
        """
        if target_rnti not in self.events:
            print(f"No events found for RNTI {target_rnti}")
            return None, None

        # Sort events by timestamp
        events = sorted(self.events[target_rnti], key=lambda x: x['timestamp'])
        
        # Set base time for all events
        base_time = events[0]['timestamp']
        for event in events:
            event['base_time'] = base_time
        
        # Convert timestamps to relative time (ms since first event)
        for event in events:
            event['rel_time'] = (event['timestamp'] - base_time) * 1000  # convert to ms
            
        # Calculate time differences between consecutive SR/BSR/PRB events
        valid_event_types = {'SR', 'BSR', 'PRB'}
        valid_events = [e for e in events if e['type'] in valid_event_types]
        
        # Set time_diff for first valid event
        if valid_events:
            valid_events[0]['time_diff'] = 0
            # Calculate time_diff for subsequent valid events
            for i in range(1, len(valid_events)):
                time_diff = (valid_events[i]['timestamp'] - valid_events[i-1]['timestamp']) * 1000
                valid_events[i]['time_diff'] = time_diff
        
        # Create a mapping of timestamps to time_diffs for valid events
        time_diff_map = {e['timestamp']: e['time_diff'] for e in valid_events}
        
        # Apply time_diffs to original events list
        for event in events:
            event['time_diff'] = time_diff_map.get(event['timestamp'], 0)

        # Count event types
        sr_count = sum(1 for e in events if e['type'] == 'SR')
        bsr_count = sum(1 for e in events if e['type'] == 'BSR')
        prb_count = sum(1 for e in events if e['type'] == 'PRB')
        req_count = sum(1 for e in events if e['type'] == 'REQUEST_START')
        
        # print(f"Total events: {len(events)}")
        # print(f"SR: {sr_count}, BSR: {bsr_count}, PRB: {prb_count}, Requests: {req_count}")

        # Find all request start-end pairs with their indices
        request_pairs = []  # [(start_idx, end_idx, seq_num), ...]
        for i, event in enumerate(events):
            if event['type'] == 'REQUEST_START':
                seq_num = event['value']['seq']
                # Find corresponding end
                for j in range(i+1, len(events)):
                    if (events[j]['type'] == 'REQUEST_END' and 
                        events[j]['value']['seq'] == seq_num):
                        request_pairs.append((i, j, seq_num))
                        break

        # Process events and label BSRs
        quantized_events = []
        bsr_request_map = {}  # Map to store BSR index to request sequence number

        # Process each event
        for event in events:
            is_new_request = 0
            quantized_events.append(self.quantize_event(event, is_new_request))

        bsr_gap_threshold = 2
        # Process each request independently
        for start_idx, end_idx, seq_num in request_pairs:
            request_start_time = events[start_idx]['timestamp']
            
            # First find the first unlabeled BSR in this request
            first_bsr_idx = None
            for i in range(start_idx, end_idx+1):
                if events[i]['type'] == 'BSR':
                    time_diff = (events[i]['timestamp'] - request_start_time) * 1000
                    if time_diff > bsr_gap_threshold:
                        first_bsr_idx = i
                        break
            
            if first_bsr_idx is None:
                continue  # No unlabeled BSR in this request
                
            # Now find the last BSR before the first unlabeled BSR
            last_bsr = None
            prb_count = 0
            for i in range(first_bsr_idx-1, -1, -1):
                if events[i]['type'] == 'BSR':
                    last_bsr = events[i]
                    break
                if events[i]['type'] == 'PRB':
                    prb_count += events[i]['value']['prbs']
            outside_last_bsr = last_bsr
            
            found_increase = False
            
            # Process BSRs in this request, starting from first unlabeled BSR
            for i in range(start_idx, end_idx+1):
                if events[i]['type'] == 'BSR':
                    current_bsr = events[i]
                    time_diff = (current_bsr['timestamp'] - request_start_time) * 1000  # convert to ms
                    
                    if last_bsr is not None:
                        if current_bsr['value']['bytes'] > last_bsr['value']['bytes'] and time_diff > bsr_gap_threshold:
                            quantized_events[i][5] = 1
                            if bsr_request_map.get(i) is None:
                                bsr_request_map[i] = [seq_num]
                            else:
                                bsr_request_map[i].append(seq_num)
                            found_increase = True
                            break
                        elif current_bsr['value']['bytes'] == last_bsr['value']['bytes'] and prb_count > 50 and current_bsr['value']['bytes'] != 0 and time_diff > bsr_gap_threshold:
                            quantized_events[i][5] = 1
                            if bsr_request_map.get(i) is None:
                                bsr_request_map[i] = [seq_num]
                            else:
                                bsr_request_map[i].append(seq_num)
                            found_increase = True
                            break
                        elif current_bsr['value']['bytes'] < last_bsr['value']['bytes'] and prb_count > 50 and current_bsr['value']['bytes'] != 0 and time_diff > bsr_gap_threshold:
                            quantized_events[i][5] = 1
                            if bsr_request_map.get(i) is None:
                                bsr_request_map[i] = [seq_num]
                            else:
                                bsr_request_map[i].append(seq_num)
                            found_increase = True
                            break
                        else:
                            prb_count = 0

                    elif current_bsr['value']['bytes'] > 0 and time_diff > bsr_gap_threshold:
                        quantized_events[i][5] = 1
                        if bsr_request_map.get(i) is None:
                            bsr_request_map[i] = [seq_num]
                        else:
                            bsr_request_map[i].append(seq_num)
                        found_increase = True
                        break
                    last_bsr = current_bsr  
                elif events[i]['type'] == 'PRB':
                    prb_count += events[i]['value']['prbs']
            
            # If no increase found, mark the first unlabeled BSR
            if not found_increase:
                quantized_events[first_bsr_idx][5] = 1
                if bsr_request_map.get(first_bsr_idx) is None:
                    bsr_request_map[first_bsr_idx] = [seq_num]
                else:
                    bsr_request_map[first_bsr_idx].append(seq_num)

        quantized_events = np.array(quantized_events)
        
        # Filter out REQUEST_START and REQUEST_END events
        mask = (quantized_events[:, 0] < 3)
        quantized_events = quantized_events[mask]
        labeled_count = sum(1 for e in quantized_events if e[5] == 1)
        print(f"Labeled BSRs: {labeled_count}")
        
        return quantized_events, bsr_request_map

    def print_request_timing(self, rnti):
        if rnti not in self.events:
            return
        
        events = sorted(self.events[rnti], key=lambda x: x['timestamp'])
        base_time = events[0]['timestamp']

        request_starts = [(i, e) for i, e in enumerate(events) 
                         if e['type'] == 'REQUEST_START']
        
        print(f"\nRequest timing analysis for RNTI {rnti}:")
        print("Format: ReqSeq | ReqStart(ms) | LabeledBSR(ms) | TimeDiff(ms)")
        
        for start_idx, start_event in request_starts:
            seq_num = start_event['value']['seq']
            start_time_ms = (start_event['timestamp'] - base_time) * 1000
            
            labeled_bsr = None
            for i in range(start_idx, len(events)):
                if (events[i]['type'] == 'REQUEST_END' and 
                    events[i]['value']['seq'] == seq_num):
                    break
                if events[i]['type'] == 'BSR':
                    event_idx = events.index(events[i])
                    if event_idx < len(self.events[rnti]):
                        labeled_bsr = events[i]
                        labeled_time_ms = (labeled_bsr['timestamp'] - base_time) * 1000
                        time_diff = labeled_time_ms - start_time_ms
                        print(f"{seq_num:6d} | {start_time_ms:11.2f} | {labeled_time_ms:12.2f} | {time_diff:11.2f}")
                        break
            
            if labeled_bsr is None:
                print(f"{seq_num:6d} | {start_time_ms:11.2f} | {'N/A':>12} | {'N/A':>11}")

    def generate_full_events(self, target_rnti, bsr_request_map):
        """
        Generate a dataset containing all events including REQUEST_START/END
        The 6th column will store request sequence number
        The 7th column will store BSR labels (from bsr_only analysis)
        """
        if target_rnti not in self.events:
            print(f"No events found for RNTI {target_rnti}")
            return None
        
        # Sort events by timestamp
        events = sorted(self.events[target_rnti], key=lambda x: x['timestamp'])
        
        # Set base time for all events
        base_time = events[0]['timestamp']
        for event in events:
            event['base_time'] = base_time
        
        # Calculate time differences between all consecutive events
        for i in range(1, len(events)):
            time_diff = (events[i]['timestamp'] - events[i-1]['timestamp']) * 1000
            events[i]['time_diff'] = time_diff
        events[0]['time_diff'] = 0

        
        # Process events and store request sequence numbers
        quantized_events = []
        seq_num_list = []
        bsr_request_seq_set = set()

        for i, event in enumerate(events):
            # Initialize event data
            event_type = self.EVENT_TYPES[event['type']]
            bsr_bytes = event['value']['bytes'] if event['type'] == 'BSR' else 0
            prbs = event['value']['prbs'] if event['type'] == 'PRB' else 0
            rel_time = (event['timestamp'] - base_time) * 1000
            time_diff = event['time_diff']
            
            # Set request sequence number
            if event['type'] in ['REQUEST_START', 'REQUEST_END']:
                seq_num = event['value']['seq']
                seq_num_list.append(seq_num)
            else:
                seq_num = 0
            
            # Set BSR label using index mapping
            bsr_request_seq = bsr_request_map.get(i, [0])
            # if len(bsr_request_seq) > 1:
            #     bsr_request_seq_set.add(bsr_request_seq[-1])
            #     print(f"bsr_request_seq: {bsr_request_seq}")

            # Create quantized event with BSR label
            quantized_event = np.array([
                event_type,      # Event type (SR=0, BSR=1, PRB=2, REQ_START=3, REQ_END=4)
                bsr_bytes,       # BSR bytes
                prbs,           # PRBs
                rel_time,       # Relative time (ms)
                time_diff,      # Time difference from previous event (ms)
                seq_num,        # Request sequence number
                bsr_request_seq[-1] # BSR label (from bsr_only analysis)
            ], dtype=np.float32)
            
            quantized_events.append(quantized_event)

        # for seq_num in seq_num_list:
        #     if seq_num not in bsr_request_seq_set:
        #         print(f"seq_num not in bsr_request_seq_set: {seq_num}:bsr")
        
        return np.array(quantized_events)

    def analyze_all_ues(self):
        """
        Analyze events for all UEs and generate both filtered and full datasets
        """
        print(f"\nFound {len(self.active_rntis)} RNTIs with requests: {sorted(self.active_rntis)}")
        
        # Create process pool
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_cores)
        
        # Prepare arguments for parallel processing
        process_func = partial(process_single_rnti, events_dict=self.events)
        
        # Process RNTIs in parallel
        results = pool.map(process_func, sorted(self.active_rntis))
        
        pool.close()
        pool.join()
        
        # Organize results
        all_ue_data = {}
        all_ue_full_data = {}  # Dictionary for full event data
        
        for rnti, (ue_data, bsr_request_map, ue_full_data) in results:
            if ue_data is not None:
                all_ue_data[rnti] = ue_data
                all_ue_full_data[rnti] = ue_full_data
        
        return all_ue_data, all_ue_full_data

def print_numpy_data_info(label_type, data):
    """
    Print information about the labeled numpy data
    """
    print(f"\n{label_type} Data Analysis:")
    print(f"Number of RNTIs: {len(data)}")
    
    for rnti, ue_data in data.items():
        print(f"\nRNTI {rnti}, {label_type}:")
        # Count total events
        total_events = len(ue_data)
        # Count event types
        sr_count = np.sum(ue_data[:, 0] == 0)  # SR type is 0
        bsr_count = np.sum(ue_data[:, 0] == 1)  # BSR type is 1
        prb_count = np.sum(ue_data[:, 0] == 2)  # PRB type is 2
        # Count labeled events
        labeled_count = np.sum(ue_data[:, 5] == 1)
        
        print(f"Total events: {total_events}")
        print(f"Event distribution: SR={sr_count}, BSR={bsr_count}, PRB={prb_count}")
        print(f"Labeled events: {labeled_count}")
        
        print("\nAll events details:")
        print("Type | Timestamp(ms) | BSR bytes | PRBs | TimeDiff(ms) | Label")
        for event in ue_data:
            event_type = "SR" if event[0] == 0 else "BSR" if event[0] == 1 else "PRB"
            label = "1" if event[5] == 1 else "0"
            print(f"{event_type:4} | {event[3]:11.2f} | {event[1]:9.0f} | {event[2]:4.0f} | {event[4]:11.2f} | {label:5}")
        print("-" * 70)

def print_full_events_info(data):
    """
    Print information about the full events data (including REQUEST events)
    Args:
        data: dictionary of RNTI to full event arrays
    """
    print("\nFull Events Data Analysis:")
    print(f"Number of RNTIs: {len(data)}")
    
    for rnti, ue_data in data.items():
        print(f"\nRNTI {rnti}:")
        # Count total events
        total_events = len(ue_data)
        
        # Count event types
        sr_count = np.sum(ue_data[:, 0] == 0)     # SR type is 0
        bsr_count = np.sum(ue_data[:, 0] == 1)    # BSR type is 1
        prb_count = np.sum(ue_data[:, 0] == 2)    # PRB type is 2
        req_start_count = np.sum(ue_data[:, 0] == 3)  # REQUEST_START is 3
        req_end_count = np.sum(ue_data[:, 0] == 4)    # REQUEST_END is 4
        
        # Count unique request sequences (excluding 0)
        unique_requests = len(np.unique(ue_data[ue_data[:, 5] > 0, 5]))
        
        print(f"Total events: {total_events}")
        print(f"Event distribution:")
        print(f"  SR: {sr_count}")
        print(f"  BSR: {bsr_count}")
        print(f"  PRB: {prb_count}")
        print(f"  REQUEST_START: {req_start_count}")
        print(f"  REQUEST_END: {req_end_count}")
        print(f"Number of unique requests: {unique_requests}")
        
        print("\nAll events details:")
        print("Type      | Timestamp(ms) | BSR bytes | PRBs | TimeDiff(ms) | ReqSeq | BSR_Label")
        for event in ue_data:
            event_type = {
                0: "SR",
                1: "BSR",
                2: "PRB",
                3: "REQ_START",
                4: "REQ_END"
            }.get(event[0], "UNKNOWN")
            
            req_seq = int(event[5])
            req_seq_str = str(req_seq) if req_seq > 0 else "0"
            bsr_label = str(int(event[6])) + ":bsr" if event[6] > 0 else "0"
            
            print(f"{event_type:9} | {event[3]:11.2f} | {event[1]:9.0f} | {event[2]:4.0f} | {event[4]:11.2f} | {req_seq_str:6} | {bsr_label:9}")
        print("-" * 80)

def process_single_rnti(rnti, events_dict=None):
    """
    Process a single RNTI's data
    """
    labeler = TrainDataLabeler()
    labeler.events = {rnti: events_dict[rnti]}
    labeler.active_rntis = {rnti}
    
    # Run all analyses
    ue_data, bsr_request_map = labeler.analyze_ue_events(rnti)
    ue_full_data = labeler.generate_full_events(rnti, bsr_request_map)
    
    return rnti, (ue_data, bsr_request_map, ue_full_data)

def main():
    parser = argparse.ArgumentParser(description='Process log file and generate training data labels')
    parser.add_argument('log_file', help='Path to the log file to process')
    parser.add_argument('--output', help='Output file path for labeled data (.npy)')
    parser.add_argument('--threads', type=int, help='Number of threads to use', default=16)
    parser.add_argument('--base-dir', default='labeled_data', 
                       help='Base directory for output (default: labeled_data)')
    
    args = parser.parse_args()
    
    if args.output is None:
        input_filename = args.log_file.split('/')[-1]
        base_name = input_filename.rsplit('.', 1)[0]
        output_dir = f"{args.base_dir}/{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        args.output = f"{output_dir}/{base_name}"
    
    # Set number of processes
    if args.threads:
        multiprocessing.set_start_method('spawn')
        pool = multiprocessing.Pool(processes=args.threads)
    
    start_time = time.time()
    
    labeler = TrainDataLabeler()
    labeler.parse_log_file(args.log_file)
    labeled_data, labeled_full_data = labeler.analyze_all_ues()
    
    end_time = time.time()
    print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
    
    # Save all versions of labeled data
    np.save(f"{args.output}_bsr_only.npy", labeled_data)
    np.save(f"{args.output}_full_events.npy", labeled_full_data)  # Save full events data
    
    # Print information about saved data
    # print_numpy_data_info("BSR-only", labeled_data)
    # print_full_events_info(labeled_full_data)  # Print full events info
    
    print(f"\nLabeled data saved to directory: labeled_data/{base_name}/")
    print(f"Files:")
    print(f"BSR-only: {base_name}_bsr_only.npy")
    print(f"Full Events: {base_name}_full_events.npy")  # Add to output list

if __name__ == "__main__":
    main()
