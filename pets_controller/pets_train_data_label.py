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
    
    # BSR index to value mapping
    BSR_VALUES = [
        0,     10,    14,    20,    28,    38,    53,    74,     # 0-7
        102,   142,   198,   276,   384,   535,   745,   1038,   # 8-15
        1446,  2014,  2806,  3909,  5446,  7587,  10570, 14726,  # 16-23
        20516, 28581, 39818, 55474, 77284, 107669,150000,300000  # 24-31
    ]
    
    def __init__(self):
        # Store events for each RNTI
        self.events = {}
        # Store all RNTIs that have requests
        self.active_rntis = set()
        # Store request sizes for each RNTI
        self.request_sizes = {}
        
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
                        
                    # Extract request size from UE registration
                    if 'New UE registered' in line:
                        parts = line.split(',')
                        rnti = parts[0].split('RNTI:')[1].strip().replace('0x', '')
                        size = int(parts[3].split('Size:')[1].split()[0])
                        self.request_sizes[rnti] = size
                        continue

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
                        parts = line.split('RNTI=')[1].split(',')
                        rnti = parts[0].replace('0x', '')
                        slot = int(parts[1].split('slot=')[1].split()[0])
                        event = self.add_event(rnti, timestamp, 'SR')
                        event['slot'] = slot
                        
                    elif 'bsr received' in line.lower():
                        parts = line.split('RNTI=')[1].split(',')
                        rnti = parts[0].replace('0x', '')
                        slot = int(parts[1].split('slot=')[1].split()[0])
                        bytes_str = parts[2].split('bytes=')[1].split()[0]
                        bytes_val = int(bytes_str)
                        event = self.add_event(rnti, timestamp, 'BSR', {'bytes': bytes_val})
                        event['slot'] = slot
                        
                    elif 'PRB received' in line:
                        parts = line.split('RNTI=')[1].split(',')
                        rnti = parts[0].replace('0x', '')
                        slot = int(parts[1].split('slot=')[1].split()[0])
                        prbs = int(parts[2].split('prbs=')[1].split()[0])
                        event = self.add_event(rnti, timestamp, 'PRB', {'prbs': prbs})
                        event['slot'] = slot
                        
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
        Returns:
            dict: The added event data
        """
        if rnti not in self.events:
            self.events[rnti] = []
        
        event_data = {
            'timestamp': timestamp,
            'type': event_type,
            'value': value if value else {},
            'slot': None
        }
        self.events[rnti].append(event_data)
        return event_data

    def quantize_event(self, event, is_new_request=0, base_time=0):
        """
        Quantize a single event into the required format with label
        Returns: [type, bytes, prbs, timestamp, slot, is_new_request]
        """
        event_type = event['type']
        quantized = np.zeros(6, dtype=np.float32)
        
        # Set event type
        quantized[0] = self.EVENT_TYPES[event_type]
        
        # Set BSR bytes or PRB count
        quantized[1] = event['value']['bytes'] if event_type == 'BSR' else 0
        quantized[2] = event['value']['prbs'] if event_type == 'PRB' else 0
        
        # Set timestamp
        quantized[3] = (event['timestamp'] - base_time) * 1000
        
        # Set slot (use -1 for request events)
        quantized[4] = event['slot'] if event['slot'] is not None else -1
        
        # Set request label
        quantized[5] = is_new_request
        
        return quantized

    def normalize_slots(self, events):
        """
        Convert cyclic slots (0-20480) into a continuous increasing sequence
        and make them relative to the first event
        """
        SLOT_MAX = 20480
        current_cycle = 0
        last_slot = None
        
        # First convert to continuous sequence
        for event in events:
            if event['slot'] is not None:
                current_slot = event['slot']
                
                if last_slot is not None:
                    # If current slot is much smaller than last slot, we've wrapped around
                    if current_slot < last_slot - SLOT_MAX//2:
                        current_cycle += 1
                    # If current slot is much larger than last slot, we've gone backwards
                    elif current_slot > last_slot + SLOT_MAX//2:
                        current_cycle -= 1
                
                # Update the slot value with the cycle count
                event['slot'] = current_slot + (current_cycle * SLOT_MAX)
                last_slot = current_slot
        
        # Find first slot value
        first_slot = None
        for event in events:
            if event['slot'] is not None:
                first_slot = event['slot']
                break
        
        # Make slots relative to first slot
        if first_slot is not None:
            for event in events:
                if event['slot'] is not None:
                    event['slot'] = event['slot'] - first_slot

    def sort_events_by_slot(self, events):
        """
        Sort SR, PRB, BSR events by slot and merge with request events
        """
        # Separate events with and without slots
        slot_events = []
        request_events = []
        
        for event in events:
            if event['slot'] is not None:
                slot_events.append(event)
            else:
                request_events.append(event)
        
        # Sort events with slots
        slot_events.sort(key=lambda x: (x['slot'], 
                                      # When slots are equal, PRB comes before BSR
                                      0 if x['type'] == 'PRB' else 1 if x['type'] == 'BSR' else -1))
        
        # Merge sorted events
        sorted_events = []
        slot_idx = 0
        req_idx = 0
        
        while slot_idx < len(slot_events) and req_idx < len(request_events):
            if slot_events[slot_idx]['timestamp'] <= request_events[req_idx]['timestamp']:
                sorted_events.append(slot_events[slot_idx])
                slot_idx += 1
            else:
                sorted_events.append(request_events[req_idx])
                req_idx += 1
        
        # Add remaining events
        sorted_events.extend(slot_events[slot_idx:])
        sorted_events.extend(request_events[req_idx:])
        
        return sorted_events

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
        
        events = self.events[target_rnti]
            
        # Calculate time differences between consecutive SR/BSR/PRB events
        valid_event_types = {'SR', 'BSR', 'PRB'}
        valid_events = [e for e in events if e['type'] in valid_event_types]

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
        base_time = events[0]['timestamp']
        for event in events:
            is_new_request = 0
            quantized_events.append(self.quantize_event(event, is_new_request, base_time))

        # Process each request independently
        for start_idx, end_idx, seq_num in request_pairs:
            # Now find the last BSR before the first unlabeled BSR
            last_bsr = None
            prb_count = 0
            for i in range(start_idx-1, -1, -1):
                if events[i]['type'] == 'BSR':
                    last_bsr = events[i]
                    break
                if events[i]['type'] == 'PRB':
                    prb_count += events[i]['value']['prbs']
            
            found_increase = False
            
            # Process BSRs in this request, starting from first unlabeled BSR
            for i in range(start_idx, end_idx+1):
                if events[i]['type'] == 'BSR':
                    current_bsr = events[i]
                    last_bsr_value_lower_bound = self.get_prev_bsr_value(last_bsr['value']['bytes']) + 1
                    current_bsr_lower_bound_w_request = last_bsr_value_lower_bound + self.request_sizes[target_rnti] - prb_count * 90
                    normalized_current_bsr_lower_bound = self.get_prev_bsr_value(current_bsr_lower_bound_w_request)

                    last_bsr_value_higher_bound = last_bsr['value']['bytes']
                    current_bsr_higher_bound_w_request = last_bsr_value_higher_bound + self.request_sizes[target_rnti] - prb_count * 50
                    normalized_current_bsr_higher_bound = self.get_next_bsr_value(current_bsr_higher_bound_w_request)
                    if target_rnti == '4601':
                        print(seq_num, last_bsr['value']['bytes'], current_bsr['value']['bytes'], normalized_current_bsr_lower_bound, normalized_current_bsr_higher_bound)
                    if last_bsr is not None:
                        if current_bsr['value']['bytes'] > last_bsr['value']['bytes']:
                            quantized_events[i][5] = 1
                            if bsr_request_map.get(i) is None:
                                bsr_request_map[i] = [seq_num]
                            else:
                                bsr_request_map[i].append(seq_num)
                            found_increase = True
                            break
                        elif current_bsr['value']['bytes'] == last_bsr['value']['bytes'] and current_bsr['value']['bytes'] > normalized_current_bsr_lower_bound and current_bsr['value']['bytes'] < normalized_current_bsr_higher_bound:
                            quantized_events[i][5] = 1
                            if bsr_request_map.get(i) is None:
                                bsr_request_map[i] = [seq_num]
                            else:
                                bsr_request_map[i].append(seq_num)
                            found_increase = True
                            break
                        elif current_bsr['value']['bytes'] < last_bsr['value']['bytes'] and current_bsr['value']['bytes'] > normalized_current_bsr_lower_bound and current_bsr['value']['bytes'] < normalized_current_bsr_higher_bound:
                            quantized_events[i][5] = 1
                            if bsr_request_map.get(i) is None:
                                bsr_request_map[i] = [seq_num]
                            else:
                                bsr_request_map[i].append(seq_num)
                            found_increase = True
                            break
                        else:
                            prb_count = 0

                    elif current_bsr['value']['bytes'] > 0:
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
            
            # Only print if request was not labeled
            if not found_increase:
                print(f"{seq_num:6d}")

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
        Returns array with columns:
            [0]: Event type (SR=0, BSR=1, PRB=2, REQ_START=3, REQ_END=4)
            [1]: BSR bytes
            [2]: PRBs
            [3]: Relative time (ms)
            [4]: Slot
            [5]: Request sequence number
            [6]: BSR label
        """
        if target_rnti not in self.events:
            print(f"No events found for RNTI {target_rnti}")
            return None
        
        events = self.events[target_rnti]
        base_time = events[0]['timestamp']
        
        quantized_events = []
        for i, event in enumerate(events):
            # Initialize event data
            event_type = self.EVENT_TYPES[event['type']]
            bsr_bytes = event['value']['bytes'] if event['type'] == 'BSR' else 0
            prbs = event['value']['prbs'] if event['type'] == 'PRB' else 0
            rel_time = (event['timestamp'] - base_time) * 1000
            slot = event['slot'] if event['slot'] is not None else -1
            
            # Set request sequence number
            seq_num = event['value'].get('seq', 0) if event['type'] in ['REQUEST_START', 'REQUEST_END'] else 0
            
            # Set BSR label using index mapping
            bsr_label = bsr_request_map.get(i, [0])[-1]

            # Create quantized event
            quantized_event = np.array([
                event_type,      # Event type
                bsr_bytes,       # BSR bytes
                prbs,           # PRBs
                rel_time,       # Relative time (ms)
                slot,           # Slot
                seq_num,        # Request sequence number
                bsr_label       # BSR label
            ], dtype=np.float32)
            
            quantized_events.append(quantized_event)
        
        return np.array(quantized_events)

    def analyze_all_ues(self):
        """
        Analyze events for all UEs and generate both filtered and full datasets
        """
        print(f"\nFound {len(self.active_rntis)} RNTIs with requests: {sorted(self.active_rntis)}")
        
        # Normalize slots and sort events for each RNTI
        for rnti in self.active_rntis:
            self.normalize_slots(self.events[rnti])
            self.events[rnti] = self.sort_events_by_slot(self.events[rnti])
        
        # Create process pool
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_cores)
        
        # Prepare arguments for parallel processing
        process_func = partial(process_single_rnti, events_dict=self.events, request_sizes=self.request_sizes)
        
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

    def get_prev_bsr_value(self, bsr_bytes):
        """
        Get the previous BSR value
        Args:
            bsr_bytes: Current BSR bytes
        Returns:
            int: Previous BSR value in the table
        """
        for i in range(len(self.BSR_VALUES)-1, -1, -1):
            if bsr_bytes > self.BSR_VALUES[i]:
                return self.BSR_VALUES[i]
        return -1

    def get_next_bsr_value(self, bsr_bytes):
        """
        Get the next BSR value
        Args:
            bsr_bytes: Current BSR bytes
        Returns:
            int: Next BSR value in the table
        """
        for i in range(len(self.BSR_VALUES)):
            if bsr_bytes <= self.BSR_VALUES[i]:
                return self.BSR_VALUES[i]
        return self.BSR_VALUES[-1]

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
        print("Type | Timestamp(ms) | BSR bytes | PRBs | Slot | Label")
        for event in ue_data:
            event_type = "SR" if event[0] == 0 else "BSR" if event[0] == 1 else "PRB"
            label = "1" if event[5] == 1 else "0"
            print(f"{event_type:4} | {event[3]:11.2f} | {event[1]:9.0f} | {event[2]:4.0f} | {event[4]:4.0f} | {label:5}")
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
        print("Type      | Timestamp(ms) | BSR bytes | PRBs | Slot      | ReqSeq | BSR_Label")
        for event in ue_data:
            event_type = {
                0: "SR",
                1: "BSR",
                2: "PRB",
                3: "REQ_START",
                4: "REQ_END"
            }.get(event[0], "UNKNOWN")
            
            slot_str = str(int(event[4])) if event[4] >= 0 else "N/A"
            req_seq = int(event[5])
            req_seq_str = str(req_seq) if req_seq > 0 else "0"
            bsr_label = str(int(event[6])) + ":bsr" if event[6] > 0 else "0"
            
            print(f"{event_type:9} | {event[3]:11.2f} | {event[1]:9.0f} | {event[2]:4.0f} | {slot_str:9} | {req_seq_str:6} | {bsr_label:9}")
        print("-" * 90)

def process_single_rnti(rnti, events_dict=None, request_sizes=None):
    """
    Process a single RNTI's data
    """
    labeler = TrainDataLabeler()
    labeler.events = {rnti: events_dict[rnti]}
    labeler.active_rntis = {rnti}
    labeler.request_sizes = request_sizes
    
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
    print_numpy_data_info("BSR-only", labeled_data)
    print_full_events_info(labeled_full_data)  # Print full events info
    
    print(f"\nLabeled data saved to directory: labeled_data/{base_name}/")
    print(f"Files:")
    print(f"BSR-only: {base_name}_bsr_only.npy")
    print(f"Full Events: {base_name}_full_events.npy")  # Add to output list

if __name__ == "__main__":
    main()
