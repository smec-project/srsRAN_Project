import re
import argparse
import numpy as np
from datetime import datetime
import os

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
                            self.add_event(rnti, timestamp, 'REQUEST_START', {'seq': seq_num})
                        elif 'completed in' in line:
                            self.add_event(rnti, timestamp, 'REQUEST_END', {'seq': seq_num})
                        continue

                    if 'SR received' in line:
                        rnti = line.split('RNTI=')[1].split(',')[0].replace('0x', '')
                        self.add_event(rnti, timestamp, 'SR')
                        
                    elif 'bsr received' in line.lower():
                        parts = line.split('RNTI=')[1].split(',')
                        rnti = parts[0].replace('0x', '')
                        bytes_str = parts[1].split('bytes=')[1].split()[0]
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
        quantized = np.zeros(6, dtype=np.float32)  # Added time_diff dimension
        
        quantized[0] = self.EVENT_TYPES[event_type]
        quantized[1] = event['value']['bytes'] if event_type == 'BSR' else -1
        quantized[2] = event['value']['prbs'] if event_type == 'PRB' else -1
        quantized[3] = event['timestamp']
        quantized[4] = event.get('time_diff', 0)  # time difference in ms
        quantized[5] = is_new_request
        
        return quantized

    def analyze_ue_events(self, target_rnti):
        """
        Analyze and quantize all events for a specific RNTI
        """
        if target_rnti not in self.events:
            print(f"No events found for RNTI {target_rnti}")
            return None

        # Sort events by timestamp
        events = sorted(self.events[target_rnti], key=lambda x: x['timestamp'])
        
        # Convert timestamps to relative time (ms since first event)
        base_time = events[0]['timestamp']
        for event in events:
            event['rel_time'] = (event['timestamp'] - base_time) * 1000  # convert to ms
            
        # Alternatively, calculate time differences between consecutive events
        for i in range(1, len(events)):
            events[i]['time_diff'] = (events[i]['timestamp'] - events[i-1]['timestamp']) * 1000

        # Count event types
        sr_count = sum(1 for e in events if e['type'] == 'SR')
        bsr_count = sum(1 for e in events if e['type'] == 'BSR')
        prb_count = sum(1 for e in events if e['type'] == 'PRB')
        req_count = sum(1 for e in events if e['type'] == 'REQUEST_START')
        
        print(f"Total events: {len(events)}")
        print(f"SR: {sr_count}, BSR: {bsr_count}, PRB: {prb_count}, Requests: {req_count}")

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
        requests_without_increase = []

        # Process each event
        for event in events:
            is_new_request = 0
            quantized_events.append(self.quantize_event(event, is_new_request))

        # Process each request independently
        for start_idx, end_idx, seq_num in request_pairs:
            # Find last BSR before this request
            last_bsr = None
            for i in range(start_idx-1, -1, -1):
                if events[i]['type'] == 'BSR':
                    last_bsr = events[i]
                    break
            outside_last_bsr = last_bsr
            
            found_increase = False
            first_bsr_in_request = None
            
            # Process BSRs in this request
            for i in range(start_idx, end_idx+1):
                if events[i]['type'] == 'BSR':
                    current_bsr = events[i]
                    if first_bsr_in_request is None:
                        first_bsr_in_request = i
                        
                    if last_bsr is not None:
                        if current_bsr['value']['bytes'] >= last_bsr['value']['bytes']:
                            quantized_events[i][5] = 1  # Set is_new_request
                            found_increase = True
                            break
                    elif current_bsr['value']['bytes'] > 0:
                        quantized_events[i][5] = 1
                        found_increase = True
                        break
                    last_bsr = current_bsr  # Update last_bsr after comparison
            
            # If no increase found, mark the first BSR in request
            if not found_increase and first_bsr_in_request is not None:
                quantized_events[first_bsr_in_request][5] = 1
                
            if not found_increase:
                # Collect BSRs in this request for analysis
                request_bsrs = [e for e in events[start_idx:end_idx+1] 
                              if e['type'] == 'BSR']
                requests_without_increase.append({
                    'seq': seq_num,
                    'start_time': events[start_idx]['timestamp'],
                    'end_time': events[end_idx]['timestamp'],
                    'bsrs': request_bsrs,
                    'prev_bsr': outside_last_bsr
                })

        quantized_events = np.array(quantized_events)
        
        # Filter out REQUEST_START and REQUEST_END events
        mask = (quantized_events[:, 0] < 3)  # Keep only SR(0), BSR(1), PRB(2)
        quantized_events = quantized_events[mask]
        
        if requests_without_increase:
            print(f"Requests without BSR increase: {len(requests_without_increase)}")
            
        return quantized_events

    def analyze_ue_events_with_sr(self, target_rnti):
        """
        Analyze and quantize events for a specific RNTI, considering both SR and BSR
        """
        if target_rnti not in self.events:
            print(f"No events found for RNTI {target_rnti}")
            return None

        # Sort events by timestamp
        events = sorted(self.events[target_rnti], key=lambda x: x['timestamp'])
        
        # Convert timestamps to relative time
        base_time = events[0]['timestamp']
        for event in events:
            event['rel_time'] = (event['timestamp'] - base_time) * 1000
            
        # Calculate time differences between consecutive events
        for i in range(1, len(events)):
            events[i]['time_diff'] = (events[i]['timestamp'] - events[i-1]['timestamp']) * 1000

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

        # Process events and label SR/BSR
        quantized_events = []
        requests_info = []

        # Process each event
        for event in events:
            is_new_request = 0
            quantized_events.append(self.quantize_event(event, is_new_request))

        # Process each request independently
        for start_idx, end_idx, seq_num in request_pairs:
            # Find last BSR before this request
            last_bsr = None
            for i in range(start_idx-1, -1, -1):
                if events[i]['type'] == 'BSR':
                    last_bsr = events[i]
                    break

            first_sr = None
            first_bsr_increase = None
            
            # Process events in this request
            for i in range(start_idx, end_idx+1):
                if events[i]['type'] == 'SR' and first_sr is None:
                    first_sr = events[i]
                
                elif events[i]['type'] == 'BSR':
                    current_bsr = events[i]
                    if first_bsr_increase is None:
                        if last_bsr is None and current_bsr['value']['bytes'] > 0:
                            first_bsr_increase = current_bsr
                        elif last_bsr is not None and current_bsr['value']['bytes'] >= last_bsr['value']['bytes']:
                            first_bsr_increase = current_bsr
                    last_bsr = current_bsr

            # Determine which event to label
            if first_sr is not None or first_bsr_increase is not None:
                if first_sr is None:
                    event_to_label = first_bsr_increase
                elif first_bsr_increase is None:
                    event_to_label = first_sr
                else:
                    # Choose the earlier event
                    event_to_label = first_sr if first_sr['timestamp'] < first_bsr_increase['timestamp'] else first_bsr_increase

                # Label the chosen event
                event_idx = events.index(event_to_label)
                quantized_events[event_idx][5] = 1
                
                # Store request info for analysis
                requests_info.append({
                    'seq': seq_num,
                    'start_time': events[start_idx]['timestamp'],
                    'labeled_event': event_to_label
                })

        return np.array(quantized_events), requests_info

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

    def print_request_timing_with_sr(self, rnti, requests_info):
        """
        Print timing information for SR+BSR analysis
        """
        if not requests_info:
            return
            
        base_time = min(info['start_time'] for info in requests_info)
        print(f"\nRequest timing analysis for RNTI {rnti} (SR+BSR):")
        print("Format: ReqSeq | ReqStart(ms) | EventType | EventTime(ms) | TimeDiff(ms)")
        
        for info in requests_info:
            start_time_ms = (info['start_time'] - base_time) * 1000
            event_time_ms = (info['labeled_event']['timestamp'] - base_time) * 1000
            time_diff = event_time_ms - start_time_ms
            event_type = info['labeled_event']['type']
            
            print(f"{info['seq']:6d} | {start_time_ms:11.2f} | {event_type:9s} | {event_time_ms:11.2f} | {time_diff:11.2f}")

    def analyze_ue_events_first_event(self, target_rnti):
        """
        Analyze and quantize events for a specific RNTI, labeling first SR/BSR in each request
        """
        if target_rnti not in self.events:
            print(f"No events found for RNTI {target_rnti}")
            return None

        # Sort events by timestamp
        events = sorted(self.events[target_rnti], key=lambda x: x['timestamp'])
        
        # Convert timestamps to relative time
        base_time = events[0]['timestamp']
        for event in events:
            event['rel_time'] = (event['timestamp'] - base_time) * 1000
            
        # Calculate time differences between consecutive events
        for i in range(1, len(events)):
            events[i]['time_diff'] = (events[i]['timestamp'] - events[i-1]['timestamp']) * 1000

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

        # Process events and label first SR/BSR
        quantized_events = []
        requests_info = []

        # Process each event
        for event in events:
            is_new_request = 0
            quantized_events.append(self.quantize_event(event, is_new_request))

        # Process each request independently
        for start_idx, end_idx, seq_num in request_pairs:
            first_event = None
            
            # Find first SR or BSR in this request
            for i in range(start_idx, end_idx+1):
                if events[i]['type'] in ['SR', 'BSR']:
                    first_event = events[i]
                    break

            if first_event is not None:
                # Label the first event
                event_idx = events.index(first_event)
                quantized_events[event_idx][5] = 1
                
                # Store request info for analysis
                requests_info.append({
                    'seq': seq_num,
                    'start_time': events[start_idx]['timestamp'],
                    'labeled_event': first_event
                })

        return np.array(quantized_events), requests_info

    def print_request_timing_first_event(self, rnti, requests_info):
        """
        Print timing information for first event analysis
        """
        if not requests_info:
            return
            
        base_time = min(info['start_time'] for info in requests_info)
        print(f"\nRequest timing analysis for RNTI {rnti} (First Event):")
        print("Format: ReqSeq | ReqStart(ms) | EventType | EventTime(ms) | TimeDiff(ms)")
        
        for info in requests_info:
            start_time_ms = (info['start_time'] - base_time) * 1000
            event_time_ms = (info['labeled_event']['timestamp'] - base_time) * 1000
            time_diff = event_time_ms - start_time_ms
            event_type = info['labeled_event']['type']
            
            print(f"{info['seq']:6d} | {start_time_ms:11.2f} | {event_type:9s} | {event_time_ms:11.2f} | {time_diff:11.2f}")

    def analyze_ue_events_first_bsr(self, target_rnti):
        """
        Analyze and quantize events for a specific RNTI, labeling first BSR in each request
        """
        if target_rnti not in self.events:
            print(f"No events found for RNTI {target_rnti}")
            return None

        # Sort events by timestamp
        events = sorted(self.events[target_rnti], key=lambda x: x['timestamp'])
        
        # Convert timestamps to relative time
        base_time = events[0]['timestamp']
        for event in events:
            event['rel_time'] = (event['timestamp'] - base_time) * 1000
            
        # Calculate time differences between consecutive events
        for i in range(1, len(events)):
            events[i]['time_diff'] = (events[i]['timestamp'] - events[i-1]['timestamp']) * 1000

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

        # Process events and label first BSR
        quantized_events = []
        requests_info = []

        # Process each event
        for event in events:
            is_new_request = 0
            quantized_events.append(self.quantize_event(event, is_new_request))

        # Process each request independently
        for start_idx, end_idx, seq_num in request_pairs:
            first_bsr = None
            
            # Find first BSR in this request
            for i in range(start_idx, end_idx+1):
                if events[i]['type'] == 'BSR':
                    first_bsr = events[i]
                    break

            if first_bsr is not None:
                # Label the first BSR
                event_idx = events.index(first_bsr)
                quantized_events[event_idx][5] = 1
                
                # Store request info for analysis
                requests_info.append({
                    'seq': seq_num,
                    'start_time': events[start_idx]['timestamp'],
                    'labeled_event': first_bsr
                })

        return np.array(quantized_events), requests_info

    def print_request_timing_first_bsr(self, rnti, requests_info):
        """
        Print timing information for first BSR analysis
        """
        if not requests_info:
            return
            
        base_time = min(info['start_time'] for info in requests_info)
        print(f"\nRequest timing analysis for RNTI {rnti} (First BSR):")
        print("Format: ReqSeq | ReqStart(ms) | EventTime(ms) | TimeDiff(ms)")
        
        for info in requests_info:
            start_time_ms = (info['start_time'] - base_time) * 1000
            event_time_ms = (info['labeled_event']['timestamp'] - base_time) * 1000
            time_diff = event_time_ms - start_time_ms
            
            print(f"{info['seq']:6d} | {start_time_ms:11.2f} | {event_time_ms:11.2f} | {time_diff:11.2f}")

    def analyze_all_ues(self):
        """
        Analyze events for all UEs that have requests
        """
        print(f"\nFound {len(self.active_rntis)} RNTIs with requests: {sorted(self.active_rntis)}")
        
        all_ue_data = {}
        all_ue_data_with_sr = {}
        all_ue_data_first_event = {}
        all_ue_data_first_bsr = {}
        
        for rnti in sorted(self.active_rntis):
            print(f"\nAnalyzing RNTI {rnti}:")
            print("\nBSR-only analysis:")
            ue_data = self.analyze_ue_events(rnti)
            if ue_data is not None:
                all_ue_data[rnti] = ue_data
                # self.print_request_timing(rnti)

            print("\nSR+BSR analysis:")
            ue_data_with_sr, requests_info = self.analyze_ue_events_with_sr(rnti)
            if ue_data_with_sr is not None:
                all_ue_data_with_sr[rnti] = ue_data_with_sr
                # self.print_request_timing_with_sr(rnti, requests_info)

            print("\nFirst Event analysis:")
            ue_data_first_event, requests_info = self.analyze_ue_events_first_event(rnti)
            if ue_data_first_event is not None:
                all_ue_data_first_event[rnti] = ue_data_first_event
                # self.print_request_timing_first_event(rnti, requests_info)

            print("\nFirst BSR analysis:")
            ue_data_first_bsr, requests_info = self.analyze_ue_events_first_bsr(rnti)
            if ue_data_first_bsr is not None:
                all_ue_data_first_bsr[rnti] = ue_data_first_bsr
                # self.print_request_timing_first_bsr(rnti, requests_info)
                
        return all_ue_data, all_ue_data_with_sr, all_ue_data_first_event, all_ue_data_first_bsr

def main():
    parser = argparse.ArgumentParser(description='Process log file and generate training data labels')
    parser.add_argument('log_file', help='Path to the log file to process')
    parser.add_argument('--output', help='Output file path for labeled data (.npy)')
    
    args = parser.parse_args()
    
    if args.output is None:
        input_filename = args.log_file.split('/')[-1]
        base_name = input_filename.rsplit('.', 1)[0]
        # Create a subdirectory with the input file name
        output_dir = f"labeled_data/{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        args.output = f"{output_dir}/{base_name}"
    
    labeler = TrainDataLabeler()
    labeler.parse_log_file(args.log_file)
    labeled_data, labeled_data_with_sr, labeled_data_first_event, labeled_data_first_bsr = labeler.analyze_all_ues()
    
    # Save all versions of labeled data
    np.save(f"{args.output}_bsr_only.npy", labeled_data)
    np.save(f"{args.output}_sr_bsr.npy", labeled_data_with_sr)
    np.save(f"{args.output}_first_event.npy", labeled_data_first_event)
    np.save(f"{args.output}_first_bsr.npy", labeled_data_first_bsr)
    print(f"\nLabeled data saved to directory: labeled_data/{base_name}/")
    print(f"Files:")
    print(f"BSR-only: {base_name}_bsr_only.npy")
    print(f"SR+BSR: {base_name}_sr_bsr.npy")
    print(f"First Event: {base_name}_first_event.npy")
    print(f"First BSR: {base_name}_first_bsr.npy")

if __name__ == "__main__":
    main()
