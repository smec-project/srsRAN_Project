import re
from datetime import datetime

class RequestAnalyzer:
    def __init__(self):
        # Store events for each RNTI
        self.events = {}
        # Store request timings for analysis
        self.request_timings = {}
        
    def parse_log_file(self, filename):
        """
        Parse the log file and extract SR, BSR, and PRB events
        """
        with open(filename, 'r') as f:
            for line in f:
                try:
                    # Skip lines without timestamp
                    if 'at ' not in line:
                        continue
                        
                    # Extract timestamp first
                    timestamp = float(line.split('at ')[-1].strip())
                    
                    # Parse different event types
                    if 'SR received' in line:
                        # Extract RNTI without 0x prefix (will be added when needed)
                        rnti = line.split('RNTI=')[1].split(',')[0].replace('0x', '')
                        slot = int(line.split('slot=')[1].split()[0])
                        self.add_event(rnti, timestamp, 'SR', {'slot': slot})
                        
                    elif 'bsr received' in line.lower():
                        # Extract RNTI without 0x prefix
                        parts = line.split('RNTI=')[1].split(',')
                        rnti = parts[0].replace('0x', '')
                        # Extract bytes value
                        bytes_str = parts[1].split('bytes=')[1].split()[0]
                        bytes_val = int(bytes_str)
                        self.add_event(rnti, timestamp, 'BSR', {'bytes': bytes_val})  # 使用正确的timestamp
                        
                    elif 'PRB received' in line:
                        # Extract RNTI without 0x prefix
                        rnti = line.split('RNTI=')[1].split(',')[0].replace('0x', '')
                        # Extract slot and PRBs
                        parts = line.split(',')
                        slot = int(parts[1].split('=')[1])
                        prbs = int(parts[2].split('=')[1].split()[0])
                        self.add_event(rnti, timestamp, 'PRB', {'prbs': prbs, 'slot': slot})
                        
                except Exception as e:
                    print(f"Error parsing line: {line.strip()}")
                    print(f"Error: {str(e)}")
                    continue

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

    def analyze_requests(self, target_rnti='4603'):
        """
        Analyze requests for a specific RNTI by looking at event patterns
        """
        if target_rnti not in self.events:
            print(f"No events found for RNTI {target_rnti}")
            return []

        events = sorted(self.events[target_rnti], key=lambda x: x['timestamp'])
        requests = []
        last_sr_time = None
        total_sr_count = 0
        last_bsr_bytes = 0
        last_bsr_time = None
        total_prbs_between_bsr = 0

        # Find all SR events and BSR increases
        i = 0
        while i < len(events):
            event = events[i]
            if event['type'] == 'SR':
                total_sr_count += 1
                current_sr_time = event['timestamp']
                
                # Check if this SR is too close to the last one
                if last_sr_time and (current_sr_time - last_sr_time) < 0.010:  # 10ms
                    i += 1
                    continue
                
                # Look for the first BSR after this SR
                next_bsr = None
                next_bsr_idx = None
                found_srs = [(current_sr_time, "Found SR at {}")]
                
                for j, next_event in enumerate(events[i+1:], start=i+1):
                    if next_event['type'] == 'SR':
                        # If we find another SR before BSR and it's within 10ms
                        if (next_event['timestamp'] - current_sr_time) < 0.010:
                            found_srs.append((next_event['timestamp'], 
                                "Found another SR at {} within 10ms, continuing search for BSR"))
                            continue
                        else:
                            break
                    if next_event['type'] == 'BSR':
                        next_bsr = next_event
                        next_bsr_idx = j
                        break
                
                # If we found a BSR and it has bytes > 0, output the whole sequence
                if next_bsr and next_bsr['value']['bytes'] > 0:
                    # Print all found SRs
                    for timestamp, msg in found_srs:
                        print(msg.format(timestamp))
                    print(f"Found BSR with bytes={next_bsr['value']['bytes']} at {next_bsr['timestamp']}\n")
                    
                    request = {
                        'start_time': event['timestamp']
                    }    
                    requests.append(request)
                
                last_sr_time = current_sr_time
            
            # Track PRBs between BSRs
            elif event['type'] == 'PRB':
                if last_bsr_time and event['timestamp'] > last_bsr_time:
                    total_prbs_between_bsr += event['value']['prbs']
            
            # Check for BSR bytes increase or equal BSR with PRB allocations
            elif event['type'] == 'BSR':
                current_bytes = event['value']['bytes']
                current_time = event['timestamp']
                
                if (current_bytes > last_bsr_bytes or 
                    (current_bytes == last_bsr_bytes and total_prbs_between_bsr > 0 and current_bytes > 0) or
                    (current_bytes < last_bsr_bytes and (last_bsr_bytes - current_bytes) < 80 * total_prbs_between_bsr)):
                    
                    if current_bytes > last_bsr_bytes:
                        print(f"Found BSR with increased bytes={current_bytes} at {current_time} (previous: {last_bsr_bytes})\n")
                    elif current_bytes == last_bsr_bytes:
                        print(f"Found BSR with same bytes={current_bytes} at {current_time} with PRB allocations={total_prbs_between_bsr}\n")
                    else:
                        print(f"Found BSR with decreased bytes={current_bytes} at {current_time} (previous: {last_bsr_bytes}, PRB allocations={total_prbs_between_bsr}, bytes diff={last_bsr_bytes - current_bytes})\n")
                    
                    request = {
                        'start_time': current_time
                    }
                    requests.append(request)
                
                last_bsr_bytes = current_bytes
                last_bsr_time = current_time
                total_prbs_between_bsr = 0
            
            i += 1

        # Sort requests by timestamp to ensure correct order
        requests.sort(key=lambda x: x['start_time'])

        print(f"\nAnalysis Summary:")
        print(f"Total SR events: {total_sr_count}")
        print(f"Total complete requests found: {len(requests)}")
        
        # Print request timestamps and intervals
        print("\nRequest Details:")
        print("Index    Timestamp          Interval(ms)")
        print("----------------------------------------")
        for i, req in enumerate(requests):
            if i == 0:
                print(f"{i:<8} {req['start_time']:<18} -")
            else:
                interval = (req['start_time'] - requests[i-1]['start_time']) * 1000  # convert to ms
                print(f"{i:<8} {req['start_time']:<18} {interval:.1f}")

        return requests


def main():
    # Create analyzer instance
    analyzer = RequestAnalyzer()
    
    # Parse log file
    analyzer.parse_log_file('controller-ue1-s30-i16-3ue-s100-i100-low-latency.txt')
    
    # Analyze requests for RNTI 0x4603
    requests = analyzer.analyze_requests('4604')


if __name__ == "__main__":
    main() 