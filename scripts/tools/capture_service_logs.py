"""
Service Log Capture Utility

Captures structured JSON logs from running microservices for log analysis.

Usage:
    python scripts/tools/capture_service_logs.py [service_name] [duration_minutes]

Example:
    python scripts/tools/capture_service_logs.py notification-service 10

This will:
1. Connect to the running service container
2. Stream logs to service_logs.json
3. Properly format as JSON lines
4. Run for specified duration (or until Ctrl+C)

Author: Anomaly Detection System
Date: March 2026
"""

import subprocess
import sys
import os
import json
import signal
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def capture_docker_logs(container_name: str, output_file: str, duration_minutes: int = 10):
    """
    Capture logs from Docker container.
    
    Args:
        container_name: Docker container name
        output_file: Path to output JSON file
        duration_minutes: How long to capture (minutes)
    """
    print(f"📝 Capturing logs from '{container_name}'...")
    print(f"   Output: {output_file}")
    print(f"   Duration: {duration_minutes} minutes")
    print(f"   Press Ctrl+C to stop early\n")
    
    # Open output file
    with open(output_file, 'a') as f:
        try:
            # Docker logs command
            cmd = ['docker', 'logs', '-f', '--tail', '100', container_name]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            line_count = 0
            start_time = datetime.now()
            
            # Stream logs
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                
                # Try to parse as JSON
                try:
                    log_data = json.loads(line)
                    # Already JSON - write as-is
                    f.write(line + '\n')
                    f.flush()
                except json.JSONDecodeError:
                    # Plain text - wrap in JSON structure
                    log_entry = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'service': container_name,
                        'level': 'INFO',
                        'message': line
                    }
                    f.write(json.dumps(log_entry) + '\n')
                    f.flush()
                
                line_count += 1
                
                # Print progress
                if line_count % 10 == 0:
                    elapsed = (datetime.now() - start_time).seconds
                    print(f"  Captured {line_count} lines ({elapsed}s elapsed)", end='\r')
                
                # Check duration
                if (datetime.now() - start_time).seconds >= duration_minutes * 60:
                    print(f"\n\n✓ Duration reached ({duration_minutes} min)")
                    break
            
            process.terminate()
            print(f"\n✓ Captured {line_count} log lines")
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user")
            print(f"✓ Captured {line_count} log lines")
            process.terminate()
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if 'process' in locals():
                process.terminate()


def capture_kubectl_logs(pod_name: str, output_file: str, duration_minutes: int = 10):
    """
    Capture logs from Kubernetes pod.
    
    Args:
        pod_name: Pod name or label selector
        output_file: Path to output JSON file
        duration_minutes: How long to capture
    """
    print(f"📝 Capturing logs from pod '{pod_name}'...")
    print(f"   Output: {output_file}")
    print(f"   Duration: {duration_minutes} minutes")
    print(f"   Press Ctrl+C to stop early\n")
    
    with open(output_file, 'a') as f:
        try:
            # kubectl logs command
            cmd = ['kubectl', 'logs', '-f', '--tail=100', pod_name]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            line_count = 0
            start_time = datetime.now()
            
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    log_data = json.loads(line)
                    f.write(line + '\n')
                    f.flush()
                except json.JSONDecodeError:
                    log_entry = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'service': pod_name,
                        'level': 'INFO',
                        'message': line
                    }
                    f.write(json.dumps(log_entry) + '\n')
                    f.flush()
                
                line_count += 1
                
                if line_count % 10 == 0:
                    elapsed = (datetime.now() - start_time).seconds
                    print(f"  Captured {line_count} lines ({elapsed}s elapsed)", end='\r')
                
                if (datetime.now() - start_time).seconds >= duration_minutes * 60:
                    print(f"\n\n✓ Duration reached ({duration_minutes} min)")
                    break
            
            process.terminate()
            print(f"\n✓ Captured {line_count} log lines")
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user")
            print(f"✓ Captured {line_count} log lines")
            process.terminate()
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if 'process' in locals():
                process.terminate()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/tools/capture_service_logs.py [service_name] [duration_minutes]")
        print()
        print("Examples:")
        print("  python scripts/tools/capture_service_logs.py notification-service 10")
        print("  python scripts/tools/capture_service_logs.py my-pod 5")
        sys.exit(1)
    
    service_name = sys.argv[1]
    duration_minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    output_file = str(PROJECT_ROOT / 'service_logs.json')
    
    # Detect if Docker or Kubernetes
    print("Detecting environment...")
    
    # Try Docker first
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', f'name={service_name}', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if service_name in result.stdout:
            print(f"✓ Found Docker container: {service_name}\n")
            capture_docker_logs(service_name, output_file, duration_minutes)
            sys.exit(0)
    except:
        pass
    
    # Try Kubernetes
    try:
        result = subprocess.run(
            ['kubectl', 'get', 'pod', service_name, '-o', 'name'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(f"✓ Found Kubernetes pod: {service_name}\n")
            capture_kubectl_logs(service_name, output_file, duration_minutes)
            sys.exit(0)
    except:
        pass
    
    print(f"❌ Service '{service_name}' not found in Docker or Kubernetes")
    print(f"   Make sure the service is running")
    sys.exit(1)
