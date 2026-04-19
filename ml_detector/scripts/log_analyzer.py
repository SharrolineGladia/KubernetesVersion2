"""
Log Analyzer for Root Cause Analysis

Integrates structured logs into the RCA pipeline to provide error context.

Capabilities:
- Parse structured JSON logs from services
- Extract error messages and stack traces
- Correlate logs with traces via trace_id
- Identify error patterns and recurring issues
- Provide detailed error context for RCA

Integrates with: explainability_layer.py for enhanced RCA

Author: Anomaly Detection System
Date: March 2026
"""

import json
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import Counter
import re


@dataclass
class LogEntry:
    """Represents a single log entry."""
    timestamp: datetime
    service: str
    level: str
    message: str
    trace_id: Optional[str]
    span_id: Optional[str]
    additional_fields: Dict
    
    def is_error(self) -> bool:
        """Check if log is an error."""
        return self.level.upper() in ['ERROR', 'CRITICAL', 'FATAL']
    
    def is_warning(self) -> bool:
        """Check if log is a warning."""
        return self.level.upper() == 'WARNING'


@dataclass
class LogSummary:
    """Summary of logs for a time window."""
    time_window_start: datetime
    time_window_end: datetime
    total_logs: int
    error_logs: int
    warning_logs: int
    services_involved: List[str]
    error_patterns: List[Dict]  # Common error messages
    critical_errors: List[Dict]  # Most severe errors
    trace_correlation: Dict[str, List[str]]  # trace_id -> log messages
    
    def to_dict(self) -> Dict:
        return {
            'time_window_start': self.time_window_start.isoformat(),
            'time_window_end': self.time_window_end.isoformat(),
            'total_logs': self.total_logs,
            'error_logs': self.error_logs,
            'warning_logs': self.warning_logs,
            'error_rate': self.error_logs / max(1, self.total_logs),
            'services_involved': self.services_involved,
            'error_patterns': self.error_patterns,
            'critical_errors': self.critical_errors,
            'trace_correlation': trace_correlation
        }


class LogAnalyzer:
    """
    Analyzes structured logs to enhance RCA with error context.
    
    Workflow:
        1. Fetch/parse logs for anomaly time window
        2. Extract error messages and patterns
        3. Correlate logs with traces via trace_id
        4. Identify recurring issues
        5. Provide detailed error context
    """
    
    def __init__(
        self,
        log_source: str = "file",  # 'file', 'loki', 'elasticsearch'
        log_file_path: Optional[str] = None,
        loki_url: Optional[str] = None
    ):
        """
        Initialize log analyzer.
        
        Args:
            log_source: Where to read logs from ('file', 'loki', 'elasticsearch')
            log_file_path: Path to log file (if source='file')
            loki_url: Loki URL (if source='loki')
        """
        self.log_source = log_source
        self.log_file_path = log_file_path
        self.loki_url = loki_url.rstrip('/') if loki_url else None
    
    def analyze_time_window(
        self,
        start_time: datetime,
        end_time: datetime,
        service_name: Optional[str] = None
    ) -> LogSummary:
        """
        Analyze logs for a specific time window.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            service_name: Optional service filter
        
        Returns:
            LogSummary with analysis results
        """
        # Fetch logs
        logs = self._fetch_logs(start_time, end_time, service_name)
        
        if not logs:
            return LogSummary(
                time_window_start=start_time,
                time_window_end=end_time,
                total_logs=0,
                error_logs=0,
                warning_logs=0,
                services_involved=[],
                error_patterns=[],
                critical_errors=[],
                trace_correlation={}
            )
        
        # Analyze logs
        error_logs = [log for log in logs if log.is_error()]
        warning_logs = [log for log in logs if log.is_warning()]
        services = list(set(log.service for log in logs))
        
        # Identify error patterns
        error_patterns = self._identify_error_patterns(error_logs)
        
        # Get critical errors
        critical_errors = self._get_critical_errors(error_logs)
        
        # Build trace correlation
        trace_correlation = self._correlate_with_traces(logs)
        
        return LogSummary(
            time_window_start=start_time,
            time_window_end=end_time,
            total_logs=len(logs),
            error_logs=len(error_logs),
            warning_logs=len(warning_logs),
            services_involved=services,
            error_patterns=error_patterns,
            critical_errors=critical_errors,
            trace_correlation=trace_correlation
        )
    
    def _fetch_logs(
        self,
        start_time: datetime,
        end_time: datetime,
        service_name: Optional[str] = None
    ) -> List[LogEntry]:
        """
        Fetch logs from configured source.
        
        Supports:
        - File: Read from local log file (stdout capture)
        - Loki: Query Loki API
        - Elasticsearch: Query ES API (future)
        """
        if self.log_source == 'file':
            return self._fetch_logs_from_file(start_time, end_time, service_name)
        elif self.log_source == 'loki':
            return self._fetch_logs_from_loki(start_time, end_time, service_name)
        else:
            print(f"⚠️  Unsupported log source: {self.log_source}")
            return []
    
    def _fetch_logs_from_file(
        self,
        start_time: datetime,
        end_time: datetime,
        service_name: Optional[str] = None
    ) -> List[LogEntry]:
        """
        Parse logs from file (JSON lines format).
        
        Expects each line to be a JSON object with:
        - timestamp
        - service
        - level
        - message
        - (optional) trace_id, span_id
        """
        if not self.log_file_path:
            print("⚠️  No log file path configured")
            return []
        
        logs = []
        
        try:
            with open(self.log_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        log_data = json.loads(line)
                        
                        # Parse timestamp
                        log_timestamp = self._parse_timestamp(log_data.get('timestamp'))
                        if not log_timestamp:
                            continue
                        
                        # Filter by time window
                        if not (start_time <= log_timestamp <= end_time):
                            continue
                        
                        # Filter by service
                        if service_name and log_data.get('service') != service_name:
                            continue
                        
                        # Create LogEntry
                        log_entry = LogEntry(
                            timestamp=log_timestamp,
                            service=log_data.get('service', 'unknown'),
                            level=log_data.get('level', 'INFO'),
                            message=log_data.get('message', ''),
                            trace_id=log_data.get('trace_id'),
                            span_id=log_data.get('span_id'),
                            additional_fields={k: v for k, v in log_data.items() 
                                             if k not in ['timestamp', 'service', 'level', 'message', 'trace_id', 'span_id']}
                        )
                        
                        logs.append(log_entry)
                    
                    except json.JSONDecodeError:
                        # Skip non-JSON lines
                        continue
        
        except FileNotFoundError:
            print(f"⚠️  Log file not found: {self.log_file_path}")
            return []
        except Exception as e:
            print(f"⚠️  Error reading logs: {e}")
            return []
        
        return logs
    
    def _fetch_logs_from_loki(
        self,
        start_time: datetime,
        end_time: datetime,
        service_name: Optional[str] = None
    ) -> List[LogEntry]:
        """
        Fetch logs from Loki (Grafana log aggregator).
        
        Uses LogQL query language.
        """
        if not self.loki_url:
            print("⚠️  No Loki URL configured")
            return []
        
        # Build LogQL query
        query = '{job="kubernetes"}'
        if service_name:
            query = f'{{service="{service_name}"}}'
        
        # Convert to nanoseconds
        start_ns = int(start_time.timestamp() * 1_000_000_000)
        end_ns = int(end_time.timestamp() * 1_000_000_000)
        
        try:
            url = f"{self.loki_url}/loki/api/v1/query_range"
            response = requests.get(url, params={
                'query': query,
                'start': start_ns,
                'end': end_ns,
                'limit': 1000
            }, timeout=10)
            
            response.raise_for_status()
            data = response.json()
            
            # Parse Loki response (similar to file parsing)
            logs = []
            for stream in data.get('data', {}).get('result', []):
                for entry in stream.get('values', []):
                    # entry = [timestamp_ns, log_line]
                    timestamp_ns = int(entry[0])
                    log_line = entry[1]
                    
                    # Try to parse as JSON
                    try:
                        log_data = json.loads(log_line)
                        log_timestamp = datetime.fromtimestamp(timestamp_ns / 1_000_000_000)
                        
                        log_entry = LogEntry(
                            timestamp=log_timestamp,
                            service=log_data.get('service', 'unknown'),
                            level=log_data.get('level', 'INFO'),
                            message=log_data.get('message', ''),
                            trace_id=log_data.get('trace_id'),
                            span_id=log_data.get('span_id'),
                            additional_fields={}
                        )
                        
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Plain text log - create basic entry
                        log_entry = LogEntry(
                            timestamp=datetime.fromtimestamp(timestamp_ns / 1_000_000_000),
                            service='unknown',
                            level='INFO',
                            message=log_line,
                            trace_id=None,
                            span_id=None,
                            additional_fields={}
                        )
                        logs.append(log_entry)
            
            return logs
        
        except Exception as e:
            print(f"⚠️  Failed to fetch logs from Loki: {e}")
            return []
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if not timestamp_str:
            return None
        
        # Try ISO format first
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            pass
        
        # Try common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except:
                continue
        
        return None
    
    def _identify_error_patterns(self, error_logs: List[LogEntry]) -> List[Dict]:
        """
        Identify common error patterns.
        
        Groups similar error messages and counts occurrences.
        """
        if not error_logs:
            return []
        
        # Extract error messages
        error_messages = [log.message for log in error_logs]
        
        # Count occurrences
        message_counts = Counter(error_messages)
        
        # Format as patterns
        patterns = []
        for message, count in message_counts.most_common(5):
            patterns.append({
                'pattern': message[:100],  # Truncate long messages
                'count': count,
                'percentage': count / len(error_logs) * 100
            })
        
        return patterns
    
    def _get_critical_errors(self, error_logs: List[LogEntry]) -> List[Dict]:
        """
        Extract most critical errors for RCA.
        
        Returns recent errors with full context.
        """
        # Sort by timestamp (most recent first)
        error_logs_sorted = sorted(error_logs, key=lambda x: x.timestamp, reverse=True)
        
        critical = []
        for log in error_logs_sorted[:5]:  # Top 5 most recent
            critical.append({
                'timestamp': log.timestamp.isoformat(),
                'service': log.service,
                'message': log.message,
                'trace_id': log.trace_id,
                'additional_context': log.additional_fields
            })
        
        return critical
    
    def _correlate_with_traces(self, logs: List[LogEntry]) -> Dict[str, List[str]]:
        """
        Build trace_id -> log messages correlation.
        
        Allows linking logs to specific distributed traces.
        """
        correlation = {}
        
        for log in logs:
            if log.trace_id:
                if log.trace_id not in correlation:
                    correlation[log.trace_id] = []
                
                correlation[log.trace_id].append({
                    'timestamp': log.timestamp.isoformat(),
                    'level': log.level,
                    'service': log.service,
                    'message': log.message
                })
        
        return correlation
    
    def get_log_context_for_rca(
        self,
        anomaly_timestamp: datetime,
        window_minutes: int = 5,
        service_name: Optional[str] = None
    ) -> Dict:
        """
        Get log context for RCA report.
        
        Args:
            anomaly_timestamp: When anomaly was detected
            window_minutes: How many minutes before anomaly to analyze
            service_name: Optional service filter
        
        Returns:
            Dict with log insights for RCA
        """
        start_time = anomaly_timestamp - timedelta(minutes=window_minutes)
        end_time = anomaly_timestamp
        
        summary = self.analyze_time_window(start_time, end_time, service_name)
        
        # Format for RCA integration
        rca_context = {
            'has_log_data': summary.total_logs > 0,
            'error_rate_from_logs': summary.error_logs / max(1, summary.total_logs),
            'warning_rate': summary.warning_logs / max(1, summary.total_logs),
            'services_with_errors': summary.services_involved,
            'error_patterns': summary.error_patterns,
            'critical_errors': summary.critical_errors[:3],  # Top 3 for RCA
            'trace_correlation': summary.trace_correlation
        }
        
        return rca_context
    
    def extract_error_details(self, error_message: str) -> Dict:
        """
        Extract structured information from error message.
        
        Looks for:
        - Exception types
        - HTTP status codes
        - File/line numbers
        - Resource names
        """
        details = {
            'exception_type': None,
            'http_status': None,
            'resource': None,
            'original_message': error_message
        }
        
        # Extract exception type (e.g., "ValueError", "ConnectionError")
        exception_match = re.search(r'(\w+Error|\w+Exception)', error_message)
        if exception_match:
            details['exception_type'] = exception_match.group(1)
        
        # Extract HTTP status (e.g., "404", "500")
        http_match = re.search(r'\b([45]\d{2})\b', error_message)
        if http_match:
            details['http_status'] = int(http_match.group(1))
        
        # Extract resource names (e.g., database names, service names)
        resource_match = re.search(r'(database|service|pod|deployment|endpoint)[\s:=]+([a-zA-Z0-9-]+)', error_message, re.IGNORECASE)
        if resource_match:
            details['resource'] = resource_match.group(2)
        
        return details


# Convenience function for integration
def get_log_insights(
    anomaly_timestamp: datetime,
    log_file_path: Optional[str] = None,
    service_name: Optional[str] = None
) -> Dict:
    """
    Quick function to get log insights for RCA.
    
    Usage in RCA:
        log_insights = get_log_insights(
            anomaly_timestamp=datetime.now(),
            log_file_path='/var/log/services.log',
            service_name='notification-service'
        )
        
        if log_insights['has_log_data']:
            print(f"Error rate: {log_insights['error_rate_from_logs']}")
            print(f"Critical errors: {log_insights['critical_errors']}")
    """
    analyzer = LogAnalyzer(log_source='file', log_file_path=log_file_path)
    return analyzer.get_log_context_for_rca(anomaly_timestamp, service_name=service_name)


if __name__ == "__main__":
    print("Log Analyzer Module")
    print("Import this module to use LogAnalyzer class")
    print()
    print("Example usage:")
    print("  from log_analyzer import get_log_insights")
    print("  insights = get_log_insights(datetime.now(), log_file_path='service.log')")
