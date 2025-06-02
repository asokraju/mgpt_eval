"""
Comprehensive logging utilities for the binary classifier pipeline.
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import functools
import time
import traceback

from models.config_models import LoggingConfig


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = traceback.format_exception(*record.exc_info)
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class PipelineLogger:
    """Enhanced logger for the binary classifier pipeline."""
    
    def __init__(self, name: str, config: LoggingConfig):
        """
        Initialize the pipeline logger.
        
        Args:
            name: Logger name
            config: Logging configuration
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up the logger with appropriate handlers and formatters."""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Set logger level
        self.logger.setLevel(getattr(logging, self.config.level))
        
        # Create formatters
        detailed_formatter = logging.Formatter(self.config.format)
        colored_formatter = ColoredFormatter(self.config.format)
        json_formatter = JSONFormatter()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.console_level))
        console_handler.setFormatter(colored_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.file:
            # Parse max file size
            max_bytes = self._parse_size(self.config.max_file_size)
            
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.file,
                maxBytes=max_bytes,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, self.config.level))
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
            
            # JSON log file for structured logs
            json_file = Path(self.config.file).with_suffix('.json')
            json_handler = logging.handlers.RotatingFileHandler(
                json_file,
                maxBytes=max_bytes,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(getattr(logging, self.config.level))
            json_handler.setFormatter(json_formatter)
            self.logger.addHandler(json_handler)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes."""
        size_str = size_str.upper()
        
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def log_stage_start(self, stage: str, **kwargs):
        """Log the start of a pipeline stage."""
        self.logger.info(f"Starting stage: {stage}", extra={
            'stage': stage,
            'event': 'stage_start',
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
    
    def log_stage_end(self, stage: str, duration: float, **kwargs):
        """Log the end of a pipeline stage."""
        self.logger.info(f"Completed stage: {stage} in {duration:.2f}s", extra={
            'stage': stage,
            'event': 'stage_end',
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
    
    def log_stage_error(self, stage: str, error: Exception, **kwargs):
        """Log an error in a pipeline stage."""
        self.logger.error(f"Error in stage {stage}: {str(error)}", extra={
            'stage': stage,
            'event': 'stage_error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }, exc_info=True)
    
    def log_progress(self, stage: str, current: int, total: int, **kwargs):
        """Log progress for long-running operations."""
        progress = (current / total) * 100 if total > 0 else 0
        self.logger.info(f"Progress [{stage}]: {current}/{total} ({progress:.1f}%)", extra={
            'stage': stage,
            'event': 'progress',
            'current': current,
            'total': total,
            'progress_percent': progress,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
    
    def log_metrics(self, stage: str, metrics: Dict[str, Any], **kwargs):
        """Log performance metrics."""
        self.logger.info(f"Metrics [{stage}]: {metrics}", extra={
            'stage': stage,
            'event': 'metrics',
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
    
    def log_config(self, config: Dict[str, Any], **kwargs):
        """Log configuration information."""
        self.logger.info("Configuration loaded", extra={
            'event': 'config_loaded',
            'config': config,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
    
    def log_checkpoint(self, stage: str, checkpoint_path: str, **kwargs):
        """Log checkpoint creation."""
        self.logger.info(f"Checkpoint saved [{stage}]: {checkpoint_path}", extra={
            'stage': stage,
            'event': 'checkpoint_saved',
            'checkpoint_path': checkpoint_path,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
    
    def log_api_call(self, endpoint: str, method: str, status_code: int, 
                    duration: float, **kwargs):
        """Log API call information."""
        level = logging.INFO if status_code < 400 else logging.WARNING
        self.logger.log(level, f"API {method} {endpoint} - {status_code} ({duration:.3f}s)", extra={
            'event': 'api_call',
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
    
    # Delegate standard logging methods
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        self.logger.exception(message, extra=kwargs)


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> PipelineLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        config: Logging configuration (uses defaults if not provided)
        
    Returns:
        Configured PipelineLogger instance
    """
    if config is None:
        config = LoggingConfig()
    
    return PipelineLogger(name, config)


def log_execution_time(logger: PipelineLogger, stage: str):
    """
    Decorator to log execution time of functions.
    
    Args:
        logger: Logger instance
        stage: Stage name for logging
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.log_stage_start(stage, function=func.__name__)
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_stage_end(stage, duration, function=func.__name__)
                return result
            except Exception as e:
                logger.log_stage_error(stage, e, function=func.__name__)
                raise
        
        return wrapper
    return decorator


def log_progress_iterator(logger: PipelineLogger, stage: str, iterable, 
                         total: Optional[int] = None):
    """
    Wrap an iterable to log progress.
    
    Args:
        logger: Logger instance
        stage: Stage name for logging
        iterable: Iterable to wrap
        total: Total items (if known)
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    
    for i, item in enumerate(iterable):
        if total and i % max(1, total // 20) == 0:  # Log every 5%
            logger.log_progress(stage, i, total)
        yield item
    
    if total:
        logger.log_progress(stage, total, total)


class LogContext:
    """Context manager for structured logging with automatic timing."""
    
    def __init__(self, logger: PipelineLogger, stage: str, **extra_fields):
        """
        Initialize log context.
        
        Args:
            logger: Logger instance
            stage: Stage name
            **extra_fields: Additional fields to include in logs
        """
        self.logger = logger
        self.stage = stage
        self.extra_fields = extra_fields
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log_stage_start(self.stage, **self.extra_fields)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log_stage_end(self.stage, duration, **self.extra_fields)
        else:
            self.logger.log_stage_error(self.stage, exc_val, 
                                       duration=duration, **self.extra_fields)
        
        return False  # Don't suppress exceptions
    
    def log_progress(self, current: int, total: int, **kwargs):
        """Log progress within the context."""
        self.logger.log_progress(self.stage, current, total, 
                               **{**self.extra_fields, **kwargs})
    
    def log_metrics(self, metrics: Dict[str, Any], **kwargs):
        """Log metrics within the context."""
        self.logger.log_metrics(self.stage, metrics, 
                              **{**self.extra_fields, **kwargs})