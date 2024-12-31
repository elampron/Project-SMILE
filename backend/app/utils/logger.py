import logging
import sys
from typing import Optional

def get_logger(name: str, file_name: Optional[str] = 'logs/debug.log') -> logging.Logger:
    """
    Create and configure a logger with the given name and file name for logging.
    
    Args:
        name (str): The name of the logger.
        file_name (Optional[str]): The file name for logging. Defaults to 'logs/debug.log'.
        
    Returns:
        logging.Logger: Configured logger instance.
        
    Logs:
        DEBUG: Logger creation and configuration details.
        ERROR: If there is an error during logger setup.
    """
    try:
        # Create a logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        import os
        
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(file_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            logger.debug(f"Created log directory: {log_dir}")
        
        # Create file handler for debug messages with UTF-8 encoding
        file_handler = logging.FileHandler(file_name, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler with higher log level using sys.stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.debug(f"Logger '{name}' created and configured with file handler '{file_name}'")
        
        return logger
    
    except Exception as e:
        logging.error(f"Error creating logger '{name}': {str(e)}")
        raise

logger = get_logger('Smiles')