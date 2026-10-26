import logging
import sys

def setup_logging(name: str, config : dict, level = None, disabled = None) -> logging.Logger:
    """
    Setup logging configuration and return a logger instance. Logger should output to console only.

    Args:
        name (str): The name of the logger.
        config (dict): The logging configuration dictionary.
        level (int, optional): The logging level. Levels are 10 (DEBUG), 20 (INFO), 30 (WARNING), 40 (ERROR), 50 (CRITICAL). If not provided, it is read from the config.
        disabled (bool, optional): Whether the logger is disabled. If not provided, it is read from the config.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # Check if disabled argument is provided, else read from config
    if disabled is None:
        disabled = config['logging']['disabled']
    
    #disable logger if disabled = True
    logger.disabled = disabled

    if disabled == True:
        return logger

    #Check if level argument is provided, else read from config
    if level == None:
        level = logging.getLevelName(config['logging']['level'])

    logger.setLevel(level)

    # Create a log message formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a StreamHandler that outputs log messages to the console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(level)
    logger.addHandler(ch)

    # Create a StreamHandler that outputs log messages to output file, if it exists
    output_file = config['logging'].get('log_file') # Returns None if the log_file is missing
    if output_file:
        oh = logging.FileHandler(output_file)
        oh.setFormatter(formatter)
        oh.setLevel(level)
        logger.addHandler(oh)
    
    return logger

#TODO: Add logging to output file as well as terminal and prevent duplicates