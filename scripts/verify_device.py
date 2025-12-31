import torch
import logging
import sys

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def check_device():
    logging.info("Checking device availability...")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"CUDA current device: {torch.cuda.current_device()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
    logging.info(f"MPS available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    logging.info(f"Selected device: {device}")
    
    # Test tensor creation
    try:
        x = torch.ones(1).to(device)
        logging.info(f"Successfully created tensor on {device}: {x}")
    except Exception as e:
        logging.error(f"Failed to create tensor on {device}: {e}")

if __name__ == "__main__":
    check_device()
