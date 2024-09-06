import time
import logging

# Configure logging
logging.basicConfig(filename='long_term_job.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def long_term_job():
    try:
        while True:
            # Perform the main task
            logging.info("Job iteration started.")
            
            # Example task: simple operation
            result = sum(i for i in range(1000))
            
            logging.info(f"Job iteration completed. Result: {result}")
            
            # Sleep for two days (2 days * 24 hours/day * 60 minutes/hour * 60 seconds/minute)
            time.sleep(2 * 24 * 60 * 60)  # Sleep for 2 days
    except KeyboardInterrupt:
        logging.info("Job interrupted by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        logging.info("Job terminated.")

if __name__ == "__main__":
    long_term_job()
