import sys
import os
import logging

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('deepsleepnet.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def download_data():
    logger.info("Download started")
    from scripts.download_data import main as download_main
    download_main()
    logger.info("Download completed")


def preprocess_data():
    logger.info("Preprocessing started")
    from scripts.run_preprocessing import main as preprocess_main
    preprocess_main()
    logger.info("Preprocessing completed")


def analyze_data():
    logger.info("Analysis started")
    from scripts.analyze_data import main as analyze_main
    analyze_main()
    logger.info("Analysis completed")


def train_model():
    logger.info("Training started")
    from scripts.train import main as train_main
    train_main()
    logger.info("Training completed")


def evaluate_model():
    logger.info("Evaluation started")
    from scripts.evaluate import main as evaluate_main
    evaluate_main()
    logger.info("Evaluation completed")


def visualize_results():
    logger.info("Visualization started")
    from scripts.visualize_metrics import main as visualize_main
    visualize_main()
    logger.info("Visualization completed")


def main():
    logger.info("Pipeline started")
    logger.info(f"Root: {ROOT}")

    raw_path = os.path.join(ROOT, 'data', 'raw')
    proc_path = os.path.join(ROOT, 'data', 'preprocessed')

    if not os.path.exists(raw_path) or not os.listdir(raw_path):
        download_data()
    else:
        logger.info("Raw data exists, skipping download")

    if not os.path.exists(proc_path) or not os.listdir(proc_path):
        preprocess_data()
    else:
        logger.info("Preprocessed data exists, skipping preprocessing")

    try:
        analyze_data()
    except Exception as e:
        logger.warning(f"Analysis failed: {e}")

    train_model()
    evaluate_model()
    visualize_results()

    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
