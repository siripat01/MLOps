from zenml import pipeline
from zenml.logger import get_logger

from pipelines.training.steps.ingest import IngestData

logger = get_logger(__name__)


@pipeline
def load_data():
    IngestData()


if __name__ == "__main__":
    load_data()
