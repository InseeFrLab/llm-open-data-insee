"""
Create Label Studio tasks for Insee Contact data.
For now, we keep only the first part of exchanges:
question asked by the user and first response.
"""


# Configuration
process_args()  # Strict minimal arguments processing
logger = logging.getLogger(__name__)




if __name__ == "__main__":
    insee_contact_to_s3()
