import subprocess
import os


def load_chroma_db(s3_path, persist_directory):
    if not os.path.exists(persist_directory):
        subprocess.run(["mc", "cp", "-r", s3_path, persist_directory])
