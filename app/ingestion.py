"""Unstructure data ingestion utilities."""

import fsspec
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from config import settings

# TODO: Move this to a separate module?


def read_docs(
    input_dir: str,
    **kwargs,
) -> list[Document]:
    """Reads all documents in the specified directory recursively. Supports local
    file system or remote S3/GCS bucket.

    Reference: https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader.html

    Args:
        input_dir: The directory to locate the documents.
        **kwargs: Additional keyword args passed to SimpleDirectoryReader.

    Returns:
        Llama-index document nodes.
    """
    fs: fsspec.AbstractFileSystem | None = kwargs.get("fs")
    if fs is None and input_dir.startswith("s3://"):
        fs = S3FileSystem(
            key=settings.minio_access_key,
            secret=settings.minio_secret_key,
            endpoint_url=settings.minio_endpoint,
            use_ssl=False,  # FIXME: should default to True for deployment
        )
        input_dir = input_dir.removeprefix("s3://")

    elif fs is None and input_dir.startswith("gs://"):
        # TODO: figure out the best default auth for gcp
        #       https://gcsfs.readthedocs.io/en/latest/#credentials
        fs = GCSFileSystem(
            project=settings.gcp_project,
        )
        input_dir = input_dir.removeprefix("gs://")

    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=kwargs.pop("recursive", True),
        fs=fs,
        **kwargs,
    )
    docs = reader.load_data()

    return docs
