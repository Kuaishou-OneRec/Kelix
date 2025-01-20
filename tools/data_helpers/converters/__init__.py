from .converter import (
    ConverterBase,
    EmptyConverter
)
from .the_cauldron_converter import TheCaulDronConverter
from .kwai_video import KwaiVideoCaptionConverter
from .dense_fusion_converter import DenseFusionConverter
from .llava_cc3m_converter import LlavaCC3MPretrainConverter
from .doc_matrix_converter import DocmatrixConverter
from .blobstore_downloader_converter import BlobstoreDownloaderConverter
from .wds_to_parquet_converter import WDSToParquetConverter
from .vlm_sft_converter import VlmSftImageConverter, VlmSftTextConverter
from .resample_converter import ResampleConverter


def create_converter(cfg) -> ConverterBase:
    return eval(cfg.class_name)(**cfg.kwargs)