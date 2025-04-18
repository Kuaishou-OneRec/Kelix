from .converter import (
    ConverterBase,
    EmptyConverter
)
from .the_cauldron_converter import TheCaulDronConverter
from .kwai_video import KwaiVideoCaptionConverter
from .kwai_video import KwaiWenJuanCaptionVideoConverter
from .kwai_video import i2iConverter
from .kwai_video import KwaiWenJuanCaptionFrameConverter
from .dense_fusion_converter import DenseFusionConverter
from .llava_cc3m_converter import LlavaCC3MPretrainConverter
from .doc_matrix_converter import DocmatrixConverter
from .blobstore_downloader_converter import BlobstoreDownloaderConverter
from .wds_to_parquet_converter import WDSToParquetConverter
from .vlm_sft_converter import VlmSftImageConverter, VlmSftTextConverter
from .resample_converter import ResampleConverter
from .blobstore_caption_converter import BlobstoreCaptionConverter
from .webdataset_caption_converter import WebDatasetCaptionConverter
from .clean_html_converter import CleanHtmlConverter
from .clean_links_converter import CleanLinksConverter
from .infinity_instruct_converter import InfinityInstructConverter
from .gpt4o_converter import GPT4oConverter
from .gpt4o_qa_converter import GPT4oQAConverter
from .grounding_converter import GroundingConverter
from .pubtabnet_converter import PubTabNetConverter
from .fintabnet_converter import FinTabNetConverter
from .kwai_video import KwaiVideoTitleCaptionConverter
from .kwai_video import KwaiVideoClickAfterShowConverter
from .kwai_video import KwaiVideoCategoryConverter
from .kwai_video import KwaiVideoClickAfterShow10Converter
from .kwai_video import KwaiVideoShuffleConverter
from .web_comment import WebCommentConverter
from .OpenImages_Caption import OpenImagesCaptionConverter
from .conversation_Caption import ConversationCaptionConverter

def create_converter(cfg) -> ConverterBase:
    return eval(cfg.class_name)(**cfg.kwargs)