import json
import uuid
import base64
from typing import Dict, List, Sequence, Optional
from .converter import (
    ConverterBase
)

class WDSToParquetConverter(ConverterBase):

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        metadata = json.loads(src['json'])
        source_name = self.get_source_name(metadata)
        images = dict()
        for key in src.keys():
            if key.endswith("jpg") or key.endswith("png"):
                images[key] = base64.b64encode(src[key]).decode('ascii')
        videos = []
        messages = metadata.get("messages", None) or metadata.get("message", None)
        segments = metadata.get("segments", None)
        videos.extend(self.parse_videos_from_messages(messages))
        videos.extend(self.parse_videos_from_segments(segments))
        result = {
            "images": json.dumps(images),
            "videos": json.dumps(videos),
            "source": source_name,
            "messages": json.dumps(messages),
            "segments": json.dumps(segments),
            "metadata": None,
            "uuid": str(uuid.uuid1()),
        }
        return result
    
    def parse_videos_from_messages(self, messages):
        result = []
        if messages is None:
            return result
        for msg in messages:
            if isinstance(msg['content'], List):
                for c in msg['content']:
                    if isinstance(c, Dict) and c['type'] == 'video':
                        result.append(c['video'])
        return result
    
    def parse_videos_from_segments(self, segments):
        result = []
        if segments is None:
            return result
        for seg in segments:
            if isinstance(seg, Dict) and seg['type'] == 'video':
                result.append(seg['video'])
        return result


    def get_source_name(self, metadata):
        try:
            source_name = metadata['source']
            # WARN: ugly code, for dirty dataset.
            if source_name.startswith("PDFA"):
                source_name = "PDFA"
            elif source_name.startswith("/llm_reco_ssd/luoxinchen/dataset/"):
                source_name = source_name.split("/")[4]
        except:
            source_name = "None"
        return source_name
