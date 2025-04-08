#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from typing import Annotated, Literal
from abc import ABC
import httpx
import ormsgpack
from pydantic import BaseModel, conint
from rag.utils import num_tokens_from_string
import json


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "mp3"
    mp3_bitrate: Literal[64, 128, 192] = 128
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # Balance mode will reduce latency to 300ms, but may decrease stability
    latency: Literal["normal", "balanced"] = "normal"


class Base(ABC):
    def __init__(self, key, model_name, base_url):
        pass

    def transcription(self, audio):
        pass


class FishAudioTTS(Base):
    def __init__(self, key, model_name, base_url="https://api.fish.audio/v1/tts"):
        if not base_url:
            base_url = "https://api.fish.audio/v1/tts"
        key = json.loads(key)
        self.headers = {
            "api-key": key.get("fish_audio_ak"),
            "content-type": "application/msgpack",
        }
        self.ref_id = key.get("fish_audio_refid")
        self.base_url = base_url

    def transcription(self, text):
        from http import HTTPStatus

        request = request = ServeTTSRequest(text=text, reference_id=self.ref_id)

        with httpx.Client() as client:
            try:
                with client.stream(
                    method="POST",
                    url=self.base_url,
                    content=ormsgpack.packb(
                        request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC
                    ),
                    headers=self.headers,
                    timeout=None,
                ) as response:
                    if response.status_code == HTTPStatus.OK:
                        for chunk in response.iter_bytes():
                            yield chunk
                    else:
                        response.raise_for_status()

                yield num_tokens_from_string(text)

            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"**ERROR**: {e}")
