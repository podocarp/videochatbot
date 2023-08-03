from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    pipeline,
)
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from langchain import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate
from threading import Thread
import time
import torch
import numpy as np
from scipy.io.wavfile import write
from IPython.display import Audio, Video
from Wav2Lip import inference as wav2lip
import random
import string


speaker_embeddings = {
    "BDL": "spkemb/cmu_us_bdl_arctic-wav-arctic_a0009.npy",  # male
    "CLB": "spkemb/cmu_us_clb_arctic-wav-arctic_a0144.npy",  # female
    "KSP": "spkemb/cmu_us_ksp_arctic-wav-arctic_b0087.npy",  # male
    "RMS": "spkemb/cmu_us_rms_arctic-wav-arctic_b0353.npy",  # male
    "SLT": "spkemb/cmu_us_slt_arctic-wav-arctic_a0508.npy",  # female
}


class VideoGen:
    def __init__(self):
        # Load the models and stuff
        # IMPORTANT: If you want to rerun this cell, restart the kernel! (or delete the model variable)
        # The old model is still eating VRAM, transformers may start loading into CPU RAM!
        # Should not run for more than 30s!

        llm_model_path = "/home/models/vicuna/13B-1.3"
        tokenizer = LlamaTokenizer.from_pretrained(llm_model_path)
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16,
        )
        print("Loaded LLM model.")

        tts_model_path = "/home/models/speecht5_tts/"
        gan_path = "/home/models/speecht5_hifigan/"
        self.processor = SpeechT5Processor.from_pretrained(tts_model_path)
        self.speech_model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_path)
        self.vocoder = SpeechT5HifiGan.from_pretrained(gan_path)
        print("Loaded text to speech model.")

        llm_pipeline = pipeline(
            "text-generation",
            model=llm_model,
            temperature=0,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )

        template = """The following is a friendly conversation between a human and an AI. The AI is helpful, detailed, but also concise.

        Current conversation:
        {history}
        USER: {input}
        ASSISTANT:"""
        prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        self.llmChain = ConversationChain(
            llm=llm,
            prompt=prompt,
            verbose=False,
            memory=ConversationSummaryBufferMemory(llm=llm, k=3, ai_prefix="ASSISTANT"),
        )

        self.wav2lip = wav2lip.Inference("checkpoints/wav2lip_gan.pth")

    def __textToSpeech(self, text, speaker):
        if len(text.strip()) == 0:
            return np.zeros(0).astype(np.int16)

        inputs = self.processor(text=text, return_tensors="pt")

        # limit input length
        input_ids = inputs["input_ids"]
        input_ids = input_ids[..., : self.speech_model.config.max_text_positions]
        speaker_embedding = np.load(speaker_embeddings[speaker[:3]])
        speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0)
        speech = self.speech_model.generate_speech(
            input_ids, speaker_embedding, vocoder=self.vocoder
        )
        speech = (speech.numpy() * 32767).astype(np.int16)

        return speech

    def text_to_speech(self, text: str, speaker="RMS"):
        sentences = text.split("\n")
        outputs = []
        for sentence in sentences:
            outputs.append(self.__textToSpeech(sentence, speaker))

        return np.concatenate(outputs)

    def random_filename(self):
        return "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
        )

    def llm_to_video(self, input) -> (str, str):
        filename = self.random_filename()
        output_path = f"temp/{filename}.mp4"
        wav_path = f"temp/{filename}.wav"
        text = self.llmChain.predict(input=input)
        print(text)
        speech = self.text_to_speech(text)
        write(wav_path, 16000, speech)
        self.wav2lip.generate(
            video_path="videos/kennedy1min.mp4",
            audio_path=wav_path,
            out_path=output_path,
            batch_size=32,
            is_static=False,
        )
        return output_path, filename
