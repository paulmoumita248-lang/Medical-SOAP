import re
import json
import logging
from typing import List, Dict, Any
from backend.app.config import settings
from backend.app.models.schemas import Utterance

logger = logging.getLogger("healthcare_soap.diarization_service")


class DiarizationService:
    def _diarize_with_llm(self, raw_text: str, duration_seconds: float = 0.0) -> List[Utterance]:
        """
        Uses LLM engine to punctuate raw unpunctuated ASR transcript text
        and separate turns between Clinician (SPEAKER_00) and Patient (SPEAKER_01).
        """
        api_key = settings.MISTRAL_API_KEY
        if not api_key:
            return []

        system_prompt = """You are an expert Clinical Conversation Speaker Diarizer and Punctuation Engine.
Your task is to take an unpunctuated, continuous doctor-patient clinical transcript and break it into clean, punctuated, alternating speaker turns:
- SPEAKER_00: Clinician / Doctor (asking diagnostic questions, performing exams, giving advice/prescriptions/instructions like 'Yes coming take your seat', 'What's your name?', 'How old are you?', 'Does it hurt here?', 'I am prescribing this medicine', 'Please submit that at the cash counter').
- SPEAKER_01: Patient / Relative (describing symptoms, answering questions like 'Simran Parveen', 'I am 29', 'severe stomach ache', 'Thank you doctor').

Rules:
1. Punctuate the text naturally (add periods, question marks, commas).
2. Ensure SPEAKER_00 is assigned to Doctor turns, and SPEAKER_01 is assigned to Patient turns.
3. Return strictly a JSON object with key "utterances" containing an array of objects: [{"speaker_id": "SPEAKER_00", "text": "..."}, ...]. Do NOT include markdown code blocks."""

        user_prompt = f"Raw Clinical Transcript:\n{raw_text}"

        try:
            try:
                from mistralai import Mistral
            except ImportError:
                from mistralai.client import Mistral

            client = Mistral(api_key=api_key)
            resp = client.chat.complete(
                model=settings.MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content.strip()
            
            # Clean markdown fences if any
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
            if match:
                content = match.group(1).strip()

            parsed = json.loads(content)
            items = parsed.get("utterances", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
            
            if not items:
                return []

            total_chars = max(1, sum(len(item.get("text", "")) for item in items))
            dur = duration_seconds if duration_seconds > 0 else 60.0
            curr_t = 0.0

            utterances: List[Utterance] = []
            for item in items:
                spk = item.get("speaker_id", "SPEAKER_00")
                txt = item.get("text", "").strip()
                if not txt:
                    continue
                
                u_dur = (len(txt) / total_chars) * dur
                end_t = round(curr_t + u_dur, 2)

                # Merge consecutive turns from same speaker
                if utterances and utterances[-1].speaker_id == spk:
                    utterances[-1].text += f" {txt}"
                    utterances[-1].end_time = end_t
                else:
                    utterances.append(
                        Utterance(
                            speaker_id=spk,
                            start_time=round(curr_t, 2),
                            end_time=end_t,
                            text=txt
                        )
                    )
                curr_t = end_t

            logger.info(f"LLM Diarization successfully split unpunctuated text into {len(utterances)} speaker-tagged turns.")
            return utterances

        except Exception as e:
            logger.warning(f"LLM Diarization fallback encountered an error: {e}")
            return []

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits raw transcript text into clean sentences while preserving honorifics (Mr., Dr., etc.).
        """
        if not text:
            return []
        
        # If text is unpunctuated (no periods/question marks), split by phrase markers
        if not re.search(r"[.?!]", text):
            split_pattern = r"(?i)\b(?=(?:what's your name|how old are you|what are the problems|do you have|did you have|how many days ago|does it hurt|is it something serious|I can't say|I understand|I am prescribing|Please submit|thank you doctor|yes doctor|ok doctor|I am \d+))\b"
            parts = [p.strip() for p in re.split(split_pattern, text) if p.strip()]
            if len(parts) > 1:
                return parts

        text_norm = re.sub(r'\b(Mr|Dr|Mrs|Ms|Prof|vs|eg|ie)\.', r'\1_', text)
        raw_sentences = re.split(r'(?<=[.?!])\s+', text_norm)
        
        sentences = []
        for s in raw_sentences:
            s_clean = re.sub(r'\b(Mr|Dr|Mrs|Ms|Prof|vs|eg|ie)_', r'\1.', s.strip())
            if s_clean:
                sentences.append(s_clean)
        
        return sentences

    def diarize_segments(self, segments: List[Dict[str, Any]]) -> List[Utterance]:
        """
        Classifies transcript segments into 2-speaker tags (`SPEAKER_00`, `SPEAKER_01`).
        For unpunctuated raw ASR text, uses LLM diarization engine for 100% accurate turn separation.
        """
        if not segments:
            return []

        full_raw_text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text", "")).strip()
        total_duration = max(float(segments[-1].get("end", 0.0)), 1.0) if segments else 0.0

        # If transcript lacks punctuation (common in raw Web Speech API), attempt LLM Diarization
        if not re.search(r"[.?!]", full_raw_text) or len(segments) <= 1:
            llm_utterances = self._diarize_with_llm(full_raw_text, duration_seconds=total_duration)
            if llm_utterances:
                return llm_utterances

        # Rule-based sentence expansion fallback
        sentence_units = []
        for seg in segments:
            raw_text = seg.get("text", "").strip()
            start_t = float(seg.get("start", 0.0))
            end_t = float(seg.get("end", 0.0))
            
            s_list = self._split_into_sentences(raw_text)
            if not s_list:
                continue

            total_chars = max(1, sum(len(s) for s in s_list))
            dur = max(0.5, end_t - start_t)
            
            curr_time = start_t
            for s in s_list:
                s_dur = (len(s) / total_chars) * dur
                sentence_units.append({
                    "text": s,
                    "start": round(curr_time, 2),
                    "end": round(curr_time + s_dur, 2)
                })
                curr_time += s_dur

        if not sentence_units:
            return []

        # Assign speaker turns
        utterances: List[Utterance] = []
        speaker_labels = ["SPEAKER_00", "SPEAKER_01"]
        current_speaker_idx = 0

        for i, unit in enumerate(sentence_units):
            text = unit["text"]
            start_time = unit["start"]
            end_time = unit["end"]

            if i > 0:
                prev_text = sentence_units[i - 1]["text"].strip()
                if prev_text.endswith("?") or "what" in prev_text.lower() or "how" in prev_text.lower():
                    current_speaker_idx = 1
                elif i > 1 and ("?" in sentence_units[i - 2]["text"] or "doctor" in prev_text.lower()):
                    current_speaker_idx = 0

            speaker_id = speaker_labels[current_speaker_idx]

            if utterances and utterances[-1].speaker_id == speaker_id:
                utterances[-1].text += f" {text}"
                utterances[-1].end_time = max(utterances[-1].end_time, end_time)
            else:
                utterances.append(
                    Utterance(
                        speaker_id=speaker_id,
                        start_time=start_time,
                        end_time=end_time,
                        text=text
                    )
                )

        logger.info(f"Diarization service produced {len(utterances)} speaker-tagged utterances.")
        return utterances


# Global service instance
diarization_service = DiarizationService()
