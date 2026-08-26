import logging
from typing import List, Dict, Any
from backend.app.models.schemas import Utterance

logger = logging.getLogger("healthcare_soap.diarization_service")


class DiarizationService:
    def diarize_segments(self, segments: List[Dict[str, Any]]) -> List[Utterance]:
        """
        Classifies transcript segments into 2-speaker tags (`SPEAKER_00`, `SPEAKER_01`).
        Employs alternating conversational turn heuristics for multi-speaker dialogues.
        """
        if not segments:
            return []

        utterances: List[Utterance] = []
        current_speaker_idx = 0
        speaker_labels = ["SPEAKER_00", "SPEAKER_01"]

        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                continue

            start_time = float(seg.get("start", 0.0))
            end_time = float(seg.get("end", 0.0))

            # Alternate speaker turns when pauses > 0.8s or turn changes
            if i > 0:
                prev_end = float(segments[i - 1].get("end", 0.0))
                pause_duration = start_time - prev_end
                
                # Turn heuristic: pause > 0.8 seconds or question mark ending indicates speaker switch
                prev_text = segments[i - 1].get("text", "").strip()
                if pause_duration > 0.8 or prev_text.endswith("?") or prev_text.endswith(":"):
                    current_speaker_idx = (current_speaker_idx + 1) % len(speaker_labels)

            speaker_id = speaker_labels[current_speaker_idx]

            # Merge with preceding utterance if same speaker and short gap
            if utterances and utterances[-1].speaker_id == speaker_id and (start_time - utterances[-1].end_time < 0.5):
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
