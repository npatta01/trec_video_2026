"""Write the private, human-reviewed subset of the Qwen frame/audio audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REVIEW_NOTES = {
    "E3hGqRv4_K8": ("present", "visual+audio_transcript", "Coca-Cola is visible on the can and named in the transcript; the baseline answer is a punctuation/hyphen variant."),
    "G4jfirTeNCY": ("present", "visual+audio_transcript", "The football sequence is visible and the transcript says two for England; the treatment is the word-form of the baseline number."),
    "GH7kcBkraNg": ("present", "visual", "The contact sheet labels smartphone, mini smartphone, flip phone, tablet, and notebook: five items."),
    "GIdtda2RH_g": ("present", "visual", "The multi-level highway is visible across the selected aerial frames; the treatment is the word-form of the baseline number."),
    "HD2_qedLUTE": ("present", "visual+audio_transcript", "The first two spoken sentences are visible as captions and reproduced by the transcript; the baseline is a shortened punctuation variant."),
    "HVh_en2b0-U": ("present", "visual", "The background and single French-door opening are visible; the treatment is the word-form of the baseline number."),
    "HhyW9mStn7Y": ("present", "visual", "The sports clip shows the single scoring play; the treatment is the word-form of the baseline number."),
    "HpAFuQ51Fmg": ("present", "visual+audio_transcript", "The girl is visibly inside a garbage can and the transcript names it; trash can is a semantic equivalent."),
    "J1CS47pBX_M": ("present", "audio_transcript", "The transcript gives the opening sentence exactly; the selected frames show the iPad but also a misleading Break out of your shell overlay."),
    "J5cGZ_EA8R8": ("uncertain", "visual_temporal", "The sampled frames show the obstacle sequence but do not conclusively expose the flip count; the transcript has no useful evidence."),
    "JYgbRMKoZwU": ("present", "visual_temporal", "The pianist, keyboard, and microphone are visible across the sequence; the count is temporal and not stated in the transcript."),
    "JZzkwsnRHio": ("present", "visual", "The selected frames show one glass of water; the treatment is the word-form of the baseline number."),
    "KanVJWDHduE": ("uncertain", "visual_temporal", "The fishing sequence is visible, but the uniform contact sheet is not sufficient to verify the total fish count confidently."),
    "L75wb-sK0Ws": ("uncertain", "visual_temporal", "The selected frames show the lawnmowers and road but do not make the gate count unambiguous; audio is unrelated."),
    "LKiaCf1QZFw": ("present", "audio_transcript", "The transcript contains the end utterance Oh my God, she's insane; the visual frames do not reveal that answer."),
    "LSz_xBDBPNc": ("present", "audio_transcript", "The transcript explicitly repeats put it down after the calcium glows; frames show the glowing piece but not the full instruction."),
    "Loyn1vxCI6E": ("present", "visual+audio_transcript", "On-screen text and the transcript describe the girl giving one bottle of water; the treatment is a word-form variant."),
    "N5-nI2V_GS0": ("uncertain", "visual_temporal", "The selected frames show several ping-pong phases but not a reliable complete hit count; the transcript is unrelated."),
    "NbhaZCj6cVE": ("present", "visual+audio_transcript", "The cartoon figures visibly represent 7, 8, and 9 and the transcript says Seven Eight Nine; the baseline uses digits."),
    "NsF80sL5wy4": ("present", "visual+audio_transcript", "The ghost figures are visible and the transcript says Happy Halloween; the frames alone cannot establish what they say."),
    "OVAy9Y-zh1E": ("present", "visual_temporal+audio_transcript", "The cricket sequence contains one diving catch and the transcript describes the save; the treatment is the word-form of the baseline number."),
    "Od_pstVp5Sk": ("present", "visual", "One light fixture is visible by the doorway in the selected frames; the treatment is the word-form of the baseline number."),
    "PAmo_TwQvOI": ("uncertain", "visual_temporal", "The snowy outdoor sequence is visible, but a street-lamp count is not unambiguous in the selected frames and audio is unrelated."),
    "PZWAlJSL7Kc": ("present", "visual", "The white compound is visibly applied to the wood; glue and wood glue are semantic equivalents."),
    "Pne0cC8Zn70": ("present", "visual", "The table scene exposes the two-chair setup; the treatment is the word-form of the baseline number."),
    "Qz39qVt6NGA": ("present", "visual+audio_transcript", "The captions and demonstration explicitly show shoulders back and unhunched; the baseline contradicts the displayed correction only by one word."),
    "RWxbEgaAcSE": ("present", "visual_temporal", "The selected sequence shows the rider crash, get up, and leave; the baseline is a shortened answer."),
    "FiXXyP5ZRxA": ("present", "visual", "The animation visibly demonstrates drawing a doll; the verbose treatment is semantically correct but fails strict exact match."),
    "ITL429RMscI": ("present", "visual+audio_conflict", "Three stars are visibly shown on the uniform and support Major general; unrelated audio caused the treatment to answer Brigadier. This is a real Whisper regression."),
    "POH-lLoKOLs": ("present", "visual", "One clipboard is visible; 1 is semantically identical to One, so this is an exact-match formatting regression."),
    "RwDFP3Vpwoo": ("present", "visual", "The object under the hand is visibly a computer mouse; the treatment is a semantically more specific answer than the baseline."),
    "B24dV_uori4": ("present", "visual+audio_transcript", "The transcript states shampoo twice and explains why it lathers; the selected frames show the shampooing sequence."),
    "BDKkVYmWqdM": ("present", "visual+audio_transcript", "The frames show the bicyclist crossing the tennis net and the transcript says the same; both model answers are semantic variants of the reference."),
    "BSVHVuVnlGw": ("present", "visual+audio_transcript", "The frames show an electric toothbrush demonstration and the transcript explains the topic; both answers capture the subject at different specificity."),
    "Bvv6lMUTlEw": ("present", "visual+audio_transcript", "The title and frames show Ellie's birthday with cakes, and the transcript says it is her birthday; the evidence is sufficient."),
    "C4j3Pp79KSg": ("present", "visual", "The woman reaches for and holds the man's hand at the corner; the visual evidence is clear, while the translated audio is not useful for this question."),
    "C7ADL93Zg4Q": ("present", "visual_temporal", "The woman kicks the Pac-Man-like character; the key event is visible even though the audio is unrelated."),
    "D_HTq7IiSPo": ("present", "visual+audio_transcript", "The frames caption the British-accent joke and the transcript includes feeling left out; both evidence channels support the event."),
    "E0diSOsIyKg": ("present", "audio_transcript+visual", "The transcript explicitly identifies deep-fried garlic as the last ingredient; the frames show the finished dish but not the spoken ingredient list."),
    "GSo0ep3dlh4": ("present", "visual+audio_transcript", "The frames show the shirt being turned into a cowl-neck/vest and the transcript describes each step; the answers differ in specificity."),
    "HhJRo_1C5iY": ("present", "visual+audio_transcript", "The dog paws at the woman while captions say Gimme treats and the transcript mentions treats; the answer is directly supported."),
    "I4JX5m-gJxM": ("present", "visual+audio_transcript", "The frames display to wakesurf and Four steps, and the transcript repeats the opening text; the question is answerable from both."),
    "I9QFj6lz15w": ("present", "visual+audio_transcript", "The frames show toothpicks suspending an avocado seed in water and the transcript explains that purpose."),
    "IgHdataMrsI": ("present", "visual+audio_transcript", "The doctor demonstrates stimulation on a stuffed bear; captions and transcript describe touching the belly and stimulating senses."),
    "IluSoQAf7iQ": ("present", "visual+audio_transcript", "The captions show write down 5 objections and the transcript explains creating an irresistible offer; the evidence is explicit."),
    "IsuDOLuMBR0": ("uncertain", "audio_transcript+visual", "The wedding-party visuals are clear, but the exact sound at the end is not reliably identified by the translated transcript alone."),
    "Iuf0gTj4S0M": ("present", "visual_temporal", "The selected frames show children jumping into the pool after answering; the audio is a repeated question rather than the answer."),
    "I_TaWo2AMDA": ("uncertain", "visual_temporal", "The frames show push-up training but uniform sampling does not establish the completed count confidently; the transcript is generic."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    rows = {row["video_id"]: row for row in audit["rows"]}
    missing = sorted(set(REVIEW_NOTES) - set(rows))
    if missing:
        raise SystemExit(f"Review IDs missing from audit: {missing}")
    reviewed = []
    for video_id, (status, basis, note) in REVIEW_NOTES.items():
        row = rows[video_id]
        reviewed.append(
            {
                "video_id": video_id,
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "frames_only_answer": row["frames_only_answer"],
                "frames_whisper_answer": row["frames_whisper_answer"],
                "wrong_conditions": row["wrong_conditions"],
                "transition": (
                    "both_correct" if not row["wrong_conditions"] else
                    "frames_only_only_wrong" if row["wrong_conditions"] == ["qwen35-4b-12f-frames-only"] else
                    "whisper_only_wrong" if row["wrong_conditions"] == ["qwen35-4b-12f-whisper-large-v3"] else
                    "both_wrong"
                ),
                "manual_review": {
                    "status": status,
                    "evidence_basis": basis,
                    "note": note,
                },
                "frame_evidence": row["frame_evidence"],
                "audio_evidence": row["audio_evidence"],
                "whisper_evidence": row["whisper_evidence"],
            }
        )
    payload = {
        "schema_version": 1,
        "run_id": audit["run_id"],
        "source_audit": str(args.audit),
        "review_scope": {
            "all_wrong_rows_indexed": audit["wrong_row_count"],
            "manually_reviewed_rows": len(reviewed),
            "rows_remaining_for_manual_review": audit["wrong_row_count"] - len(reviewed),
            "selection": "all 31 exact-match transition rows plus a stratified sample of both-wrong rows covering audio, visual, temporal, captions, and explanation questions",
            "visual_review": "12-frame contact sheets viewed with their selected PTS labels",
            "audio_review": "timestamped Whisper transcript inspected; raw audio playback was not available in this runtime",
        },
        "summary": {
            "status_counts": dict(Counter(row["manual_review"]["status"] for row in reviewed)),
            "transition_counts": dict(Counter(row["transition"] for row in reviewed)),
            "evidence_basis_counts": dict(Counter(row["manual_review"]["evidence_basis"] for row in reviewed)),
            "clear_real_whisper_regressions": ["ITL429RMscI"],
            "strict_metric_or_semantic_variant_regressions": ["FiXXyP5ZRxA", "POH-lLoKOLs", "RwDFP3Vpwoo"],
        },
        "rows": reviewed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(reviewed), "summary": payload["summary"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
