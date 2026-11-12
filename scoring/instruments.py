"""Item definitions and anchors for the five instruments, used to build prompts."""

PHQ_ANCHORS = "0 = Not at all | 1 = Several days | 2 = More than half the days | 3 = Nearly every day"

PHQ9 = [
    ("Anhedonia", "Little interest or pleasure in doing things"),
    ("Depressed mood", "Feeling down, depressed or hopeless"),
    ("Sleep", "Trouble falling or staying asleep, or sleeping too much"),
    ("Fatigue", "Feeling tired or having little energy"),
    ("Appetite", "Poor appetite or overeating"),
    ("Worthlessness", "Feeling bad about yourself, or that you are a failure or have let yourself or your family down"),
    ("Concentration", "Trouble concentrating on things, such as reading the newspaper or watching television"),
    ("Psychomotor", "Moving or speaking so slowly that other people could have noticed, or the opposite: being so fidgety or restless that you have been moving around a lot more than usual"),
    ("Self-harm", "Thoughts that you would be better off dead, or of hurting yourself in some way"),
]

HAMD = [
    ("Depressed mood", 4, "0 = Absent | 1 = Feeling states indicated only on questioning | 2 = Feeling states spontaneously reported verbally | 3 = Communicates feeling states non-verbally (facial expression, posture, voice, tendency to weep) | 4 = Patient reports virtually only these feeling states in spontaneous verbal and non-verbal communication"),
    ("Feelings of guilt", 4, "0 = Absent | 1 = Self-reproach, feels he/she has let people down | 2 = Ideas of guilt or rumination over past errors or sinful deeds | 3 = Present illness is a punishment; delusions of guilt | 4 = Hears accusatory or denunciatory voices and/or experiences threatening visual hallucinations"),
    ("Suicide", 4, "0 = Absent | 1 = Feels life is not worth living | 2 = Wishes he/she were dead or any thoughts of possible death to self | 3 = Suicidal ideas or gesture | 4 = Attempts at suicide"),
    ("Insomnia: early in the night", 2, "0 = No difficulty falling asleep | 1 = Occasional difficulty falling asleep (more than 1/2 hour) | 2 = Nightly difficulty falling asleep"),
    ("Insomnia: middle of the night", 2, "0 = No difficulty | 1 = Restless and disturbed during the night | 2 = Waking during the night; any getting out of bed"),
    ("Insomnia: early hours of the morning", 2, "0 = No difficulty | 1 = Waking in early hours but goes back to sleep | 2 = Unable to fall asleep again if gets out of bed"),
    ("Work and activities", 4, "0 = No difficulty | 1 = Thoughts and feelings of incapacity, fatigue or weakness related to activities, work or hobbies | 2 = Loss of interest in activities, hobbies or work, by direct report or indirectly (listlessness, indecision, vacillation) | 3 = Decrease in actual time spent in activities or decrease in productivity | 4 = Stopped working because of present illness"),
    ("Retardation (slowness of thought and speech, impaired concentration, decreased motor activity)", 4, "0 = Normal speech and thought | 1 = Slight retardation during the interview | 2 = Obvious retardation during the interview | 3 = Interview difficult | 4 = Complete stupor"),
    ("Agitation", 4, "0 = None | 1 = Fidgetiness | 2 = Playing with hands, hair, etc. | 3 = Moving about, can't sit still | 4 = Hand wringing, nail biting, hair-pulling, biting of lips"),
    ("Anxiety: psychic", 4, "0 = No difficulty | 1 = Subjective tension and irritability | 2 = Worrying about minor matters | 3 = Apprehensive attitude apparent in face or speech | 4 = Fears expressed without questioning"),
    ("Anxiety: somatic (dry mouth, indigestion, palpitations, headaches, hyperventilation, sweating, urinary frequency)", 4, "0 = Absent | 1 = Mild | 2 = Moderate | 3 = Severe | 4 = Incapacitating"),
    ("Somatic symptoms: gastro-intestinal", 2, "0 = None | 1 = Loss of appetite but eating without encouragement; heavy feelings in abdomen | 2 = Difficulty eating without urging; requests or requires laxatives or medication for GI symptoms"),
    ("General somatic symptoms", 2, "0 = None | 1 = Heaviness in limbs, back or head; backaches, headaches, muscle aches; loss of energy and fatigability | 2 = Any clear-cut symptom"),
    ("Genital symptoms (loss of libido, menstrual disturbances)", 2, "0 = Absent | 1 = Mild | 2 = Severe"),
    ("Hypochondriasis", 4, "0 = Not present | 1 = Self-absorption (bodily) | 2 = Preoccupation with health | 3 = Frequent complaints, requests for help | 4 = Hypochondriacal delusions"),
    ("Loss of weight", 2, "0 = No weight loss | 1 = Probable weight loss associated with present illness | 2 = Definite weight loss"),
    ("Insight", 2, "0 = Acknowledges being depressed and ill | 1 = Acknowledges illness but attributes cause to bad food, climate, overwork, virus, need for rest | 2 = Denies being ill at all"),
]

MADRS = [
    ("Apparent sadness", "Despondency, gloom and despair reflected in speech, facial expression and posture", "0 = No sadness | 2 = Looks dispirited but does brighten up without difficulty | 4 = Appears sad and unhappy most of the time | 6 = Looks miserable all the time; extremely despondent"),
    ("Reported sadness", "Reports of depressed mood, regardless of whether it is reflected in appearance", "0 = Occasional sadness in keeping with the circumstances | 2 = Sad or low but brightens up without difficulty | 4 = Pervasive feelings of sadness or gloominess; mood still influenced by external circumstances | 6 = Continuous or unvarying sadness, misery or despondency"),
    ("Inner tension", "Ill-defined discomfort, edginess, inner turmoil, mental tension mounting to panic, dread or anguish", "0 = Placid; only fleeting inner tension | 2 = Occasional feelings of edginess and ill-defined discomfort | 4 = Continuous feelings of inner tension or intermittent panic which the patient can only master with some difficulty | 6 = Unrelenting dread or anguish; overwhelming panic"),
    ("Reduced sleep", "Reduced duration or depth of sleep compared with the subject's own normal pattern when well", "0 = Sleeps as usual | 2 = Slight difficulty dropping off to sleep or slightly reduced, light or fitful sleep | 4 = Sleep reduced or broken by at least 2 hours | 6 = Less than 2 or 3 hours of sleep"),
    ("Reduced appetite", "Loss of appetite compared with when well", "0 = Normal or increased appetite | 2 = Slightly reduced appetite | 4 = No appetite; food is tasteless | 6 = Needs persuasion to eat at all"),
    ("Concentration difficulties", "Difficulties in collecting one's thoughts, mounting to incapacitating lack of concentration", "0 = No difficulties in concentrating | 2 = Occasional difficulties in collecting one's thoughts | 4 = Difficulties in concentrating and sustaining thought which reduce ability to read or hold a conversation | 6 = Unable to read or converse without great difficulty"),
    ("Lassitude", "Difficulty getting started or slowness initiating and performing everyday activities", "0 = Hardly any difficulty in getting started; no sluggishness | 2 = Difficulties in starting activities | 4 = Difficulties in starting simple routine activities, which are carried out with effort | 6 = Complete lassitude; unable to do anything without help"),
    ("Inability to feel", "Subjective experience of reduced interest in the surroundings or activities that normally give pleasure", "0 = Normal interest in the surroundings and in other people | 2 = Reduced ability to enjoy usual interests | 4 = Loss of interest in the surroundings; loss of feelings for friends and acquaintances | 6 = The experience of being emotionally paralysed; inability to feel anger, grief or pleasure"),
    ("Pessimistic thoughts", "Thoughts of guilt, inferiority, self-reproach, sinfulness, remorse and ruin", "0 = No pessimistic thoughts | 2 = Fluctuating ideas of failure, self-reproach or self-depreciation | 4 = Persistent self-accusations, or definite but still rational ideas of guilt or sin; increasingly pessimistic about the future | 6 = Delusions of ruin, remorse or unredeemable sin; absurd and unshakable self-accusations"),
    ("Suicidal thoughts", "Feeling that life is not worth living, that a natural death would be welcome, suicidal thoughts and preparations", "0 = Enjoys life or takes it as it comes | 2 = Weary of life; only fleeting suicidal thoughts | 4 = Probably better off dead; suicidal thoughts are common and suicide is considered a possible solution, but without specific plans | 6 = Explicit plans for suicide when there is an opportunity; active preparations"),
]

HAMA = [
    ("Anxious mood", "Worries, anticipation of the worst, fearful anticipation, irritability"),
    ("Tension", "Feelings of tension, fatigability, startle response, moved to tears easily, trembling, feelings of restlessness, inability to relax"),
    ("Fears", "Of dark, of strangers, of being left alone, of animals, of traffic, of crowds"),
    ("Insomnia", "Difficulty in falling asleep, broken sleep, unsatisfying sleep and fatigue on waking, dreams, nightmares, night terrors"),
    ("Intellectual", "Difficulty in concentration, poor memory"),
    ("Depressed mood", "Loss of interest, lack of pleasure in hobbies, depression, early waking, diurnal swing"),
    ("Somatic (muscular)", "Pains and aches, twitching, stiffness, myoclonic jerks, grinding of teeth, unsteady voice, increased muscular tone"),
    ("Somatic (sensory)", "Tinnitus, blurring of vision, hot and cold flushes, feelings of weakness, pricking sensation"),
    ("Cardiovascular symptoms", "Tachycardia, palpitations, pain in chest, throbbing of vessels, fainting feelings, missing beat"),
    ("Respiratory symptoms", "Pressure or constriction in chest, choking feelings, sighing, dyspnea"),
    ("Gastrointestinal symptoms", "Difficulty in swallowing, wind, abdominal pain, burning sensations, abdominal fullness, nausea, vomiting, borborygmi, looseness of bowels, loss of weight, constipation"),
    ("Genitourinary symptoms", "Frequency or urgency of micturition, amenorrhea, menorrhagia, frigidity, premature ejaculation, loss of libido, impotence"),
    ("Autonomic symptoms", "Dry mouth, flushing, pallor, tendency to sweat, giddiness, tension headache, raising of hair"),
    ("Behaviour at interview", "Fidgeting, restlessness or pacing, tremor of hands, furrowed brow, strained face, sighing or rapid respiration, facial pallor, swallowing"),
]
HAMA_ANCHORS = "0 = Not present | 1 = Mild | 2 = Moderate | 3 = Severe | 4 = Very severe"


def items(instr):
    """Return list of (item_no, name, max, text) for an instrument."""
    if instr in ("PHQ-9", "PHQ-8"):
        n = 9 if instr == "PHQ-9" else 8
        return [(i + 1, a, 3, f"{b}. {PHQ_ANCHORS}") for i, (a, b) in enumerate(PHQ9[:n])]
    if instr == "HAM-D":
        return [(i + 1, a, m, anc) for i, (a, m, anc) in enumerate(HAMD)]
    if instr == "MADRS":
        return [(i + 1, a, 6, f"{d}. {anc} (intermediate scores 1, 3, 5 permitted)")
                for i, (a, d, anc) in enumerate(MADRS)]
    if instr == "HAM-A":
        return [(i + 1, a, 4, f"{d}. {HAMA_ANCHORS}") for i, (a, d) in enumerate(HAMA)]
    raise KeyError(instr)


INTRO = {
    "PHQ-9": "You are a trained psychiatric rater. Your task is to score the Patient Health Questionnaire-9 (PHQ-9), which covers the last two weeks,",
    "PHQ-8": "You are a trained psychiatric rater. Your task is to score the Patient Health Questionnaire-8 (PHQ-8), which covers the last two weeks,",
    "HAM-D": "You are an expert psychiatric rater trained in the Hamilton Depression Rating Scale (HAM-D, 17 items). Your task is to score each item",
    "MADRS": "You are an expert psychiatric rater trained in the Montgomery-Asberg Depression Rating Scale (MADRS). Your task is to score each of the 10 items",
    "HAM-A": "You are a trained rater for the Hamilton Anxiety Rating Scale (HAM-A). Your task is to score each of the 14 items",
}


def build_prompt(instr, transcript):
    its = items(instr)
    lines = [f"{INTRO[instr]} based solely on information contained in the clinical interview transcript below.",
             "The transcript may be in Hindi, English or a mixture, and was produced by automatic speech recognition, so it contains recognition errors.",
             "For each item, provide:",
             "(1) score: an integer from 0 to the item maximum, using the anchor definitions",
             "(2) evidence_tag: E (explicit: the transcript directly states information sufficient to score the item), I (inferred: derived from contextual signals without a direct statement) or A (absent: insufficient information; a conservative default was applied)",
             "(3) confidence: your certainty that the score is correct, from 0.0 to 1.0",
             "(4) rationale: 1-2 sentences citing the evidence (a short quote) or the inference path",
             "", "ITEMS:"]
    for no, name, mx, text in its:
        lines.append(f"Item {no} - {name} (0-{mx}): {text}")
    lines += ["", "TRANSCRIPT:", transcript, "",
              f"Respond ONLY with a valid JSON array of {len(its)} objects, one per item in order, "
              "each with keys: item_id, score, evidence_tag, confidence, rationale."]
    return "\n".join(lines)


def schema(instr):
    its = items(instr)
    return {"type": "array", "minItems": len(its), "maxItems": len(its),
            "items": {"type": "object",
                      "properties": {"item_id": {"type": "integer"},
                                     "score": {"type": "integer", "minimum": 0, "maximum": max(m for _, _, m, _ in its)},
                                     "evidence_tag": {"type": "string", "enum": ["E", "I", "A"]},
                                     "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                     "rationale": {"type": "string", "maxLength": 400}},
                      "required": ["item_id", "score", "evidence_tag", "confidence", "rationale"]}}
