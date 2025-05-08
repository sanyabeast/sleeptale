import lmstudio as lms
from pydantic import BaseModel
import random

class StoryChunk(BaseModel):
    sentences: list[str]
    short_summary: str
    estimated_duration: int  # in minutes

class Topics(BaseModel):
    topics: list[str]

model = lms.llm("gemma-3-4b-it-qat", config={
    "temperature": 0.7,
    "top_p": 0.9
})

def generate_topics():
    topics = model.respond(
        """
        You are a creative assistant helping to generate bedtime stories.

        Task:
        - Provide exactly 5 calm, relaxing, slightly boring topics suitable for long, sleep-inducing stories.
        - Avoid topics that are too exciting, dramatic, or emotional.
        - Example topics: clouds drifting, quiet forests, soft rain on windows.
        - Format response strictly as JSON: { "topics": [ "topic1", "topic2", "topic3", "topic4", "topic5" ] }
        """,
        response_format=Topics
    )
    return topics.parsed["topics"]

def generate_story(topic, target_duration):
    chunks = []
    sentences_per_chunk = 10
    total_duration = 0
    all_sentences = []
    cumulative_summary = ""

    print(f"🛌 Generating story on topic: {topic}")

    while total_duration < target_duration:
        # Take last few sentences as context (up to 5)
        last_sentences = all_sentences[-5:] if len(all_sentences) >= 5 else all_sentences
        context_snippet = " ".join(last_sentences) or cumulative_summary or topic

        prompt = f"""
        You are generating the next part of a continuous, calm, sleep-inducing story.

        Instructions:
        - Continue where the story left off, using the following as context: "{context_snippet}"
        - Write {sentences_per_chunk} **new** slow-paced, non-exciting sentences.
        - Focus on gentle descriptions, minor environmental changes, and mild progression.
        - Do NOT repeat or restart earlier details.
        - Imagine the reader is drifting off to sleep, so avoid action or surprises.
        - Estimate the time (in minutes) to read this new chunk slowly.
        - Provide a brief summary (1–2 sentences) of only this **new** section.

        Response format (strict JSON):
        {{
            "sentences": ["sentence1", "sentence2", "..."],
            "short_summary": "summary here",
            "estimated_duration": integer_minutes
        }}
        """

        chunk = model.respond(prompt, response_format=StoryChunk)

        chunks.append(chunk.parsed)
        cumulative_summary += " " + chunk.parsed["short_summary"]
        total_duration += chunk.parsed["estimated_duration"]
        all_sentences.extend(chunk.parsed["sentences"])

        print(f"🧩 Preview: {chunk.parsed['sentences'][0]}")
        print(f"⏳ Current total duration: {total_duration} minutes")

    return {
        "topic": topic,
        "sentences": all_sentences,
        "chunks": chunks,
        "total_duration": total_duration
    }

if __name__ == "__main__":
    topics = generate_topics()
    print(f"🌙 Suggested topics: {topics}")

    selected_topic = random.choice(topics)
    print(f"🎯 Selected topic: {selected_topic}")

    final_story = generate_story(selected_topic, 60)
    print(f"✅ Generated story with {len(final_story['sentences'])} sentences, total ~{final_story['total_duration']} minutes")

    model.unload()
